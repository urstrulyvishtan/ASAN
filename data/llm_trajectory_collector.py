"""
LLM Trajectory Collector for ASAN

Captures temporal sequences of LLM internal states during inference:
- Attention weights at each layer over time
- Hidden state activations
- Token probability distributions
- Layer-wise gradient flows (optional)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict
import time
import json
from pathlib import Path


class LLMTrajectoryCollector:
    """
    Collect temporal sequences of LLM internal states during inference
    
    What we capture:
    - Attention weights at each layer over time
    - Hidden state activations
    - Token probability distributions
    - Layer-wise gradient flows (optional)
    """
    
    def __init__(self, model, layers_to_monitor='all', device='cuda'):
        """
        Initialize collector with hooks into model layers
        
        Args:
            model: HuggingFace model or custom PyTorch model
            layers_to_monitor: 'all', list of layer indices, or 'key_layers'
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.layers_to_monitor = layers_to_monitor
        self.model.to(device)
        
        # Storage for trajectories
        self.attention_trajectories = []
        self.hidden_state_trajectories = []
        self.token_prob_trajectories = []
        self.generated_tokens = []
        self.generation_timestamps = []
        
        # Hook storage
        self.hooks = []
        self.layer_names = []
        
        # Current generation state
        self.current_generation = {
            'attention_patterns': [],
            'hidden_states': [],
            'token_probs': [],
            'tokens': [],
            'timestamps': []
        }
        
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture internal states"""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style model
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA style model
            layers = self.model.model.layers
        else:
            # Generic model - try to find transformer layers
            layers = self._find_transformer_layers()
            
        if layers is None:
            raise ValueError("Could not find transformer layers in model")
            
        # Determine which layers to monitor
        if self.layers_to_monitor == 'all':
            layer_indices = list(range(len(layers)))
        elif self.layers_to_monitor == 'key_layers':
            # Monitor early, middle, and late layers
            layer_indices = [0, len(layers)//2, len(layers)-1]
        else:
            layer_indices = self.layers_to_monitor
            
        # Register hooks for each selected layer
        for i in layer_indices:
            if i < len(layers):
                hook = layers[i].register_forward_hook(
                    self._create_layer_hook(i)
                )
                self.hooks.append(hook)
                self.layer_names.append(f'layer_{i}')
                
    def _find_transformer_layers(self):
        """Find transformer layers in various model architectures"""
        for attr_name in ['transformer', 'model', 'encoder', 'decoder']:
            if hasattr(self.model, attr_name):
                module = getattr(self.model, attr_name)
                for layer_attr in ['h', 'layers', 'layer']:
                    if hasattr(module, layer_attr):
                        return getattr(module, layer_attr)
        return None
        
    def _create_layer_hook(self, layer_idx):
        """Create a hook function for a specific layer"""
        def hook_fn(module, input, output):
            # Extract attention weights if available
            attention_weights = None
            if hasattr(module, 'attn') and hasattr(module.attn, 'attention_probs'):
                attention_weights = module.attn.attention_probs
            elif hasattr(output, 'attentions') and output.attentions is not None:
                attention_weights = output.attentions
                
            # Extract hidden states
            hidden_states = None
            if isinstance(output, tuple):
                hidden_states = output[0]  # First element is usually hidden states
            elif hasattr(output, 'last_hidden_state'):
                hidden_states = output.last_hidden_state
            else:
                hidden_states = output
                
            # Store in current generation
            timestamp = time.time()
            
            if attention_weights is not None:
                self.current_generation['attention_patterns'].append({
                    'layer': layer_idx,
                    'weights': attention_weights.detach().cpu(),
                    'timestamp': timestamp
                })
                
            if hidden_states is not None:
                self.current_generation['hidden_states'].append({
                    'layer': layer_idx,
                    'states': hidden_states.detach().cpu(),
                    'timestamp': timestamp
                })
                
        return hook_fn
        
    def collect_during_generation(self, input_ids, max_length=50, temperature=1.0, 
                                do_sample=True, pad_token_id=None):
        """
        Collect trajectories during model generation
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            
        Returns:
            trajectories: dict with keys:
                - 'attention_patterns': [num_layers, num_heads, seq_len, seq_len] over time
                - 'hidden_states': [num_layers, seq_len, hidden_dim] over time
                - 'token_probs': [seq_len, vocab_size] over time
                - 'generated_tokens': list of token ids
                - 'generation_timestamps': list of timestamps
        """
        self.model.eval()
        self.current_generation = {
            'attention_patterns': [],
            'hidden_states': [],
            'token_probs': [],
            'tokens': [],
            'timestamps': []
        }
        
        generated_tokens = []
        timestamps = []
        
        with torch.no_grad():
            # Initial forward pass
            current_ids = input_ids.clone()
            start_time = time.time()
            
            for step in range(max_length):
                step_start = time.time()
                
                # Forward pass
                outputs = self.model(current_ids)
                logits = outputs.logits
                
                # Get token probabilities
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                self.current_generation['token_probs'].append({
                    'step': step,
                    'probs': probs.detach().cpu(),
                    'timestamp': step_start
                })
                
                # Sample next token
                if do_sample:
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                    
                # Check for end token
                if pad_token_id is not None and next_token.item() == pad_token_id:
                    break
                    
                generated_tokens.append(next_token.item())
                timestamps.append(time.time())
                
                # Update current_ids for next iteration
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Limit context length to prevent memory issues
                if current_ids.size(-1) > 1024:
                    current_ids = current_ids[:, -512:]
                    
        # Organize collected data
        trajectories = self._organize_trajectory_data(generated_tokens, timestamps)
        
        return trajectories
        
    def _organize_trajectory_data(self, generated_tokens, timestamps):
        """Organize collected data into structured format"""
        trajectories = {
            'attention_patterns': defaultdict(list),
            'hidden_states': defaultdict(list),
            'token_probs': [],
            'generated_tokens': generated_tokens,
            'generation_timestamps': timestamps
        }
        
        # Organize attention patterns by layer
        for attn_data in self.current_generation['attention_patterns']:
            layer_idx = attn_data['layer']
            trajectories['attention_patterns'][layer_idx].append(attn_data['weights'])
            
        # Organize hidden states by layer
        for hidden_data in self.current_generation['hidden_states']:
            layer_idx = hidden_data['layer']
            trajectories['hidden_states'][layer_idx].append(hidden_data['states'])
            
        # Organize token probabilities
        for prob_data in self.current_generation['token_probs']:
            trajectories['token_probs'].append(prob_data['probs'])
            
        return trajectories
        
    def extract_temporal_features(self, trajectories):
        """
        Convert raw trajectories into temporal sequences for ASAN
        
        Extract:
        - Attention entropy over time (high entropy = uncertainty)
        - Hidden state magnitude changes (sudden spikes = regime shift)
        - Token probability concentration (low concentration = hallucination signal)
        - Cross-layer attention flow patterns
        
        Args:
            trajectories: Organized trajectory data
            
        Returns:
            temporal_features: [num_timesteps, feature_dim]
        """
        features = []
        
        # Extract attention entropy over time
        attention_entropies = self._compute_attention_entropy(trajectories['attention_patterns'])
        if attention_entropies:
            features.append(attention_entropies)
            
        # Extract hidden state magnitude changes
        hidden_state_features = self._compute_hidden_state_features(trajectories['hidden_states'])
        if hidden_state_features:
            features.append(hidden_state_features)
            
        # Extract token probability features
        token_prob_features = self._compute_token_prob_features(trajectories['token_probs'])
        if token_prob_features:
            features.append(token_prob_features)
            
        # Combine all features
        if features:
            temporal_features = torch.cat(features, dim=-1)
        else:
            # Fallback: create dummy features
            num_timesteps = len(trajectories['generated_tokens'])
            temporal_features = torch.zeros(num_timesteps, 10)
            
        return temporal_features
        
    def _compute_attention_entropy(self, attention_patterns):
        """Compute attention entropy for each timestep"""
        if not attention_patterns:
            return None
            
        entropies = []
        for layer_idx in sorted(attention_patterns.keys()):
            layer_entropies = []
            for attn_weights in attention_patterns[layer_idx]:
                # Compute entropy across attention heads
                # attn_weights: [batch, num_heads, seq_len, seq_len]
                if attn_weights.dim() == 4:
                    # Average across heads
                    avg_attn = attn_weights.mean(dim=1)  # [batch, seq_len, seq_len]
                    # Compute entropy for each position
                    entropy = -torch.sum(avg_attn * torch.log(avg_attn + 1e-10), dim=-1)
                    layer_entropies.append(entropy.mean().item())
                else:
                    layer_entropies.append(0.0)
            entropies.append(layer_entropies)
            
        if entropies:
            # Stack and transpose to get [timesteps, layers]
            return torch.tensor(entropies).T
        return None
        
    def _compute_hidden_state_features(self, hidden_states):
        """Compute hidden state magnitude and change features"""
        if not hidden_states:
            return None
            
        features = []
        for layer_idx in sorted(hidden_states.keys()):
            layer_features = []
            prev_states = None
            
            for states in hidden_states[layer_idx]:
                # Compute magnitude
                magnitude = torch.norm(states, dim=-1).mean().item()
                layer_features.append(magnitude)
                
                # Compute change from previous timestep
                if prev_states is not None:
                    change = torch.norm(states - prev_states, dim=-1).mean().item()
                    layer_features.append(change)
                else:
                    layer_features.append(0.0)
                    
                prev_states = states
                
            features.append(layer_features)
            
        if features:
            return torch.tensor(features).T
        return None
        
    def _compute_token_prob_features(self, token_probs):
        """Compute token probability concentration features"""
        if not token_probs:
            return None
            
        features = []
        for probs in token_probs:
            # Compute entropy (concentration measure)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            features.append(entropy.mean().item())
            
            # Compute top-k probability mass
            top_k_probs = torch.topk(probs, k=5, dim=-1)[0]
            top_k_mass = top_k_probs.sum(dim=-1)
            features.append(top_k_mass.mean().item())
            
        return torch.tensor(features).unsqueeze(-1)
        
    def save_trajectory(self, trajectory, label, metadata, save_path):
        """
        Save trajectory with label (safe/harmful) and metadata
        
        Args:
            trajectory: Collected trajectory data
            label: 'safe' or 'harmful'
            metadata: dict with additional information
            save_path: Path to save the trajectory
        """
        save_data = {
            'trajectory': trajectory,
            'label': label,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        # Convert tensors to lists for JSON serialization
        def tensor_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [tensor_to_list(item) for item in obj]
            else:
                return obj
                
        save_data = tensor_to_list(save_data)
        
        # Save to file
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
    def load_trajectory(self, load_path):
        """Load a saved trajectory"""
        with open(load_path, 'r') as f:
            data = json.load(f)
            
        # Convert lists back to tensors
        def list_to_tensor(obj):
            if isinstance(obj, list) and len(obj) > 0:
                if isinstance(obj[0], (int, float)):
                    return torch.tensor(obj)
                elif isinstance(obj[0], list):
                    return torch.tensor(obj)
            elif isinstance(obj, dict):
                return {k: list_to_tensor(v) for k, v in obj.items()}
            return obj
            
        data['trajectory'] = list_to_tensor(data['trajectory'])
        return data
        
    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def __del__(self):
        """Cleanup hooks when object is destroyed"""
        self.cleanup_hooks()
