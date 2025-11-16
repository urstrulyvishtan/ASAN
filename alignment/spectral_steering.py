"""
Spectral Steering Mechanism

Use ASAN's spectral analysis to steer LLM generation in real-time by modifying
internal states to avoid harmful trajectories while preserving output quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

try:
    from models.asan_predictor import ASANPredictor
except ImportError:
    from ..models.asan_predictor import ASANPredictor


@dataclass
class SteeringConfig:
    """Configuration for spectral steering"""
    steering_strength: float = 0.3
    min_harm_prob_threshold: float = 0.5
    max_steering_magnitude: float = 0.5
    smoothing_window: int = 3
    preserve_semantic_coherence: bool = True
    adaptive_strength: bool = True


class SpectralSteeringController:
    """
    Use ASAN's spectral analysis to steer LLM generation in real-time
    
    Key Insight:
    - ASAN identifies which frequency bands contain harmful signatures
    - We can modify those specific frequency components to "correct" trajectory
    - This preserves useful content while removing harmful patterns
    """
    
    def __init__(self, asan_predictor: ASANPredictor, config: SteeringConfig):
        """
        Args:
            asan_predictor: Trained ASAN model
            config: Steering configuration
        """
        self.asan_predictor = asan_predictor
        self.config = config
        self.asan_predictor.eval()  # Use in eval mode for inference
        
        # Learned safe trajectory embeddings (can be populated from training)
        self.safe_trajectory_embeddings = None
        
    def compute_steering_vector(self, 
                               current_trajectory: Dict[str, Any],
                               target_safety_level: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Compute how to modify model states to achieve target safety
        
        Process:
        1. Run ASAN to get spectral decomposition
        2. Identify frequency bands with harmful signatures
        3. Compute "safe direction" in those frequency bands
        4. Generate steering vector that pushes trajectory toward safety
        
        Returns:
            steering_vector: Modifications to apply to hidden states
            frequency_band_corrections: Which bands need correction
            confidence: How confident we are in steering direction
        """
        with torch.no_grad():
            # Get ASAN prediction and spectral analysis
            asan_output = self.asan_predictor(
                current_trajectory['attention_patterns'],
                current_trajectory['hidden_states'],
                current_trajectory['token_probs']
            )
            
            harm_prob = asan_output['harm_probability'].item()
            confidence = asan_output['confidence'].item()
            wavelet_coeffs = asan_output.get('wavelet_coefficients', {})
            encoded_rep = asan_output['encoded_representation']
            
            # Identify problematic frequency bands
            frequency_contributions = asan_output['frequency_contributions']
            problematic_bands = self._identify_problematic_bands(
                frequency_contributions, harm_prob, wavelet_coeffs
            )
            
            # Compute steering vector
            steering_vector = self._compute_steering_from_spectral(
                encoded_rep, wavelet_coeffs, problematic_bands, 
                harm_prob, target_safety_level
            )
            
            # Compute frequency band corrections
            frequency_band_corrections = self._compute_band_corrections(
                wavelet_coeffs, problematic_bands, harm_prob
            )
            
        return {
            'steering_vector': steering_vector,
            'frequency_band_corrections': frequency_band_corrections,
            'confidence': confidence,
            'harm_probability': harm_prob,
            'problematic_bands': problematic_bands
        }
    
    def _identify_problematic_bands(self, 
                                   frequency_contributions: torch.Tensor,
                                   harm_prob: float,
                                   wavelet_coeffs: Dict[str, torch.Tensor]) -> List[str]:
        """Identify which frequency bands contribute most to harmful pattern"""
        # Bands with high contribution to harm prediction
        band_names = sorted(wavelet_coeffs.keys())
        contributions_list = frequency_contributions.cpu().numpy()
        
        # Identify bands with above-average contribution
        mean_contribution = contributions_list.mean()
        problematic = []
        
        for i, band_name in enumerate(band_names):
            if contributions_list[i] > mean_contribution * 1.2 and harm_prob > 0.5:
                problematic.append(band_name)
        
        return problematic
    
    def _compute_steering_from_spectral(self,
                                       encoded_rep: torch.Tensor,
                                       wavelet_coeffs: Dict[str, torch.Tensor],
                                       problematic_bands: List[str],
                                       harm_prob: float,
                                       target_safety: float) -> torch.Tensor:
        """Compute steering vector from spectral analysis"""
        # Get encoded representation shape
        batch_size, encoding_dim = encoded_rep.shape
        
        # Initialize steering vector
        steering_vector = torch.zeros_like(encoded_rep)
        
        if len(problematic_bands) == 0 or harm_prob < self.config.min_harm_prob_threshold:
            return steering_vector
        
        # Compute safe direction
        # Strategy: Push toward regions with lower harm probability
        # This is a simplified version - in practice, could use learned safe embeddings
        
        # Compute gradient direction that would reduce harm probability
        # Approximate by pushing toward lower-energy regions in problematic bands
        for band_name in problematic_bands:
            if band_name in wavelet_coeffs:
                band_coeffs = wavelet_coeffs[band_name]  # [batch, time, features]
                
                # Compute correction: reduce energy in problematic frequency band
                band_energy = torch.mean(band_coeffs ** 2, dim=1)  # [batch, features]
                correction = -band_energy * (harm_prob - target_safety)
                
                # Project correction back to encoding space
                # Simple linear projection (could be learned)
                if band_energy.shape[1] == encoding_dim:
                    steering_vector += correction * 0.1
        
        # Normalize steering vector to prevent extreme modifications
        steering_magnitude = torch.norm(steering_vector)
        if steering_magnitude > self.config.max_steering_magnitude:
            steering_vector = steering_vector / steering_magnitude * self.config.max_steering_magnitude
        
        return steering_vector
    
    def _compute_band_corrections(self,
                                 wavelet_coeffs: Dict[str, torch.Tensor],
                                 problematic_bands: List[str],
                                 harm_prob: float) -> Dict[str, torch.Tensor]:
        """Compute corrections for each frequency band"""
        corrections = {}
        
        for band_name in problematic_bands:
            if band_name in wavelet_coeffs:
                coeffs = wavelet_coeffs[band_name]
                
                # Correction: dampen coefficients in problematic band
                correction_strength = (harm_prob - 0.1) * self.config.steering_strength
                corrections[band_name] = -coeffs * correction_strength
        
        return corrections
    
    def apply_steering(self, 
                      hidden_states: torch.Tensor,
                      steering_vector: torch.Tensor,
                      layer_idx: int) -> Tuple[torch.Tensor, float]:
        """
        Apply steering vector to model's hidden states at specific layer
        
        Key considerations:
        - Apply gradually (not sudden jumps)
        - Preserve semantic coherence
        - Monitor for over-steering (output becomes nonsensical)
        
        Returns:
            modified_hidden_states: Steered hidden states
            steering_magnitude: How much we modified
        """
        # Compute adaptive steering strength
        adaptive_strength = self.adaptive_steering_strength(
            steering_vector,
            self.config.steering_strength
        )
        
        # Apply steering with smoothing
        if hidden_states.dim() == 2:  # [seq_len, hidden_dim]
            # Project steering vector to hidden state space
            # In practice, would need learned projection matrix
            # For now, use simple interpolation if dimensions match
            if steering_vector.shape[-1] == hidden_states.shape[-1]:
                # Mean pool steering vector across sequence length
                steering_applied = steering_vector.mean(dim=0) if steering_vector.dim() > 1 else steering_vector
                
                # Apply with adaptive strength
                modified_hidden_states = hidden_states + steering_applied * adaptive_strength
            else:
                # Dimensions don't match - no steering applied
                modified_hidden_states = hidden_states
                adaptive_strength = 0.0
        elif hidden_states.dim() == 3:  # [batch, seq_len, hidden_dim]
            if steering_vector.shape[-1] == hidden_states.shape[-1]:
                steering_applied = steering_vector.mean(dim=0) if steering_vector.dim() > 1 else steering_vector
                modified_hidden_states = hidden_states + steering_applied.unsqueeze(0) * adaptive_strength
            else:
                modified_hidden_states = hidden_states
                adaptive_strength = 0.0
        else:
            modified_hidden_states = hidden_states
            adaptive_strength = 0.0
        
        # Clip to prevent extreme values
        if self.config.preserve_semantic_coherence:
            # Preserve relative magnitudes
            original_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
            modified_norm = torch.norm(modified_hidden_states, dim=-1, keepdim=True)
            # Scale to preserve norm if change is too large
            max_change_ratio = 1.2
            scale_factor = torch.clamp(modified_norm / (original_norm + 1e-8), 
                                      1.0 / max_change_ratio, max_change_ratio)
            modified_hidden_states = modified_hidden_states / (scale_factor + 1e-8) * original_norm
        
        steering_magnitude = torch.norm(steering_vector).item() * adaptive_strength
        
        return modified_hidden_states, steering_magnitude
    
    def adaptive_steering_strength(self, 
                                  steering_vector: torch.Tensor,
                                  base_strength: float) -> float:
        """
        Adjust steering strength based on ASAN predictions and steering vector
        
        Strategy:
        - Low harm prob + high confidence: minimal steering
        - High harm prob + high confidence: aggressive steering
        - Low/high harm prob + low confidence: conservative steering
        """
        if not self.config.adaptive_strength:
            return base_strength
        
        # Base strength
        strength = base_strength
        
        # Adjust based on steering vector magnitude (confidence in correction)
        steering_magnitude = torch.norm(steering_vector).item()
        if steering_magnitude > 0:
            # More confident in correction -> higher strength
            strength *= min(1.5, 0.5 + steering_magnitude * 2)
        
        return strength
    
    def compute_safe_direction(self, current_trajectory: Dict[str, Any]) -> torch.Tensor:
        """
        Compute direction toward safer trajectory using learned safe embeddings
        
        If safe trajectory embeddings are available, use them to guide steering
        """
        if self.safe_trajectory_embeddings is None:
            # Fallback: use spectral analysis
            steering_result = self.compute_steering_vector(current_trajectory)
            return steering_result['steering_vector']
        
        # Compute similarity to safe trajectories
        with torch.no_grad():
            asan_output = self.asan_predictor(
                current_trajectory['attention_patterns'],
                current_trajectory['hidden_states'],
                current_trajectory['token_probs']
            )
            current_embedding = asan_output['encoded_representation']
            
            # Find nearest safe trajectory
            similarities = torch.cosine_similarity(
                current_embedding.unsqueeze(0),
                self.safe_trajectory_embeddings,
                dim=1
            )
            nearest_safe_idx = similarities.argmax()
            nearest_safe = self.safe_trajectory_embeddings[nearest_safe_idx]
            
            # Direction toward safe trajectory
            safe_direction = nearest_safe - current_embedding.squeeze(0)
            safe_direction = safe_direction / (torch.norm(safe_direction) + 1e-8)
            
        return safe_direction


class InferenceTimeSteering:
    """
    Integration of ASAN steering with LLM generation
    
    Hooks into model generation to intercept and modify hidden states
    """
    
    def __init__(self, 
                 llm_model: Any,
                 asan_predictor: ASANPredictor,
                 steering_controller: SpectralSteeringController):
        """
        Set up hooks to intercept and modify hidden states during generation
        
        Args:
            llm_model: The LLM to generate with
            asan_predictor: Trained ASAN model
            steering_controller: Steering controller
        """
        self.llm_model = llm_model
        self.asan_predictor = asan_predictor
        self.steering_controller = steering_controller
        
        # Hooks for intercepting hidden states
        self.hooks = []
        self.current_trajectory = {
            'attention_patterns': {},
            'hidden_states': {},
            'token_probs': []
        }
        self.steering_log = []
        
    def _register_hooks(self):
        """Register forward hooks to intercept hidden states"""
        # This is model-specific - would need to adapt based on model architecture
        # Example for transformer models:
        if hasattr(self.llm_model, 'transformer'):
            for layer_idx, layer in enumerate(self.llm_model.transformer.h):
                hook_handle = layer.register_forward_hook(
                    self._create_hook(layer_idx)
                )
                self.hooks.append(hook_handle)
    
    def _create_hook(self, layer_idx: int):
        """Create hook function for specific layer"""
        def hook_fn(module, input, output):
            # Extract hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store for ASAN analysis
            if layer_idx not in self.current_trajectory['hidden_states']:
                self.current_trajectory['hidden_states'][layer_idx] = []
            
            # Clone to avoid modifying original
            self.current_trajectory['hidden_states'][layer_idx].append(
                hidden_states.clone()
            )
        return hook_fn
    
    def _unregister_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_with_steering(self, 
                              prompt: str,
                              max_length: int = 50,
                              harm_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate text with real-time ASAN-guided steering
        
        Generation loop:
        1. Model computes next token logits
        2. Extract hidden states before final layer
        3. Run ASAN prediction on current trajectory
        4. If harm_prob > threshold:
           - Compute steering vector
           - Modify hidden states
           - Recompute token logits with steered states
        5. Sample token and continue
        
        Returns:
            generated_text: text with steering applied
            steering_log: record of when/how steering occurred
            safety_metrics: harm probabilities over time
        """
        # Reset trajectory tracking
        self.current_trajectory = {
            'attention_patterns': {},
            'hidden_states': {},
            'token_probs': []
        }
        self.steering_log = []
        safety_metrics = []
        
        # Register hooks
        self._register_hooks()
        
        try:
            # Generate with steering
            generated_tokens = []
            current_input = prompt
            
            for step in range(max_length):
                # Forward pass (hooks will capture hidden states)
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        current_input,
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_hidden_states=True
                    )
                    
                    # Get logits and hidden states
                    logits = outputs.scores[0] if 'scores' in outputs else None
                    next_token_logits = logits[0] if logits is not None else None
                    
                    # Store token probabilities
                    if next_token_logits is not None:
                        token_probs = F.softmax(next_token_logits, dim=-1)
                        self.current_trajectory['token_probs'].append(token_probs)
                    
                    # Run ASAN prediction on current trajectory
                    if len(self.current_trajectory['token_probs']) >= 2:
                        asan_result = self.steering_controller.compute_steering_vector(
                            self.current_trajectory
                        )
                        
                        harm_prob = asan_result['harm_probability']
                        safety_metrics.append({
                            'step': step,
                            'harm_probability': harm_prob,
                            'confidence': asan_result['confidence']
                        })
                        
                        # Apply steering if needed
                        if harm_prob > harm_threshold:
                            # Get latest hidden states
                            latest_layer = max(self.current_trajectory['hidden_states'].keys())
                            if latest_layer in self.current_trajectory['hidden_states']:
                                latest_hidden = self.current_trajectory['hidden_states'][latest_layer][-1]
                                
                                # Apply steering
                                steering_vector = asan_result['steering_vector']
                                modified_hidden, steering_mag = self.steering_controller.apply_steering(
                                    latest_hidden,
                                    steering_vector,
                                    latest_layer
                                )
                                
                                # Log steering action
                                self.steering_log.append({
                                    'step': step,
                                    'harm_probability': harm_prob,
                                    'steering_magnitude': steering_mag,
                                    'layer': latest_layer,
                                    'problematic_bands': asan_result['problematic_bands']
                                })
                                
                                # Update hidden states (simplified - full implementation would need model-specific code)
                                # self.current_trajectory['hidden_states'][latest_layer][-1] = modified_hidden
                    
                    # Sample next token
                    if next_token_logits is not None:
                        next_token_id = torch.multinomial(
                            F.softmax(next_token_logits, dim=-1), 
                            num_samples=1
                        ).item()
                        generated_tokens.append(next_token_id)
                        
                        # Update input for next iteration
                        current_input += f" {next_token_id}"  # Simplified
                    else:
                        break
        finally:
            # Always unregister hooks
            self._unregister_hooks()
        
        # Decode generated tokens (simplified - would use model's tokenizer)
        generated_text = f"{prompt} {' '.join(map(str, generated_tokens))}"
        
        return {
            'generated_text': generated_text,
            'steering_log': self.steering_log,
            'safety_metrics': safety_metrics,
            'trajectory': self.current_trajectory
        }
