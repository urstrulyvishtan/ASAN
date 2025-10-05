"""
Synthetic LLM Simulator for ASAN Testing

Creates simplified LLM behavior simulator for rapid experimentation without needing real LLMs.
This enables fast iteration during development with controlled ground truth.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    """Configuration for synthetic LLM simulator"""
    num_layers: int = 12
    num_heads: int = 8
    hidden_dim: int = 768
    vocab_size: int = 50257
    max_seq_len: int = 1024
    device: str = 'cuda'


class SyntheticLLMSimulator:
    """
    Simulate LLM internal states for testing ASAN without needing real LLMs
    
    Why synthetic data:
    - Fast iteration during development
    - Controlled ground truth for validation
    - Test edge cases systematically
    - No GPU requirements for initial testing
    """
    
    def __init__(self, config: SimulatorConfig):
        """Initialize simulator with realistic LLM architecture parameters"""
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize random generators for reproducibility
        self.rng = np.random.RandomState(42)
        torch.manual_seed(42)
        
        # Create synthetic model components
        self._create_attention_matrices()
        self._create_hidden_state_projections()
        self._create_token_probability_generators()
        
    def _create_attention_matrices(self):
        """Create synthetic attention weight generators"""
        self.attention_generators = {}
        
        for layer in range(self.config.num_layers):
            # Create attention patterns that vary by layer
            if layer < self.config.num_layers // 3:
                # Early layers: more local attention
                pattern_type = 'local'
            elif layer < 2 * self.config.num_layers // 3:
                # Middle layers: balanced attention
                pattern_type = 'balanced'
            else:
                # Late layers: more global attention
                pattern_type = 'global'
                
            self.attention_generators[layer] = self._create_attention_pattern(pattern_type)
            
    def _create_attention_pattern(self, pattern_type: str):
        """Create attention pattern generator for specific type"""
        def generator(seq_len: int, timestep: int):
            if pattern_type == 'local':
                # Local attention: focus on recent tokens
                attention = torch.zeros(seq_len, seq_len)
                for i in range(seq_len):
                    # Focus on last 3 tokens
                    start_idx = max(0, i - 2)
                    attention[i, start_idx:i+1] = 1.0 / (i - start_idx + 1)
                    
            elif pattern_type == 'global':
                # Global attention: more uniform distribution
                attention = torch.ones(seq_len, seq_len) / seq_len
                # Add some structure
                for i in range(seq_len):
                    attention[i, i] *= 2.0  # Self-attention boost
                    
            else:  # balanced
                # Balanced: mix of local and global
                attention = torch.zeros(seq_len, seq_len)
                for i in range(seq_len):
                    # Local component
                    local_start = max(0, i - 1)
                    attention[i, local_start:i+1] = 0.6 / (i - local_start + 1)
                    # Global component
                    attention[i, :] += 0.4 / seq_len
                    
            return attention
            
        return generator
        
    def _create_hidden_state_projections(self):
        """Create hidden state evolution patterns"""
        self.hidden_state_evolvers = {}
        
        for layer in range(self.config.num_layers):
            # Each layer has different evolution characteristics
            if layer < self.config.num_layers // 3:
                # Early layers: gradual evolution
                evolution_type = 'gradual'
            elif layer < 2 * self.config.num_layers // 3:
                # Middle layers: moderate changes
                evolution_type = 'moderate'
            else:
                # Late layers: can have sudden changes
                evolution_type = 'variable'
                
            self.hidden_state_evolvers[layer] = self._create_hidden_evolver(evolution_type)
            
    def _create_hidden_evolver(self, evolution_type: str):
        """Create hidden state evolution function"""
        def evolver(prev_states: torch.Tensor, timestep: int, seq_len: int):
            if prev_states is None:
                # Initialize with random states
                return torch.randn(seq_len, self.config.hidden_dim) * 0.1
                
            if evolution_type == 'gradual':
                # Gradual evolution with small changes
                change = torch.randn_like(prev_states) * 0.05
                return prev_states + change
                
            elif evolution_type == 'moderate':
                # Moderate changes
                change = torch.randn_like(prev_states) * 0.1
                return prev_states + change
                
            else:  # variable
                # Can have sudden changes
                if timestep % 5 == 0:  # Every 5th timestep
                    change = torch.randn_like(prev_states) * 0.3
                else:
                    change = torch.randn_like(prev_states) * 0.05
                return prev_states + change
                
        return evolver
        
    def _create_token_probability_generators(self):
        """Create token probability generators"""
        self.token_prob_generators = {
            'safe': self._create_safe_token_generator(),
            'harmful': self._create_harmful_token_generator(),
            'hallucination': self._create_hallucination_token_generator(),
            'bias': self._create_bias_token_generator()
        }
        
    def _create_safe_token_generator(self):
        """Create token probability generator for safe generation"""
        def generator(timestep: int, seq_len: int):
            # Safe generation: concentrated probabilities
            probs = torch.zeros(self.config.vocab_size)
            
            # Create a few high-probability tokens
            num_high_prob = 5
            high_prob_indices = torch.randperm(self.config.vocab_size)[:num_high_prob]
            high_prob_values = torch.softmax(torch.randn(num_high_prob), dim=0)
            
            probs[high_prob_indices] = high_prob_values * 0.8
            probs += torch.randn(self.config.vocab_size) * 0.01  # Small noise
            
            return torch.softmax(probs, dim=0)
            
        return generator
        
    def _create_harmful_token_generator(self):
        """Create token probability generator for harmful generation"""
        def generator(timestep: int, seq_len: int):
            # Harmful generation: broader distribution, less concentrated
            probs = torch.zeros(self.config.vocab_size)
            
            # Create many medium-probability tokens (uncertainty)
            num_medium_prob = 50
            medium_prob_indices = torch.randperm(self.config.vocab_size)[:num_medium_prob]
            medium_prob_values = torch.softmax(torch.randn(num_medium_prob), dim=0)
            
            probs[medium_prob_indices] = medium_prob_values * 0.3
            probs += torch.randn(self.config.vocab_size) * 0.05  # More noise
            
            return torch.softmax(probs, dim=0)
            
        return generator
        
    def _create_hallucination_token_generator(self):
        """Create token probability generator for hallucination"""
        def generator(timestep: int, seq_len: int):
            # Hallucination: very broad distribution
            probs = torch.randn(self.config.vocab_size) * 0.1
            return torch.softmax(probs, dim=0)
            
        return generator
        
    def _create_bias_token_generator(self):
        """Create token probability generator for bias amplification"""
        def generator(timestep: int, seq_len: int):
            # Bias: concentrated on specific token clusters
            probs = torch.zeros(self.config.vocab_size)
            
            # Create biased clusters
            cluster_size = 100
            num_clusters = 3
            
            for i in range(num_clusters):
                start_idx = i * cluster_size
                end_idx = min(start_idx + cluster_size, self.config.vocab_size)
                cluster_probs = torch.softmax(torch.randn(end_idx - start_idx), dim=0)
                probs[start_idx:end_idx] = cluster_probs * 0.2
                
            probs += torch.randn(self.config.vocab_size) * 0.01
            return torch.softmax(probs, dim=0)
            
        return generator
        
    def simulate_safe_trajectory(self, length: int = 20) -> Dict:
        """
        Simulate internal states for safe generation
        
        Characteristics:
        - Stable attention patterns
        - Gradual hidden state evolution
        - Concentrated token probabilities
        - Consistent cross-layer patterns
        """
        trajectory = {
            'attention_patterns': {},
            'hidden_states': {},
            'token_probs': [],
            'generated_tokens': [],
            'generation_timestamps': [],
            'label': 'safe'
        }
        
        # Generate token probabilities
        token_generator = self.token_prob_generators['safe']
        
        for timestep in range(length):
            # Generate attention patterns for each layer
            seq_len = min(timestep + 10, self.config.max_seq_len)
            
            for layer in range(self.config.num_layers):
                if layer not in trajectory['attention_patterns']:
                    trajectory['attention_patterns'][layer] = []
                    
                attention = self.attention_generators[layer](seq_len, timestep)
                trajectory['attention_patterns'][layer].append(attention)
                
            # Generate hidden states for each layer
            for layer in range(self.config.num_layers):
                if layer not in trajectory['hidden_states']:
                    trajectory['hidden_states'][layer] = []
                    prev_states = None
                else:
                    prev_states = trajectory['hidden_states'][layer][-1] if trajectory['hidden_states'][layer] else None
                    
                hidden_states = self.hidden_state_evolvers[layer](prev_states, timestep, seq_len)
                trajectory['hidden_states'][layer].append(hidden_states)
                
            # Generate token probabilities
            token_probs = token_generator(timestep, seq_len)
            trajectory['token_probs'].append(token_probs)
            
            # Sample token
            token = torch.multinomial(token_probs, 1).item()
            trajectory['generated_tokens'].append(token)
            trajectory['generation_timestamps'].append(timestep)
            
        return trajectory
        
    def simulate_harmful_trajectory(self, harm_type: str = 'jailbreak', length: int = 20) -> Dict:
        """
        Simulate internal states for harmful generation
        
        Args:
            harm_type: Type of harm ('jailbreak', 'hallucination', 'bias')
            length: Length of trajectory
            
        Characteristics by harm type:
        
        Jailbreak:
        - Sudden attention pattern shift at specific layer
        - Hidden state magnitude spike
        - Broadening token probability distribution
        
        Hallucination:
        - Increasing attention entropy over time
        - Weakening cross-layer coherence
        - Low-confidence token selection
        
        Bias amplification:
        - Specific attention head activation pattern
        - Clustering in hidden state subspace
        """
        trajectory = {
            'attention_patterns': {},
            'hidden_states': {},
            'token_probs': [],
            'generated_tokens': [],
            'generation_timestamps': [],
            'label': 'harmful',
            'harm_type': harm_type
        }
        
        # Choose appropriate token generator
        if harm_type == 'hallucination':
            token_generator = self.token_prob_generators['hallucination']
        elif harm_type == 'bias':
            token_generator = self.token_prob_generators['bias']
        else:  # jailbreak
            token_generator = self.token_prob_generators['harmful']
            
        # Determine failure point (when harmful pattern emerges)
        failure_point = length // 2
        
        for timestep in range(length):
            seq_len = min(timestep + 10, self.config.max_seq_len)
            
            # Generate attention patterns
            for layer in range(self.config.num_layers):
                if layer not in trajectory['attention_patterns']:
                    trajectory['attention_patterns'][layer] = []
                    
                if harm_type == 'jailbreak' and timestep >= failure_point and layer >= self.config.num_layers // 2:
                    # Sudden attention shift for jailbreak
                    attention = torch.rand(seq_len, seq_len)
                    attention = attention / attention.sum(dim=-1, keepdim=True)
                else:
                    attention = self.attention_generators[layer](seq_len, timestep)
                    
                trajectory['attention_patterns'][layer].append(attention)
                
            # Generate hidden states
            for layer in range(self.config.num_layers):
                if layer not in trajectory['hidden_states']:
                    trajectory['hidden_states'][layer] = []
                    prev_states = None
                else:
                    prev_states = trajectory['hidden_states'][layer][-1] if trajectory['hidden_states'][layer] else None
                    
                if harm_type == 'jailbreak' and timestep >= failure_point and layer >= self.config.num_layers // 2:
                    # Hidden state spike for jailbreak
                    hidden_states = torch.randn(seq_len, self.config.hidden_dim) * 0.5
                else:
                    hidden_states = self.hidden_state_evolvers[layer](prev_states, timestep, seq_len)
                    
                trajectory['hidden_states'][layer].append(hidden_states)
                
            # Generate token probabilities
            if timestep >= failure_point:
                # Switch to harmful token generation
                token_probs = token_generator(timestep, seq_len)
            else:
                # Start with safe generation
                token_probs = self.token_prob_generators['safe'](timestep, seq_len)
                
            trajectory['token_probs'].append(token_probs)
            
            # Sample token
            token = torch.multinomial(token_probs, 1).item()
            trajectory['generated_tokens'].append(token)
            trajectory['generation_timestamps'].append(timestep)
            
        return trajectory
        
    def inject_failure_signature(self, safe_trajectory: Dict, failure_point: int, 
                               harm_type: str) -> Dict:
        """
        Inject harmful pattern signature into safe trajectory at specific point
        
        This creates realistic transitions from safe to harmful behavior
        """
        harmful_trajectory = safe_trajectory.copy()
        harmful_trajectory['label'] = 'harmful'
        harmful_trajectory['harm_type'] = harm_type
        
        # Modify trajectory after failure point
        for timestep in range(failure_point, len(safe_trajectory['generated_tokens'])):
            seq_len = min(timestep + 10, self.config.max_seq_len)
            
            # Modify attention patterns
            for layer in range(self.config.num_layers):
                if layer >= self.config.num_layers // 2:  # Modify late layers
                    if harm_type == 'jailbreak':
                        # Sudden attention shift
                        attention = torch.rand(seq_len, seq_len)
                        attention = attention / attention.sum(dim=-1, keepdim=True)
                        harmful_trajectory['attention_patterns'][layer][timestep] = attention
                        
            # Modify hidden states
            for layer in range(self.config.num_layers):
                if layer >= self.config.num_layers // 2:
                    if harm_type == 'jailbreak':
                        # Hidden state spike
                        hidden_states = torch.randn(seq_len, self.config.hidden_dim) * 0.5
                        harmful_trajectory['hidden_states'][layer][timestep] = hidden_states
                        
            # Modify token probabilities
            if harm_type == 'hallucination':
                token_probs = self.token_prob_generators['hallucination'](timestep, seq_len)
            elif harm_type == 'bias':
                token_probs = self.token_prob_generators['bias'](timestep, seq_len)
            else:
                token_probs = self.token_prob_generators['harmful'](timestep, seq_len)
                
            harmful_trajectory['token_probs'][timestep] = token_probs
            
            # Resample token
            token = torch.multinomial(token_probs, 1).item()
            harmful_trajectory['generated_tokens'][timestep] = token
            
        return harmful_trajectory
        
    def generate_dataset(self, n_safe: int = 5000, n_harmful: int = 5000) -> List[Dict]:
        """Generate synthetic dataset for initial ASAN training"""
        dataset = []
        
        # Generate safe trajectories
        print(f"Generating {n_safe} safe trajectories...")
        for i in range(n_safe):
            length = self.rng.randint(10, 30)
            trajectory = self.simulate_safe_trajectory(length)
            trajectory['id'] = f'safe_{i}'
            dataset.append(trajectory)
            
        # Generate harmful trajectories
        print(f"Generating {n_harmful} harmful trajectories...")
        harm_types = ['jailbreak', 'hallucination', 'bias']
        
        for i in range(n_harmful):
            length = self.rng.randint(10, 30)
            harm_type = self.rng.choice(harm_types)
            trajectory = self.simulate_harmful_trajectory(harm_type, length)
            trajectory['id'] = f'harmful_{i}'
            dataset.append(trajectory)
            
        # Shuffle dataset
        self.rng.shuffle(dataset)
        
        return dataset
        
    def create_balanced_dataset(self, n_samples_per_class: int = 1000) -> Tuple[List[Dict], List[Dict]]:
        """Create balanced dataset with equal safe and harmful examples"""
        safe_trajectories = []
        harmful_trajectories = []
        
        # Generate safe trajectories
        for i in range(n_samples_per_class):
            length = self.rng.randint(10, 30)
            trajectory = self.simulate_safe_trajectory(length)
            trajectory['id'] = f'safe_{i}'
            safe_trajectories.append(trajectory)
            
        # Generate harmful trajectories
        harm_types = ['jailbreak', 'hallucination', 'bias']
        n_per_harm_type = n_samples_per_class // len(harm_types)
        
        for harm_type in harm_types:
            for i in range(n_per_harm_type):
                length = self.rng.randint(10, 30)
                trajectory = self.simulate_harmful_trajectory(harm_type, length)
                trajectory['id'] = f'harmful_{harm_type}_{i}'
                harmful_trajectories.append(trajectory)
                
        return safe_trajectories, harmful_trajectories
        
    def create_transition_dataset(self, n_samples: int = 1000) -> List[Dict]:
        """Create dataset with transitions from safe to harmful"""
        dataset = []
        
        for i in range(n_samples):
            # Start with safe trajectory
            length = self.rng.randint(15, 35)
            safe_trajectory = self.simulate_safe_trajectory(length)
            
            # Choose random failure point
            failure_point = self.rng.randint(length // 3, 2 * length // 3)
            
            # Choose random harm type
            harm_type = self.rng.choice(['jailbreak', 'hallucination', 'bias'])
            
            # Inject failure signature
            harmful_trajectory = self.inject_failure_signature(
                safe_trajectory, failure_point, harm_type
            )
            
            harmful_trajectory['id'] = f'transition_{i}'
            harmful_trajectory['failure_point'] = failure_point
            dataset.append(harmful_trajectory)
            
        return dataset
