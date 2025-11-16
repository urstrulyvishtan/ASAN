"""
Trajectory Optimization

Optimize generation trajectory in real-time to avoid harmful outputs
while preserving quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np

try:
    from models.asan_predictor import ASANPredictor
    from alignment.spectral_steering import SpectralSteeringController
except ImportError:
    from ..models.asan_predictor import ASANPredictor
    from .spectral_steering import SpectralSteeringController


class TrajectoryOptimizer:
    """
    Optimize generation trajectory in real-time
    
    Goal: Find trajectory that minimizes harm probability while
    preserving semantic quality and coherence
    """
    
    def __init__(self, 
                 asan_predictor: ASANPredictor,
                 steering_controller: SpectralSteeringController,
                 optimization_steps: int = 3):
        """
        Args:
            asan_predictor: Trained ASAN model
            steering_controller: Steering controller
            optimization_steps: Number of optimization iterations
        """
        self.asan_predictor = asan_predictor
        self.steering_controller = steering_controller
        self.optimization_steps = optimization_steps
    
    def optimize_trajectory(self, 
                          current_trajectory: Dict[str, Any],
                          quality_preservation_weight: float = 0.5) -> Dict[str, Any]:
        """
        Optimize trajectory to minimize harm while preserving quality
        
        Optimization objective:
        min: harm_probability + λ * quality_loss
        
        Where quality_loss measures deviation from original trajectory
        
        Returns:
            optimized_trajectory: Improved trajectory
            optimization_history: Record of optimization process
        """
        optimization_history = []
        optimized_trajectory = current_trajectory.copy()
        
        # Initial state
        with torch.no_grad():
            initial_pred = self.asan_predictor(
                current_trajectory['attention_patterns'],
                current_trajectory['hidden_states'],
                current_trajectory['token_probs']
            )
            initial_harm_prob = initial_pred['harm_probability'].item()
        
        optimization_history.append({
            'step': 0,
            'harm_probability': initial_harm_prob,
            'quality_score': 1.0
        })
        
        # Iterative optimization
        for step in range(self.optimization_steps):
            # Compute steering vector
            steering_result = self.steering_controller.compute_steering_vector(
                optimized_trajectory,
                target_safety_level=0.1
            )
            
            # Apply steering to trajectory
            optimized_trajectory = self._apply_steering_to_trajectory(
                optimized_trajectory,
                steering_result['steering_vector'],
                step_size=1.0 / (step + 2)  # Decreasing step size
            )
            
            # Evaluate optimized trajectory
            with torch.no_grad():
                optimized_pred = self.asan_predictor(
                    optimized_trajectory['attention_patterns'],
                    optimized_trajectory['hidden_states'],
                    optimized_trajectory['token_probs']
                )
                optimized_harm_prob = optimized_pred['harm_probability'].item()
            
            # Compute quality preservation score
            quality_score = self._compute_quality_preservation(
                current_trajectory,
                optimized_trajectory
            )
            
            optimization_history.append({
                'step': step + 1,
                'harm_probability': optimized_harm_prob,
                'quality_score': quality_score,
                'combined_score': optimized_harm_prob * (1 - quality_preservation_weight) + 
                                 (1 - quality_score) * quality_preservation_weight
            })
            
            # Early stopping if improvement is minimal
            if step > 0:
                improvement = optimization_history[-2]['harm_probability'] - optimized_harm_prob
                if improvement < 0.01:
                    break
        
        return optimized_trajectory, optimization_history
    
    def _apply_steering_to_trajectory(self, 
                                     trajectory: Dict[str, Any],
                                     steering_vector: torch.Tensor,
                                     step_size: float) -> Dict[str, Any]:
        """Apply steering vector to entire trajectory"""
        modified_trajectory = {
            'attention_patterns': trajectory['attention_patterns'].copy(),
            'hidden_states': {},
            'token_probs': trajectory['token_probs'].copy()
        }
        
        # Apply steering to hidden states
        for layer_idx, layer_states in trajectory['hidden_states'].items():
            modified_layer_states = []
            
            for hidden_state in layer_states:
                # Project steering vector to hidden state space
                if hidden_state.dim() == 2:  # [seq_len, hidden_dim]
                    if steering_vector.dim() == 1:
                        # Broadcast steering vector
                        steering_applied = steering_vector.unsqueeze(0).expand_as(hidden_state)
                    else:
                        # Match dimensions
                        min_seq_len = min(hidden_state.shape[0], steering_vector.shape[0])
                        steering_applied = steering_vector[:min_seq_len]
                        hidden_state = hidden_state[:min_seq_len]
                    
                    modified_state = hidden_state + steering_applied * step_size
                    modified_layer_states.append(modified_state)
                else:
                    # Unknown format, keep original
                    modified_layer_states.append(hidden_state)
            
            modified_trajectory['hidden_states'][layer_idx] = modified_layer_states
        
        return modified_trajectory
    
    def _compute_quality_preservation(self, 
                                     original_trajectory: Dict[str, Any],
                                     modified_trajectory: Dict[str, Any]) -> float:
        """
        Compute how well quality is preserved
        
        Measures:
        - Cosine similarity between hidden states
        - Preservation of attention patterns
        - Token probability distributions
        """
        quality_scores = []
        
        # Compare hidden states
        for layer_idx in original_trajectory['hidden_states']:
            if layer_idx in modified_trajectory['hidden_states']:
                orig_states = original_trajectory['hidden_states'][layer_idx]
                mod_states = modified_trajectory['hidden_states'][layer_idx]
                
                if orig_states and mod_states:
                    # Compare last hidden state
                    orig_last = orig_states[-1]
                    mod_last = mod_states[-1]
                    
                    # Align dimensions
                    if orig_last.shape == mod_last.shape:
                        # Compute cosine similarity
                        orig_flat = orig_last.flatten()
                        mod_flat = mod_last.flatten()
                        
                        cosine_sim = F.cosine_similarity(
                            orig_flat.unsqueeze(0),
                            mod_flat.unsqueeze(0)
                        ).item()
                        
                        quality_scores.append(max(0.0, cosine_sim))
        
        # Compare token probabilities
        if original_trajectory['token_probs'] and modified_trajectory['token_probs']:
            orig_probs = original_trajectory['token_probs'][-1]
            mod_probs = modified_trajectory['token_probs'][-1]
            
            if orig_probs.shape == mod_probs.shape:
                # KL divergence (lower is better, convert to similarity)
                kl_div = F.kl_div(
                    F.log_softmax(mod_probs, dim=-1),
                    F.softmax(orig_probs, dim=-1),
                    reduction='batchmean'
                ).item()
                
                # Convert KL divergence to similarity score [0, 1]
                similarity = np.exp(-kl_div)
                quality_scores.append(similarity)
        
        # Average quality scores
        if quality_scores:
            return np.mean(quality_scores)
        else:
            return 0.5  # Default middle score
    
    def optimize_next_token(self, 
                           current_trajectory: Dict[str, Any],
                           candidate_logits: torch.Tensor,
                           top_k: int = 10) -> Tuple[int, Dict[str, float]]:
        """
        Optimize which token to generate next
        
        Strategy:
        1. Consider top-k candidate tokens
        2. For each, simulate trajectory continuation
        3. Evaluate harm probability for each
        4. Select token that minimizes harm while preserving quality
        """
        # Get top-k candidate tokens
        top_k_probs, top_k_indices = torch.topk(
            F.softmax(candidate_logits, dim=-1),
            k=min(top_k, candidate_logits.shape[-1])
        )
        
        best_token = top_k_indices[0].item()  # Default: top-1
        best_score = float('inf')
        token_scores = {}
        
        # Evaluate each candidate
        for token_idx, token_id in enumerate(top_k_indices[0]):
            # Simulate trajectory with this token
            simulated_trajectory = self._simulate_token_addition(
                current_trajectory, token_id.item()
            )
            
            # Evaluate harm probability
            with torch.no_grad():
                pred = self.asan_predictor(
                    simulated_trajectory['attention_patterns'],
                    simulated_trajectory['hidden_states'],
                    simulated_trajectory['token_probs']
                )
                harm_prob = pred['harm_probability'].item()
            
            # Quality: prefer tokens with higher probability
            quality = top_k_probs[0, token_idx].item()
            
            # Combined score: lower is better
            # Weight harm probability more heavily
            score = harm_prob * 0.7 + (1 - quality) * 0.3
            token_scores[token_id.item()] = {
                'harm_probability': harm_prob,
                'quality': quality,
                'score': score
            }
            
            if score < best_score:
                best_score = score
                best_token = token_id.item()
        
        return best_token, token_scores
    
    def _simulate_token_addition(self, 
                                trajectory: Dict[str, Any],
                                token_id: int) -> Dict[str, Any]:
        """
        Simulate adding a token to trajectory
        
        This is a simplified simulation - in practice would need
        full model forward pass
        """
        # Create simulated trajectory (simplified)
        simulated = {
            'attention_patterns': trajectory['attention_patterns'].copy(),
            'hidden_states': {},
            'token_probs': trajectory['token_probs'].copy()
        }
        
        # Simulate token probability (one-hot for simplicity)
        vocab_size = trajectory['token_probs'][0].shape[-1] if trajectory['token_probs'] else 50257
        token_prob = torch.zeros(vocab_size)
        if token_id < vocab_size:
            token_prob[token_id] = 1.0
        simulated['token_probs'].append(token_prob)
        
        # Simulate hidden states (copy last state)
        for layer_idx, layer_states in trajectory['hidden_states'].items():
            if layer_states:
                simulated['hidden_states'][layer_idx] = layer_states + [layer_states[-1].clone()]
            else:
                simulated['hidden_states'][layer_idx] = []
        
        return simulated
