"""
Hidden State Correction

Correct hidden states when harmful patterns are detected by ASAN,
guiding model toward safer representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..models.asan_predictor import ASANPredictor
from .spectral_steering import SpectralSteeringController


class HiddenStateCorrector:
    """
    Correct hidden states when harmful patterns are detected
    
    Strategy:
    1. Identify harmful patterns in hidden states using ASAN
    2. Compute correction direction toward safe states
    3. Apply correction gradually to preserve coherence
    """
    
    def __init__(self, asan_predictor: ASANPredictor, correction_strength: float = 0.2):
        """
        Args:
            asan_predictor: Trained ASAN model
            correction_strength: How strongly to correct hidden states
        """
        self.asan_predictor = asan_predictor
        self.asan_predictor.eval()
        self.correction_strength = correction_strength
        
        # Learned safe hidden state templates (can be populated from training)
        self.safe_state_templates = None
    
    def detect_harmful_states(self, 
                             hidden_states: Dict[int, List[torch.Tensor]],
                             current_trajectory: Dict[str, Any]) -> Dict[int, Tuple[torch.Tensor, float]]:
        """
        Identify which layers have harmful hidden states
        
        Returns:
            harmful_layers: Dict[layer_idx, (hidden_state, harm_score)]
        """
        harmful_layers = {}
        
        # Get ASAN prediction
        with torch.no_grad():
            asan_output = self.asan_predictor(
                current_trajectory['attention_patterns'],
                current_trajectory['hidden_states'],
                current_trajectory['token_probs']
            )
            
            harm_prob = asan_output['harm_probability'].item()
            
            if harm_prob < 0.5:
                return harmful_layers  # No harmful layers if low harm prob
        
        # Analyze each layer's hidden states
        for layer_idx, layer_states in hidden_states.items():
            if not layer_states:
                continue
            
            # Get latest hidden state
            latest_state = layer_states[-1]
            
            # Compute harm score for this layer
            # In practice, would use learned layer-specific harm detectors
            harm_score = self._compute_layer_harm_score(
                latest_state, layer_idx, harm_prob
            )
            
            if harm_score > 0.5:
                harmful_layers[layer_idx] = (latest_state, harm_score)
        
        return harmful_layers
    
    def _compute_layer_harm_score(self, 
                                 hidden_state: torch.Tensor,
                                 layer_idx: int,
                                 global_harm_prob: float) -> float:
        """
        Compute how harmful the hidden state in this layer is
        
        Heuristics:
        - High variance might indicate instability
        - Very large norms might indicate divergence
        - Unusual activation patterns
        """
        # Compute statistics
        state_norm = torch.norm(hidden_state).item()
        state_std = hidden_state.std().item()
        state_mean = hidden_state.mean().item()
        
        # Normalize statistics (heuristic thresholds)
        norm_score = min(1.0, state_norm / 100.0)  # Normalize by expected norm
        std_score = min(1.0, state_std / 10.0)  # Normalize by expected std
        
        # Combined harm score
        harm_score = (norm_score * 0.4 + std_score * 0.4) * global_harm_prob
        
        return min(1.0, harm_score)
    
    def compute_correction_vector(self, 
                                 hidden_state: torch.Tensor,
                                 layer_idx: int,
                                 harm_score: float) -> torch.Tensor:
        """
        Compute correction vector to push hidden state toward safer region
        
        Strategy:
        - If safe templates available, push toward nearest safe template
        - Otherwise, reduce extreme activations (smooth state)
        """
        if self.safe_state_templates is not None and layer_idx in self.safe_state_templates:
            # Push toward safe template
            safe_template = self.safe_state_templates[layer_idx]
            
            # Align dimensions
            if safe_template.shape != hidden_state.shape:
                # Pool or pad to match
                if safe_template.dim() == 2 and hidden_state.dim() == 2:
                    min_seq_len = min(safe_template.shape[0], hidden_state.shape[0])
                    safe_template = safe_template[:min_seq_len]
                    hidden_state_aligned = hidden_state[:min_seq_len]
                else:
                    # Use mean pooling
                    safe_template = safe_template.mean(dim=0, keepdim=True)
                    hidden_state_aligned = hidden_state.mean(dim=0, keepdim=True)
            else:
                hidden_state_aligned = hidden_state
            
            # Correction: direction toward safe template
            correction = (safe_template - hidden_state_aligned) * harm_score
        else:
            # Fallback: smooth extreme activations
            correction = self._compute_smoothing_correction(hidden_state, harm_score)
        
        # Scale by correction strength
        correction = correction * self.correction_strength
        
        return correction
    
    def _compute_smoothing_correction(self, 
                                     hidden_state: torch.Tensor,
                                     harm_score: float) -> torch.Tensor:
        """Compute correction by smoothing extreme activations"""
        # Compute mean and std
        mean = hidden_state.mean(dim=-1, keepdim=True)
        std = hidden_state.std(dim=-1, keepdim=True)
        
        # Identify extreme activations (more than 2 std from mean)
        extremes = (hidden_state - mean).abs() > 2 * std
        
        # Correction: pull extremes toward mean
        correction = -extremes.float() * (hidden_state - mean) * harm_score
        
        return correction
    
    def apply_correction(self, 
                        hidden_state: torch.Tensor,
                        correction_vector: torch.Tensor,
                        preserve_coherence: bool = True) -> torch.Tensor:
        """
        Apply correction to hidden state
        
        Args:
            hidden_state: Original hidden state
            correction_vector: Correction to apply
            preserve_coherence: Whether to preserve semantic coherence
            
        Returns:
            corrected_state: Hidden state after correction
        """
        # Align dimensions
        if correction_vector.shape != hidden_state.shape:
            if correction_vector.dim() == hidden_state.dim():
                # Pad or truncate
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(correction_vector.shape, hidden_state.shape))
                correction_vector = correction_vector[:min_shape[0], :min_shape[1]] if correction_vector.dim() == 2 else correction_vector[:min_shape[0]]
                hidden_state_aligned = hidden_state[:min_shape[0], :min_shape[1]] if hidden_state.dim() == 2 else hidden_state[:min_shape[0]]
            else:
                # Use mean pooling
                correction_vector = correction_vector.mean(dim=0, keepdim=True)
                hidden_state_aligned = hidden_state
        else:
            hidden_state_aligned = hidden_state
        
        # Apply correction
        corrected_state = hidden_state_aligned + correction_vector
        
        # Preserve coherence: maintain relative magnitudes
        if preserve_coherence:
            original_norm = torch.norm(hidden_state_aligned, dim=-1, keepdim=True)
            corrected_norm = torch.norm(corrected_state, dim=-1, keepdim=True)
            
            # Scale to preserve norm if change is too large
            max_change_ratio = 1.2
            scale_factor = torch.clamp(
                corrected_norm / (original_norm + 1e-8),
                1.0 / max_change_ratio,
                max_change_ratio
            )
            
            corrected_state = corrected_state / (scale_factor + 1e-8) * original_norm
        
        # Clip extreme values
        corrected_state = torch.clamp(corrected_state, -10.0, 10.0)
        
        return corrected_state
    
    def correct_trajectory(self, 
                          current_trajectory: Dict[str, Any],
                          target_safety_level: float = 0.1) -> Dict[str, Any]:
        """
        Correct entire trajectory by modifying hidden states
        
        Returns:
            corrected_trajectory: Trajectory with corrected hidden states
            correction_log: Record of corrections applied
        """
        corrected_trajectory = {
            'attention_patterns': current_trajectory['attention_patterns'].copy(),
            'hidden_states': {},
            'token_probs': current_trajectory['token_probs'].copy()
        }
        correction_log = []
        
        # Detect harmful states
        harmful_layers = self.detect_harmful_states(
            current_trajectory['hidden_states'],
            current_trajectory
        )
        
        # Apply corrections
        for layer_idx, layer_states in current_trajectory['hidden_states'].items():
            corrected_layer_states = []
            
            for timestep, hidden_state in enumerate(layer_states):
                if layer_idx in harmful_layers:
                    harm_score = harmful_layers[layer_idx][1]
                    
                    # Compute correction
                    correction = self.compute_correction_vector(
                        hidden_state, layer_idx, harm_score
                    )
                    
                    # Apply correction
                    corrected_state = self.apply_correction(
                        hidden_state, correction, preserve_coherence=True
                    )
                    
                    corrected_layer_states.append(corrected_state)
                    
                    if timestep == len(layer_states) - 1:  # Log only last timestep
                        correction_log.append({
                            'layer': layer_idx,
                            'timestep': timestep,
                            'harm_score': harm_score,
                            'correction_magnitude': torch.norm(correction).item()
                        })
                else:
                    # No correction needed
                    corrected_layer_states.append(hidden_state)
            
            corrected_trajectory['hidden_states'][layer_idx] = corrected_layer_states
        
        return corrected_trajectory, correction_log
