"""
Attention Pattern Modulation

Modify attention weights based on ASAN's frequency analysis to prevent
model from attending to patterns that lead to harmful outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..models.asan_predictor import ASANPredictor


class AttentionModulator:
    """
    Modify attention patterns based on ASAN's frequency analysis
    
    Insight: Harmful patterns often involve specific attention head behaviors
    - Certain heads focus on instruction-following during jailbreaks
    - Other heads handle factual grounding during hallucinations
    
    We can selectively dampen or amplify specific attention patterns
    """
    
    def __init__(self, asan_predictor: ASANPredictor):
        """Initialize with learned attention modification strategies"""
        self.asan_predictor = asan_predictor
        self.asan_predictor.eval()
        
        # Learned attention modification strategies
        # Maps (layer_idx, head_idx) -> modification strategy
        self.attention_strategies = {}
        
    def identify_problematic_attention_heads(self, 
                                           attention_patterns: Dict[int, List[torch.Tensor]],
                                           current_trajectory: Dict[str, Any]) -> List[Tuple[int, int, float]]:
        """
        Use ASAN to identify which attention heads are contributing to harmful patterns
        
        Analysis:
        - Compare current attention to known harmful patterns
        - Identify which heads have unusual activation
        - Compute head-level importance for harm prediction
        
        Returns:
            problematic_heads: [(layer_idx, head_idx, importance)]
        """
        problematic_heads = []
        
        # Get ASAN prediction and attention weights
        with torch.no_grad():
            asan_output = self.asan_predictor(
                current_trajectory['attention_patterns'],
                current_trajectory['hidden_states'],
                current_trajectory['token_probs']
            )
            
            attention_weights = asan_output.get('attention_weights', {})
            harm_prob = asan_output['harm_probability'].item()
            
            if harm_prob < 0.5:
                return problematic_heads  # No problematic heads if low harm prob
        
        # Analyze attention patterns for each layer
        for layer_idx, layer_attentions in attention_patterns.items():
            if not layer_attentions:
                continue
            
            # Get latest attention pattern
            latest_attn = layer_attentions[-1]
            
            # Handle different attention formats
            if latest_attn.dim() == 4:  # [batch, num_heads, seq_len, seq_len]
                num_heads = latest_attn.shape[1]
                
                for head_idx in range(num_heads):
                    head_attn = latest_attn[0, head_idx]  # [seq_len, seq_len]
                    
                    # Compute head-level features
                    head_entropy = self._compute_attention_entropy(head_attn)
                    head_concentration = self._compute_concentration(head_attn)
                    head_unusualness = self._compute_unusualness(head_attn, layer_idx, head_idx)
                    
                    # Importance score: combination of features
                    importance = (
                        head_entropy * 0.3 +
                        head_concentration * 0.3 +
                        head_unusualness * 0.4
                    ) * harm_prob
                    
                    if importance > 0.5:  # Threshold for problematic
                        problematic_heads.append((layer_idx, head_idx, importance.item()))
                        
            elif latest_attn.dim() == 2:  # [seq_len, seq_len] - single head or averaged
                # Treat as single head
                head_entropy = self._compute_attention_entropy(latest_attn)
                head_concentration = self._compute_concentration(latest_attn)
                head_unusualness = self._compute_unusualness(latest_attn, layer_idx, 0)
                
                importance = (
                    head_entropy * 0.3 +
                    head_concentration * 0.3 +
                    head_unusualness * 0.4
                ) * harm_prob
                
                if importance > 0.5:
                    problematic_heads.append((layer_idx, 0, importance.item()))
        
        # Sort by importance
        problematic_heads.sort(key=lambda x: x[2], reverse=True)
        
        return problematic_heads
    
    def _compute_attention_entropy(self, attention: torch.Tensor) -> float:
        """Compute entropy of attention distribution"""
        # Flatten and normalize
        flat_attn = attention.flatten()
        normalized = flat_attn / (flat_attn.sum() + 1e-10)
        
        # Compute entropy
        entropy = -torch.sum(normalized * torch.log(normalized + 1e-10))
        
        # Normalize to [0, 1]
        max_entropy = np.log(len(normalized))
        normalized_entropy = entropy.item() / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _compute_concentration(self, attention: torch.Tensor) -> float:
        """Compute how concentrated attention is (Gini coefficient)"""
        flat_attn = attention.flatten()
        sorted_attn, _ = torch.sort(flat_attn)
        
        n = len(sorted_attn)
        cumsum = torch.cumsum(sorted_attn, dim=0)
        
        if cumsum[-1] > 0:
            gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
        else:
            gini = torch.tensor(0.0)
        
        return gini.item()
    
    def _compute_unusualness(self, attention: torch.Tensor, layer_idx: int, head_idx: int) -> float:
        """
        Compute how unusual this attention pattern is compared to normal patterns
        
        In practice, would compare to learned normal attention patterns
        For now, use heuristics: very sparse or very dense attention is unusual
        """
        flat_attn = attention.flatten()
        normalized = flat_attn / (flat_attn.sum() + 1e-10)
        
        # Unusual if very sparse (most probability on few positions)
        sparsity = (normalized < 0.01).float().mean().item()
        
        # Unusual if very concentrated on diagonal (self-attention only)
        if attention.shape[0] == attention.shape[1]:
            diagonal_prob = torch.diag(normalized.view(attention.shape)).sum().item()
        else:
            diagonal_prob = 0.0
        
        # Combined unusualness score
        unusualness = (sparsity * 0.6 + abs(diagonal_prob - 0.1) * 0.4)
        
        return unusualness
    
    def compute_attention_correction(self, 
                                    attention_weights: torch.Tensor,
                                    problematic_heads: List[Tuple[int, int, float]],
                                    layer_idx: int) -> torch.Tensor:
        """
        Compute how to modify attention weights to avoid harmful trajectory
        
        Strategies:
        - Dampen problematic heads (reduce their influence)
        - Amplify safety-promoting heads (increase factual grounding)
        - Rebalance attention distribution
        
        Constraints:
        - Attention weights must sum to 1
        - Changes should be smooth (no discontinuities)
        - Preserve semantic coherence
        
        Returns:
            modified_attention: corrected attention weights
            modification_magnitude: how much we changed
        """
        if attention_weights.dim() == 4:  # [batch, num_heads, seq_len, seq_len]
            modified_attention = attention_weights.clone()
            modification_magnitude = 0.0
            
            # Get problematic heads for this layer
            layer_problematic = [(h_idx, imp) for l_idx, h_idx, imp in problematic_heads 
                                if l_idx == layer_idx]
            
            if not layer_problematic:
                return attention_weights, 0.0
            
            # Apply corrections to each problematic head
            for head_idx, importance in layer_problematic:
                if head_idx < modified_attention.shape[1]:
                    head_attn = modified_attention[0, head_idx]
                    
                    # Dampen problematic head
                    damping_factor = 1.0 - (importance * 0.3)  # Reduce by up to 30%
                    head_attn = head_attn * damping_factor
                    
                    # Renormalize
                    head_attn = head_attn / (head_attn.sum(dim=-1, keepdim=True) + 1e-10)
                    
                    modified_attention[0, head_idx] = head_attn
                    modification_magnitude += (1.0 - damping_factor)
            
            # Rebalance: redistribute attention from dampened heads to other heads
            if modification_magnitude > 0:
                # Increase attention in other heads proportionally
                for head_idx in range(modified_attention.shape[1]):
                    if not any(h_idx == head_idx for h_idx, _ in layer_problematic):
                        other_head_attn = modified_attention[0, head_idx]
                        boost_factor = 1.0 + (modification_magnitude / modified_attention.shape[1]) * 0.1
                        other_head_attn = other_head_attn * boost_factor
                        other_head_attn = other_head_attn / (other_head_attn.sum(dim=-1, keepdim=True) + 1e-10)
                        modified_attention[0, head_idx] = other_head_attn
            
            modification_magnitude = modification_magnitude / len(layer_problematic)
            
        elif attention_weights.dim() == 2:  # [seq_len, seq_len]
            # Single head - apply general correction
            modified_attention = attention_weights.clone()
            
            # Smooth attention: reduce sharp peaks
            smoothing_factor = 0.1
            smoothed = F.conv2d(
                modified_attention.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 3, 3, device=modified_attention.device) / 9.0,
                padding=1
            ).squeeze()
            
            modified_attention = (1 - smoothing_factor) * modified_attention + smoothing_factor * smoothed
            
            # Renormalize
            modified_attention = modified_attention / (modified_attention.sum(dim=-1, keepdim=True) + 1e-10)
            modification_magnitude = smoothing_factor
            
        else:
            # Unknown format
            return attention_weights, 0.0
        
        return modified_attention, modification_magnitude
    
    def apply_attention_modification(self, 
                                   layer_attention: torch.Tensor,
                                   corrections: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Apply computed corrections to attention weights
        
        Returns:
            modified_attention: corrected attention weights
            modification_magnitude: how much we changed
        """
        # This is a wrapper - actual modification happens in compute_attention_correction
        # But could apply learned transformations here
        
        modified_attention = layer_attention.clone()
        modification_magnitude = 0.0
        
        # Apply any stored corrections
        for key, correction in corrections.items():
            if 'attention' in key.lower():
                modified_attention = modified_attention * (1.0 + correction)
                modification_magnitude += torch.abs(correction).mean().item()
        
        # Renormalize
        if modified_attention.dim() >= 2:
            modified_attention = modified_attention / (
                modified_attention.sum(dim=-1, keepdim=True) + 1e-10
            )
        
        return modified_attention, modification_magnitude
    
    def learn_attention_strategies(self, 
                                  safe_attention_patterns: Dict[int, List[torch.Tensor]],
                                  harmful_attention_patterns: Dict[int, List[torch.Tensor]]):
        """
        Learn modification strategies from safe vs harmful attention patterns
        
        This would train a model to predict which attention modifications
        lead to safer outputs
        """
        # Placeholder for learned strategies
        # In practice, would train a small model to predict safe attention modifications
        pass
