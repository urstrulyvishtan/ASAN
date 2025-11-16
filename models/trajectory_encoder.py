"""
Trajectory Encoder for ASAN

Convert LLM internal states into unified temporal representation suitable for ASAN analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class TrajectoryEncoder(nn.Module):
    """
    Encode LLM internal states into unified temporal representation
    
    Input: Raw attention patterns, hidden states, token probabilities
    Output: Unified temporal sequence for ASAN analysis
    """
    
    def __init__(self, input_dims: Dict[str, int], encoding_dim: int = 256):
        """
        Args:
            input_dims: dict with dims for attention, hidden_states, token_probs
            encoding_dim: dimension of unified encoding
        """
        super().__init__()
        
        self.encoding_dim = encoding_dim
        
        # Separate encoders for each modality
        self.attention_encoder = nn.Sequential(
            nn.Linear(input_dims['attention'], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, encoding_dim)
        )
        
        self.hidden_state_encoder = nn.Sequential(
            nn.Linear(input_dims['hidden_state'], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, encoding_dim)
        )
        
        self.token_prob_encoder = nn.Sequential(
            nn.Linear(input_dims['token_prob'], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, encoding_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(encoding_dim * 3, encoding_dim),
            nn.LayerNorm(encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal modeling
        self.temporal_conv = nn.Conv1d(encoding_dim, encoding_dim, kernel_size=3, padding=1)
        self.temporal_norm = nn.LayerNorm(encoding_dim)
        
    def forward(self, attention_patterns: Dict, hidden_states: Dict, 
                token_probs: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode multi-modal LLM states into unified representation
        
        Args:
            attention_patterns: Dict[layer_idx, List[attention_tensors]]
            hidden_states: Dict[layer_idx, List[hidden_tensors]]
            token_probs: List[token_prob_tensors]
            
        Returns:
            encoded_trajectory: [batch, timesteps, encoding_dim]
        """
        batch_size = 1  # Assume single trajectory for now
        timesteps = len(token_probs)
        
        # Process each timestep
        encoded_timesteps = []
        
        for t in range(timesteps):
            # Extract features for this timestep
            attention_features = self._extract_attention_features(attention_patterns, t)
            hidden_features = self._extract_hidden_features(hidden_states, t)
            token_features = self._extract_token_features(token_probs[t])
            
            # Encode each modality
            # Add batch dimension if needed (encoders expect 1D input, output 1D)
            attn_encoded = self.attention_encoder(attention_features.unsqueeze(0)).squeeze(0)  # [256]
            hidden_encoded = self.hidden_state_encoder(hidden_features.unsqueeze(0)).squeeze(0)  # [256]
            token_encoded = self.token_prob_encoder(token_features.unsqueeze(0)).squeeze(0)  # [256]
            
            # Fuse modalities (add batch dim for concatenation)
            fused = torch.cat([
                attn_encoded.unsqueeze(0), 
                hidden_encoded.unsqueeze(0), 
                token_encoded.unsqueeze(0)
            ], dim=-1)  # [1, 3*encoding_dim]
            encoded_timestep = self.fusion(fused).squeeze(0)  # [encoding_dim]
            
            encoded_timesteps.append(encoded_timestep)
        
        # Stack timesteps (add batch dimension)
        # Each encoded_timestep is [encoding_dim], we want [batch=1, timesteps, encoding_dim]
        if len(encoded_timesteps) == 0:
            # Handle empty trajectory case
            encoded_trajectory = torch.zeros(1, 1, self.encoding_dim)
        else:
            encoded_trajectory = torch.stack(encoded_timesteps, dim=0).unsqueeze(0)  # [1, timesteps, encoding_dim]
        
        # Apply temporal convolution
        encoded_trajectory = encoded_trajectory.transpose(1, 2)  # [batch, encoding_dim, timesteps]
        encoded_trajectory = self.temporal_conv(encoded_trajectory)
        encoded_trajectory = encoded_trajectory.transpose(1, 2)  # [batch, timesteps, encoding_dim]
        
        # Apply layer norm
        encoded_trajectory = self.temporal_norm(encoded_trajectory)
        
        return encoded_trajectory
        
    def _extract_attention_features(self, attention_patterns: Dict, timestep: int) -> torch.Tensor:
        """Extract attention features for a specific timestep"""
        features = []
        
        for layer_idx in sorted(attention_patterns.keys()):
            if timestep < len(attention_patterns[layer_idx]):
                attn = attention_patterns[layer_idx][timestep]
                
                # Compute attention statistics
                # attn: [seq_len, seq_len]
                if attn.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    attn = attn.mean(dim=1)  # Average across heads
                    
                # Extract features
                attn_mean = attn.mean()
                attn_std = attn.std()
                attn_max = attn.max()
                attn_entropy = self._compute_attention_entropy(attn)
                
                # Attention pattern features
                diagonal_attention = torch.diag(attn).mean()
                off_diagonal_attention = (attn.sum() - diagonal_attention) / (attn.size(0) - 1)
                
                layer_features = torch.tensor([
                    attn_mean.item(), attn_std.item(), attn_max.item(), 
                    attn_entropy.item(), diagonal_attention.item(), off_diagonal_attention.item()
                ])
                features.append(layer_features)
            else:
                # Pad with zeros if timestep doesn't exist
                features.append(torch.zeros(6))
                
        # Concatenate all layer features
        if features:
            return torch.cat(features)
        else:
            return torch.zeros(6 * len(attention_patterns))
            
    def _extract_hidden_features(self, hidden_states: Dict, timestep: int) -> torch.Tensor:
        """Extract hidden state features for a specific timestep"""
        features = []
        
        for layer_idx in sorted(hidden_states.keys()):
            if timestep < len(hidden_states[layer_idx]):
                hidden = hidden_states[layer_idx][timestep]
                
                # Compute hidden state statistics
                # hidden: [seq_len, hidden_dim]
                hidden_mean = hidden.mean(dim=0)  # [hidden_dim]
                hidden_std = hidden.std(dim=0)    # [hidden_dim]
                hidden_norm = torch.norm(hidden, dim=-1).mean()
                
                # Principal component analysis features
                try:
                    U, S, V = torch.svd(hidden)
                    top_singular_values = S[:5]  # Top 5 singular values
                    explained_variance = S[:5] / S.sum()
                except:
                    top_singular_values = torch.zeros(5)
                    explained_variance = torch.zeros(5)
                
                # Combine features
                layer_features = torch.cat([
                    hidden_mean.mean().unsqueeze(0),
                    hidden_std.mean().unsqueeze(0),
                    hidden_norm.unsqueeze(0),
                    top_singular_values,
                    explained_variance
                ])
                features.append(layer_features)
            else:
                # Pad with zeros if timestep doesn't exist
                features.append(torch.zeros(8))
                
        # Concatenate all layer features
        if features:
            return torch.cat(features)
        else:
            return torch.zeros(8 * len(hidden_states))
            
    def _extract_token_features(self, token_probs: torch.Tensor) -> torch.Tensor:
        """Extract token probability features"""
        # token_probs: [vocab_size]
        
        # Basic statistics
        prob_mean = token_probs.mean()
        prob_std = token_probs.std()
        prob_max = token_probs.max()
        prob_entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10))
        
        # Top-k statistics
        top_k_values, _ = torch.topk(token_probs, k=10)
        top_k_mass = top_k_values.sum()
        top_k_ratio = top_k_mass / token_probs.sum()
        
        # Concentration measures
        gini_coefficient = self._compute_gini_coefficient(token_probs)
        
        # Combine all features
        features = torch.tensor([
            prob_mean.item(), prob_std.item(), prob_max.item(), prob_entropy.item(),
            top_k_mass.item(), top_k_ratio.item(), gini_coefficient.item()
        ])
        
        return features
        
    def _compute_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distribution"""
        # Flatten attention matrix
        flat_attention = attention.flatten()
        # Normalize to probability distribution
        normalized = flat_attention / flat_attention.sum()
        # Compute entropy
        entropy = -torch.sum(normalized * torch.log(normalized + 1e-10))
        return entropy
        
    def _compute_gini_coefficient(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute Gini coefficient as concentration measure"""
        sorted_probs, _ = torch.sort(probs)
        n = len(sorted_probs)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        return (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
        
    def compute_temporal_features(self, trajectories: Dict) -> torch.Tensor:
        """
        Compute additional temporal features:
        - Rate of change in attention patterns
        - Hidden state trajectory curvature
        - Token probability entropy trends
        - Cross-layer coherence measures
        """
        temporal_features = []
        
        # Compute attention entropy trends
        attention_entropies = self._compute_attention_entropy_trend(trajectories['attention_patterns'])
        if attention_entropies is not None:
            temporal_features.append(attention_entropies)
            
        # Compute hidden state trajectory features
        hidden_trajectory_features = self._compute_hidden_trajectory_features(trajectories['hidden_states'])
        if hidden_trajectory_features is not None:
            temporal_features.append(hidden_trajectory_features)
            
        # Compute token probability trends
        token_prob_trends = self._compute_token_prob_trends(trajectories['token_probs'])
        if token_prob_trends is not None:
            temporal_features.append(token_prob_trends)
            
        # Compute cross-layer coherence
        coherence_features = self._compute_cross_layer_coherence(trajectories)
        if coherence_features is not None:
            temporal_features.append(coherence_features)
            
        if temporal_features:
            return torch.cat(temporal_features, dim=-1)
        else:
            return torch.zeros(len(trajectories['generated_tokens']), 10)
            
    def _compute_attention_entropy_trend(self, attention_patterns: Dict) -> Optional[torch.Tensor]:
        """Compute attention entropy trend over time"""
        if not attention_patterns:
            return None
            
        entropies = []
        for layer_idx in sorted(attention_patterns.keys()):
            layer_entropies = []
            for attn in attention_patterns[layer_idx]:
                entropy = self._compute_attention_entropy(attn)
                layer_entropies.append(entropy.item())
            entropies.append(layer_entropies)
            
        if entropies:
            # Compute trend (slope of linear regression)
            timesteps = torch.arange(len(entropies[0])).float()
            trends = []
            for layer_entropies in entropies:
                if len(layer_entropies) > 1:
                    # Simple linear regression slope
                    x_mean = timesteps.mean()
                    y_mean = torch.tensor(layer_entropies).float().mean()
                    numerator = torch.sum((timesteps - x_mean) * (torch.tensor(layer_entropies).float() - y_mean))
                    denominator = torch.sum((timesteps - x_mean) ** 2)
                    slope = numerator / (denominator + 1e-10)
                    trends.append(slope.item())
                else:
                    trends.append(0.0)
                    
            return torch.tensor(trends).unsqueeze(0).repeat(len(entropies[0]), 1)
        return None
        
    def _compute_hidden_trajectory_features(self, hidden_states: Dict) -> Optional[torch.Tensor]:
        """Compute hidden state trajectory curvature and acceleration"""
        if not hidden_states:
            return None
            
        features = []
        for layer_idx in sorted(hidden_states.keys()):
            layer_states = hidden_states[layer_idx]
            if len(layer_states) < 3:
                continue
                
            # Compute velocity and acceleration
            velocities = []
            accelerations = []
            
            for i in range(1, len(layer_states)):
                # Velocity: change in hidden state
                vel = torch.norm(layer_states[i] - layer_states[i-1], dim=-1).mean()
                velocities.append(vel.item())
                
                if i > 1:
                    # Acceleration: change in velocity
                    acc = velocities[-1] - velocities[-2]
                    accelerations.append(acc)
                    
            if accelerations:
                avg_acceleration = np.mean(accelerations)
                acceleration_std = np.std(accelerations)
                features.extend([avg_acceleration, acceleration_std])
            else:
                features.extend([0.0, 0.0])
                
        if features:
            return torch.tensor(features).unsqueeze(0).repeat(len(hidden_states[list(hidden_states.keys())[0]]), 1)
        return None
        
    def _compute_token_prob_trends(self, token_probs: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute token probability entropy trends"""
        if not token_probs:
            return None
            
        entropies = []
        for probs in token_probs:
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            entropies.append(entropy.item())
            
        if len(entropies) > 1:
            # Compute trend
            timesteps = torch.arange(len(entropies)).float()
            x_mean = timesteps.mean()
            y_mean = torch.tensor(entropies).float().mean()
            numerator = torch.sum((timesteps - x_mean) * (torch.tensor(entropies).float() - y_mean))
            denominator = torch.sum((timesteps - x_mean) ** 2)
            slope = numerator / (denominator + 1e-10)
            
            return slope.unsqueeze(0).repeat(len(entropies), 1)
        return None
        
    def _compute_cross_layer_coherence(self, trajectories: Dict) -> Optional[torch.Tensor]:
        """Compute coherence between different layers"""
        if 'attention_patterns' not in trajectories or 'hidden_states' not in trajectories:
            return None
            
        attention_patterns = trajectories['attention_patterns']
        hidden_states = trajectories['hidden_states']
        
        if not attention_patterns or not hidden_states:
            return None
            
        # Compute correlation between layers
        layer_indices = sorted(set(attention_patterns.keys()) & set(hidden_states.keys()))
        if len(layer_indices) < 2:
            return None
            
        coherence_features = []
        timesteps = len(trajectories['generated_tokens'])
        
        for t in range(timesteps):
            layer_correlations = []
            
            for i in range(len(layer_indices) - 1):
                layer1_idx = layer_indices[i]
                layer2_idx = layer_indices[i + 1]
                
                if (t < len(attention_patterns[layer1_idx]) and 
                    t < len(attention_patterns[layer2_idx])):
                    
                    # Compute attention correlation
                    attn1 = attention_patterns[layer1_idx][t].flatten()
                    attn2 = attention_patterns[layer2_idx][t].flatten()
                    
                    # Ensure same length
                    min_len = min(len(attn1), len(attn2))
                    attn1 = attn1[:min_len]
                    attn2 = attn2[:min_len]
                    
                    correlation = torch.corrcoef(torch.stack([attn1, attn2]))[0, 1]
                    if not torch.isnan(correlation):
                        layer_correlations.append(correlation.item())
                    else:
                        layer_correlations.append(0.0)
                        
            if layer_correlations:
                avg_correlation = np.mean(layer_correlations)
                coherence_features.append(avg_correlation)
            else:
                coherence_features.append(0.0)
                
        if coherence_features:
            return torch.tensor(coherence_features).unsqueeze(-1)
        return None


class MultiModalFusion(nn.Module):
    """
    Advanced fusion module for combining different modalities
    """
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        
        # Individual encoders
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for dim in input_dims
        ])
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(output_dim, num_heads=8)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(input_dims), output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: List of tensors for each modality
            
        Returns:
            fused representation
        """
        # Encode each modality
        encoded_inputs = []
        for i, inp in enumerate(inputs):
            encoded = self.encoders[i](inp)
            encoded_inputs.append(encoded)
            
        # Cross-modal attention
        attended_inputs = []
        for i, encoded in enumerate(encoded_inputs):
            # Use other modalities as keys/values
            other_modalities = [encoded_inputs[j] for j in range(len(encoded_inputs)) if j != i]
            if other_modalities:
                kv = torch.stack(other_modalities, dim=0).mean(dim=0)
                attended, _ = self.cross_attention(encoded.unsqueeze(0), kv.unsqueeze(0), kv.unsqueeze(0))
                attended_inputs.append(attended.squeeze(0))
            else:
                attended_inputs.append(encoded)
                
        # Final fusion
        concatenated = torch.cat(attended_inputs, dim=-1)
        fused = self.fusion(concatenated)
        
        return fused
