"""
Core ASAN Predictor Architecture

Adaptive Spectral Alignment Network for predicting LLM failures from internal state trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

from .trajectory_encoder import TrajectoryEncoder
from .wavelets import TemporalWaveletTransform


@dataclass
class ASANConfig:
    """Configuration for ASAN predictor"""
    # Input dimensions
    attention_dim: int = 6 * 12  # 6 features per layer * 12 layers
    hidden_state_dim: int = 13 * 12  # 13 features per layer * 12 layers (matches encoder)
    token_prob_dim: int = 7  # Token probability features
    
    # Model architecture
    encoding_dim: int = 256
    attention_dim_internal: int = 128
    attention_heads: int = 8
    decomposition_levels: int = 4
    wavelet: str = 'db4'
    
    # Harm categories
    num_harm_categories: int = 5
    
    # Training
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5


class FrequencyBandAttention(nn.Module):
    """
    Attention mechanism for specific frequency bands
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, timesteps, features]
            
        Returns:
            attended_output: [batch, timesteps, features]
            attention_weights: [batch, num_heads, timesteps, timesteps]
        """
        # Self-attention
        attended, attention_weights = self.attention(x, x, x)
        
        # Residual connection and layer norm
        x = self.layer_norm1(x + attended)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        
        # Residual connection and layer norm
        output = self.layer_norm2(x + ff_output)
        
        return output, attention_weights


class ASANPredictor(nn.Module):
    """
    Adaptive Spectral Alignment Network for predicting LLM failures
    
    Architecture:
    1. Trajectory encoding (convert raw LLM states to unified format)
    2. Wavelet decomposition (separate frequency bands)
    3. Frequency-specific attention (learn important patterns per band)
    4. Cross-frequency integration (combine multi-scale information)
    5. Prediction head (output harm probability and category)
    """
    
    def __init__(self, config: ASANConfig):
        super().__init__()
        
        self.config = config
        
        # Input dimensions
        input_dims = {
            'attention': config.attention_dim,
            'hidden_state': config.hidden_state_dim,
            'token_prob': config.token_prob_dim
        }
        
        # Trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(
            input_dims=input_dims,
            encoding_dim=config.encoding_dim
        )
        
        # Wavelet decomposition
        self.wavelet_transform = TemporalWaveletTransform(
            wavelet=config.wavelet,
            levels=config.decomposition_levels
        )
        
        # Attention mechanism for each frequency band
        self.frequency_attention = nn.ModuleDict({
            f'level_{i}': FrequencyBandAttention(
                input_dim=config.encoding_dim,
                hidden_dim=config.attention_dim_internal,
                num_heads=config.attention_heads
            ) for i in range(config.decomposition_levels + 1)
        })
        
        # Cross-frequency integration
        self.frequency_fusion = nn.Sequential(
            nn.Linear(config.encoding_dim * (config.decomposition_levels + 1), 
                     config.encoding_dim),
            nn.LayerNorm(config.encoding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Prediction heads
        self.harm_probability_head = nn.Sequential(
            nn.Linear(config.encoding_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.harm_category_head = nn.Sequential(
            nn.Linear(config.encoding_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_harm_categories)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(config.encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Learnable frequency band importance weights
        self.band_weights = nn.Parameter(
            torch.ones(config.decomposition_levels + 1)
        )
        
        # Temporal consistency module
        self.temporal_consistency = nn.LSTM(
            input_size=config.encoding_dim,
            hidden_size=config.encoding_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )
        
    def forward(self, attention_patterns: Dict, hidden_states: Dict, 
                token_probs: List[torch.Tensor], current_timestep: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Predict probability of harmful output
        
        Args:
            attention_patterns: Dict[layer_idx, List[attention_tensors]]
            hidden_states: Dict[layer_idx, List[hidden_tensors]]
            token_probs: List[token_prob_tensors]
            current_timestep: if provided, predict at specific point in generation
            
        Returns:
            predictions: dict with keys:
                - 'harm_probability': [batch] probability of harmful output
                - 'harm_category': [batch, num_categories] category logits
                - 'confidence': [batch] prediction confidence
                - 'frequency_contributions': [batch, num_bands] per-band importance
                - 'attention_weights': dict of attention weights per frequency band
                - 'encoded_representation': [batch, encoding_dim] final representation
        """
        
        # 1. Encode trajectories
        encoded_trajectory = self.trajectory_encoder(
            attention_patterns, hidden_states, token_probs
        )
        
        # If current_timestep specified, only use trajectory up to that point
        if current_timestep is not None:
            encoded_trajectory = encoded_trajectory[:, :current_timestep, :]
        
        # 2. Apply temporal consistency (LSTM)
        lstm_output, _ = self.temporal_consistency(encoded_trajectory)
        
        # 3. Wavelet decomposition
        wavelet_coeffs = self.wavelet_transform(lstm_output)
        
        # 4. Apply attention to each frequency band
        attended_coeffs = {}
        attention_weights = {}
        
        for band_name, coeffs in wavelet_coeffs.items():
            attended, attn_weights = self.frequency_attention[band_name](coeffs)
            attended_coeffs[band_name] = attended
            attention_weights[band_name] = attn_weights
        
        # 5. Aggregate across frequency bands
        # Pool temporal dimension for each band
        pooled_bands = []
        for band_name in sorted(attended_coeffs.keys()):
            pooled = torch.mean(attended_coeffs[band_name], dim=1)  # [batch, encoding_dim]
            pooled_bands.append(pooled)
        
        # Weight by learned importance
        weighted_bands = [
            pooled_bands[i] * torch.softmax(self.band_weights, dim=0)[i] 
            for i in range(len(pooled_bands))
        ]
        
        # Concatenate and fuse
        concatenated = torch.cat(weighted_bands, dim=1)
        fused_representation = self.frequency_fusion(concatenated)
        
        # 6. Make predictions
        harm_prob = self.harm_probability_head(fused_representation).squeeze(-1)
        harm_categories = self.harm_category_head(fused_representation)
        confidence = self.confidence_head(fused_representation).squeeze(-1)
        
        # Compute frequency band contributions
        frequency_contributions = torch.softmax(self.band_weights, dim=0)
        
        return {
            'harm_probability': harm_prob,
            'harm_category': harm_categories,
            'confidence': confidence,
            'frequency_contributions': frequency_contributions,
            'attention_weights': attention_weights,
            'encoded_representation': fused_representation,
            'wavelet_coefficients': wavelet_coeffs
        }
        
    def predict_at_each_timestep(self, attention_patterns: Dict, hidden_states: Dict, 
                                token_probs: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Make predictions at each timestep during generation
        
        Returns predictions over time showing when harmful pattern emerges
        
        Returns:
            temporal_predictions: List of prediction dicts for each timestep
        """
        temporal_predictions = []
        
        for timestep in range(len(token_probs)):
            prediction = self.forward(
                attention_patterns, hidden_states, token_probs, current_timestep=timestep
            )
            temporal_predictions.append(prediction)
            
        return temporal_predictions
        
    def get_spectral_signature(self, trajectory: Dict) -> torch.Tensor:
        """
        Extract spectral signature of a trajectory
        
        Returns vector representation that can be compared to known harmful patterns
        """
        # Encode trajectory
        encoded_trajectory = self.trajectory_encoder(
            trajectory['attention_patterns'],
            trajectory['hidden_states'],
            trajectory['token_probs']
        )
        
        # Apply wavelet transform
        wavelet_coeffs = self.wavelet_transform(encoded_trajectory)
        
        # Extract spectral features
        spectral_features = []
        
        for band_name, coeffs in wavelet_coeffs.items():
            # Compute spectral statistics
            mean_energy = torch.mean(coeffs ** 2, dim=(1, 2))
            spectral_centroid = self._compute_spectral_centroid(coeffs)
            spectral_bandwidth = self._compute_spectral_bandwidth(coeffs)
            
            band_features = torch.cat([
                mean_energy,
                spectral_centroid,
                spectral_bandwidth
            ])
            spectral_features.append(band_features)
            
        # Concatenate all band features
        spectral_signature = torch.cat(spectral_features, dim=-1)
        
        return spectral_signature
        
    def _compute_spectral_centroid(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroid for frequency band"""
        # coeffs: [batch, timesteps, features]
        magnitude = torch.abs(coeffs)
        
        # Compute centroid along time dimension
        timesteps = torch.arange(coeffs.size(1), dtype=torch.float32)
        centroid = torch.sum(magnitude * timesteps.unsqueeze(0).unsqueeze(-1), dim=1) / torch.sum(magnitude, dim=1)
        
        return centroid.mean(dim=-1)  # Average across features
        
    def _compute_spectral_bandwidth(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Compute spectral bandwidth for frequency band"""
        # coeffs: [batch, timesteps, features]
        magnitude = torch.abs(coeffs)
        
        # Compute bandwidth as standard deviation of magnitude distribution
        bandwidth = torch.std(magnitude, dim=1)
        
        return bandwidth.mean(dim=-1)  # Average across features
        
    def compute_attention_entropy(self, attention_weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute attention entropy for interpretability
        
        Returns:
            Dictionary with entropy values for each frequency band
        """
        entropies = {}
        
        for band_name, weights in attention_weights.items():
            # weights: [batch, num_heads, timesteps, timesteps]
            
            # Compute entropy for each head
            head_entropies = []
            for head in range(weights.size(1)):
                head_weights = weights[0, head]  # [timesteps, timesteps]
                
                # Compute entropy for each timestep
                timestep_entropies = []
                for t in range(head_weights.size(0)):
                    attention_dist = F.softmax(head_weights[t], dim=-1)
                    entropy = -torch.sum(attention_dist * torch.log(attention_dist + 1e-10))
                    timestep_entropies.append(entropy.item())
                    
                head_entropies.append(np.mean(timestep_entropies))
                
            entropies[band_name] = np.mean(head_entropies)
            
        return entropies
        
    def get_frequency_band_importance(self) -> Dict[str, float]:
        """Get learned importance weights for each frequency band"""
        weights = torch.softmax(self.band_weights, dim=0)
        
        importance = {}
        for i, weight in enumerate(weights):
            importance[f'level_{i}'] = weight.item()
            
        return importance
        
    def visualize_attention_patterns(self, attention_weights: Dict[str, torch.Tensor], 
                                   save_path: Optional[str] = None) -> None:
        """Visualize attention patterns across frequency bands"""
        import matplotlib.pyplot as plt
        
        num_bands = len(attention_weights)
        fig, axes = plt.subplots(1, num_bands, figsize=(4 * num_bands, 4))
        
        if num_bands == 1:
            axes = [axes]
            
        for i, (band_name, weights) in enumerate(attention_weights.items()):
            # Average across heads and batch
            avg_weights = weights[0].mean(dim=0)  # [timesteps, timesteps]
            
            im = axes[i].imshow(avg_weights.cpu().numpy(), cmap='Blues')
            axes[i].set_title(f'Attention: {band_name}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[i])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ASANEnsemble(nn.Module):
    """
    Ensemble of ASAN predictors for improved robustness
    """
    
    def __init__(self, configs: List[ASANConfig], num_models: int = 3):
        super().__init__()
        
        self.models = nn.ModuleList([
            ASANPredictor(config) for config in configs
        ])
        
        # Ensemble fusion
        self.fusion = nn.Sequential(
            nn.Linear(num_models, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, attention_patterns: Dict, hidden_states: Dict, 
                token_probs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble"""
        
        # Get predictions from each model
        individual_predictions = []
        for model in self.models:
            pred = model(attention_patterns, hidden_states, token_probs)
            individual_predictions.append(pred['harm_probability'])
            
        # Stack predictions
        stacked_predictions = torch.stack(individual_predictions, dim=-1)
        
        # Ensemble fusion
        ensemble_prob = self.fusion(stacked_predictions).squeeze(-1)
        
        # Use first model's other outputs
        base_prediction = self.models[0](attention_patterns, hidden_states, token_probs)
        base_prediction['harm_probability'] = ensemble_prob
        
        return base_prediction
