"""
Wavelet Transform for Temporal Analysis in ASAN

Apply wavelet decomposition to temporal sequences of LLM states to separate
fast (syntactic) vs slow (semantic) processing patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class TemporalWaveletTransform(nn.Module):
    """
    Wavelet decomposition of LLM internal state trajectories
    
    Purpose: Separate fast (syntactic) vs slow (semantic) processing patterns
    
    Frequency bands interpretation:
    - High frequency (D1, D2): Token-level syntactic processing
    - Mid frequency (D3, D4): Phrase-level semantic integration
    - Low frequency (A4): Long-range contextual reasoning
    """
    
    def __init__(self, wavelet: str = 'db4', levels: int = 4, mode: str = 'symmetric'):
        """
        Args:
            wavelet: wavelet family ('db4', 'haar', 'sym4', 'coif2')
            levels: decomposition levels (3-5 optimal for LLM analysis)
            mode: padding mode for wavelet transform
        """
        super().__init__()
        
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        
        # Validate wavelet
        try:
            pywt.Wavelet(wavelet)
        except ValueError:
            warnings.warn(f"Invalid wavelet {wavelet}, using 'db4'")
            self.wavelet = 'db4'
            
        # Learnable frequency band weights
        self.band_weights = nn.Parameter(torch.ones(levels + 1))
        
        # Frequency-specific processing layers
        self.frequency_processors = nn.ModuleDict()
        for i in range(levels + 1):
            self.frequency_processors[f'level_{i}'] = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
    def forward(self, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose trajectory into frequency components
        
        Args:
            trajectory: [batch, timesteps, features]
            
        Returns:
            coefficients: dict with keys:
                - 'approximation': low-frequency [batch, timesteps//2^levels, features]
                - 'details_L4': [batch, timesteps//2^4, features]
                - 'details_L3': [batch, timesteps//2^3, features]
                - 'details_L2': [batch, timesteps//2^2, features]
                - 'details_L1': [batch, timesteps//2, features]
        """
        batch_size, timesteps, features = trajectory.shape
        
        # Process each feature dimension separately
        coefficients = {}
        
        for feat_idx in range(features):
            # Extract single feature across time
            feature_series = trajectory[:, :, feat_idx]  # [batch, timesteps]
            
            # Apply wavelet transform to each sample in batch
            batch_coeffs = []
            for b in range(batch_size):
                signal = feature_series[b].cpu().numpy()
                
                try:
                    # Perform wavelet decomposition
                    coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels, mode=self.mode)
                    
                    # Pad coefficients to same length for batching
                    padded_coeffs = self._pad_coefficients(coeffs, timesteps)
                    batch_coeffs.append(padded_coeffs)
                    
                except Exception as e:
                    warnings.warn(f"Wavelet transform failed: {e}")
                    # Fallback: create dummy coefficients
                    dummy_coeffs = self._create_dummy_coefficients(timesteps)
                    batch_coeffs.append(dummy_coeffs)
                    
            # Stack batch coefficients
            batch_coeffs = torch.stack([torch.tensor(coeffs) for coeffs in batch_coeffs])
            
            # Store coefficients for this feature
            for level_idx, level_coeffs in enumerate(batch_coeffs.transpose(1, 2)):
                level_name = f'level_{level_idx}'
                if level_name not in coefficients:
                    coefficients[level_name] = []
                coefficients[level_name].append(level_coeffs)
                
        # Concatenate features for each level
        final_coefficients = {}
        for level_name, level_coeffs_list in coefficients.items():
            # Stack features: [batch, timesteps, features]
            level_coeffs = torch.stack(level_coeffs_list, dim=-1)
            final_coefficients[level_name] = level_coeffs
            
        return final_coefficients
        
    def _pad_coefficients(self, coeffs: List[np.ndarray], target_length: int) -> List[np.ndarray]:
        """Pad coefficients to target length"""
        padded_coeffs = []
        
        for coeff in coeffs:
            if len(coeff) < target_length:
                # Pad with zeros
                padding = np.zeros(target_length - len(coeff))
                padded_coeff = np.concatenate([coeff, padding])
            elif len(coeff) > target_length:
                # Truncate
                padded_coeff = coeff[:target_length]
            else:
                padded_coeff = coeff
                
            padded_coeffs.append(padded_coeff)
            
        return padded_coeffs
        
    def _create_dummy_coefficients(self, timesteps: int) -> List[np.ndarray]:
        """Create dummy coefficients when wavelet transform fails"""
        dummy_coeffs = []
        for level in range(self.levels + 1):
            length = timesteps // (2 ** level)
            dummy_coeff = np.zeros(length)
            dummy_coeffs.append(dummy_coeff)
        return dummy_coeffs
        
    def analyze_frequency_anomalies(self, coefficients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Detect anomalies in specific frequency bands
        
        Harmful patterns often show:
        - Spikes in high-frequency (sudden token-level changes)
        - Disruptions in low-frequency (loss of long-range coherence)
        - Cross-frequency coupling changes
        """
        anomalies = {}
        
        for level_name, coeffs in coefficients.items():
            # coeffs: [batch, timesteps, features]
            
            # Compute anomaly scores
            # 1. Variance anomaly (sudden changes)
            variance_anomaly = torch.var(coeffs, dim=1)  # [batch, features]
            
            # 2. Magnitude anomaly (unusual amplitudes)
            magnitude_anomaly = torch.abs(coeffs).mean(dim=1)  # [batch, features]
            
            # 3. Temporal anomaly (unusual patterns over time)
            temporal_anomaly = self._compute_temporal_anomaly(coeffs)
            
            # Combine anomaly measures
            combined_anomaly = (
                0.4 * variance_anomaly + 
                0.3 * magnitude_anomaly + 
                0.3 * temporal_anomaly
            )
            
            anomalies[level_name] = combined_anomaly
            
        return anomalies
        
    def _compute_temporal_anomaly(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Compute temporal anomaly patterns"""
        # coeffs: [batch, timesteps, features]
        
        # Compute autocorrelation
        autocorr_scores = []
        for b in range(coeffs.size(0)):
            batch_autocorr = []
            for f in range(coeffs.size(2)):
                signal = coeffs[b, :, f]
                # Simple autocorrelation at lag 1
                if len(signal) > 1:
                    autocorr = torch.corrcoef(torch.stack([signal[:-1], signal[1:]]))[0, 1]
                    if not torch.isnan(autocorr):
                        batch_autocorr.append(autocorr.item())
                    else:
                        batch_autocorr.append(0.0)
                else:
                    batch_autocorr.append(0.0)
            autocorr_scores.append(batch_autocorr)
            
        return torch.tensor(autocorr_scores)
        
    def reconstruct_signal(self, coefficients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct signal from wavelet coefficients"""
        batch_size = next(iter(coefficients.values())).size(0)
        features = next(iter(coefficients.values())).size(2)
        
        reconstructed_signals = []
        
        for feat_idx in range(features):
            # Collect coefficients for this feature
            coeffs_list = []
            for level_name in sorted(coefficients.keys()):
                level_coeffs = coefficients[level_name][:, :, feat_idx]  # [batch, timesteps]
                coeffs_list.append(level_coeffs.cpu().numpy())
                
            # Reconstruct signal for each batch
            batch_reconstructed = []
            for b in range(batch_size):
                try:
                    # Prepare coefficients for reconstruction
                    coeffs_for_recon = [coeffs_list[level][b] for level in range(len(coeffs_list))]
                    
                    # Reconstruct using inverse wavelet transform
                    reconstructed = pywt.waverec(coeffs_for_recon, self.wavelet, mode=self.mode)
                    
                    # Ensure correct length
                    target_length = len(coeffs_list[0][b])
                    if len(reconstructed) > target_length:
                        reconstructed = reconstructed[:target_length]
                    elif len(reconstructed) < target_length:
                        padding = np.zeros(target_length - len(reconstructed))
                        reconstructed = np.concatenate([reconstructed, padding])
                        
                    batch_reconstructed.append(reconstructed)
                    
                except Exception as e:
                    warnings.warn(f"Signal reconstruction failed: {e}")
                    # Fallback: use approximation coefficients
                    batch_reconstructed.append(coeffs_list[0][b])
                    
            reconstructed_signals.append(np.stack(batch_reconstructed))
            
        # Stack features: [batch, timesteps, features]
        reconstructed = torch.tensor(np.stack(reconstructed_signals, axis=2))
        
        return reconstructed
        
    def compute_frequency_band_importance(self, coefficients: Dict[str, torch.Tensor], 
                                       labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute importance of each frequency band for classification
        
        Args:
            coefficients: Wavelet coefficients
            labels: Binary labels (0=safe, 1=harmful)
            
        Returns:
            Dictionary with importance scores for each frequency band
        """
        importance_scores = {}
        
        for level_name, coeffs in coefficients.items():
            # coeffs: [batch, timesteps, features]
            
            # Compute mean coefficient magnitude for each sample
            mean_magnitude = torch.abs(coeffs).mean(dim=(1, 2))  # [batch]
            
            # Compute correlation with labels
            if len(torch.unique(labels)) > 1:
                correlation = torch.corrcoef(torch.stack([mean_magnitude, labels.float()]))[0, 1]
                if not torch.isnan(correlation):
                    importance_scores[level_name] = abs(correlation.item())
                else:
                    importance_scores[level_name] = 0.0
            else:
                importance_scores[level_name] = 0.0
                
        return importance_scores
        
    def visualize_frequency_bands(self, coefficients: Dict[str, torch.Tensor], 
                                save_path: Optional[str] = None) -> None:
        """Visualize frequency band decomposition"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(coefficients), 1, figsize=(12, 2 * len(coefficients)))
        if len(coefficients) == 1:
            axes = [axes]
            
        for i, (level_name, coeffs) in enumerate(coefficients.items()):
            # Plot mean coefficient across batch and features
            mean_coeffs = coeffs.mean(dim=(0, 2))  # [timesteps]
            
            axes[i].plot(mean_coeffs.cpu().numpy())
            axes[i].set_title(f'Frequency Band: {level_name}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class AdaptiveWaveletTransform(nn.Module):
    """
    Adaptive wavelet transform that learns optimal wavelet parameters
    """
    
    def __init__(self, input_dim: int, num_wavelets: int = 4):
        super().__init__()
        
        self.num_wavelets = num_wavelets
        self.input_dim = input_dim
        
        # Learnable wavelet parameters
        self.wavelet_params = nn.Parameter(torch.randn(num_wavelets, input_dim))
        
        # Wavelet selection network
        self.selection_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_wavelets),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply adaptive wavelet transform
        
        Args:
            trajectory: [batch, timesteps, features]
            
        Returns:
            Dictionary of frequency band coefficients
        """
        batch_size, timesteps, features = trajectory.shape
        
        # Select wavelet weights for each sample
        # Use mean trajectory as input to selection network
        mean_trajectory = trajectory.mean(dim=1)  # [batch, features]
        wavelet_weights = self.selection_network(mean_trajectory)  # [batch, num_wavelets]
        
        # Apply weighted combination of wavelets
        coefficients = {}
        
        for level in range(4):  # Fixed number of levels
            level_coeffs = []
            
            for b in range(batch_size):
                # Get trajectory for this batch
                batch_trajectory = trajectory[b]  # [timesteps, features]
                
                # Apply weighted wavelet transform
                weighted_coeffs = torch.zeros(timesteps // (2 ** level))
                
                for w in range(self.num_wavelets):
                    weight = wavelet_weights[b, w]
                    
                    # Apply wavelet transform (simplified)
                    wavelet_coeffs = self._apply_wavelet_transform(
                        batch_trajectory, self.wavelet_params[w], level
                    )
                    
                    weighted_coeffs += weight * wavelet_coeffs
                    
                level_coeffs.append(weighted_coeffs)
                
            coefficients[f'level_{level}'] = torch.stack(level_coeffs)
            
        return coefficients
        
    def _apply_wavelet_transform(self, signal: torch.Tensor, 
                               wavelet_params: torch.Tensor, level: int) -> torch.Tensor:
        """Apply wavelet transform with given parameters"""
        # Simplified wavelet transform implementation
        # In practice, this would use learned wavelet filters
        
        timesteps = signal.size(0)
        output_length = timesteps // (2 ** level)
        
        # Apply learned filtering
        filtered = torch.matmul(signal, wavelet_params.unsqueeze(0))
        
        # Downsample
        downsampled = filtered[::(2 ** level)]
        
        # Pad or truncate to target length
        if len(downsampled) > output_length:
            return downsampled[:output_length]
        elif len(downsampled) < output_length:
            padding = torch.zeros(output_length - len(downsampled))
            return torch.cat([downsampled, padding])
        else:
            return downsampled
