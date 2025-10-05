"""
Helper Functions for ASAN

Utility functions for data processing, model utilities, and common operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
from pathlib import Path
import logging
import time
from collections import defaultdict
import warnings


def pad_sequences(sequences: List[torch.Tensor], max_length: Optional[int] = None, 
                 padding_value: float = 0.0) -> torch.Tensor:
    """Pad sequences to the same length"""
    
    if max_length is None:
        max_length = max(seq.size(0) for seq in sequences)
        
    padded_sequences = []
    for seq in sequences:
        if seq.size(0) < max_length:
            padding = torch.full((max_length - seq.size(0), *seq.shape[1:]), 
                               padding_value, dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
        
    return torch.stack(padded_sequences)


def normalize_tensor(tensor: torch.Tensor, dim: int = -1, 
                    method: str = 'l2') -> torch.Tensor:
    """Normalize tensor along specified dimension"""
    
    if method == 'l2':
        norm = torch.norm(tensor, dim=dim, keepdim=True)
        return tensor / (norm + 1e-8)
    elif method == 'l1':
        norm = torch.sum(torch.abs(tensor), dim=dim, keepdim=True)
        return tensor / (norm + 1e-8)
    elif method == 'max':
        max_val = torch.max(torch.abs(tensor), dim=dim, keepdim=True)[0]
        return tensor / (max_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention weights"""
    
    # attention_weights: [batch, heads, seq_len, seq_len]
    # Normalize to probability distribution
    attention_probs = F.softmax(attention_weights, dim=-1)
    
    # Compute entropy
    entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-10), dim=-1)
    
    return entropy


def compute_attention_diversity(attention_weights: torch.Tensor) -> torch.Tensor:
    """Compute diversity of attention patterns across heads"""
    
    # attention_weights: [batch, heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # Compute pairwise cosine similarity between heads
    similarities = []
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            head_i = attention_weights[:, i].flatten(1)  # [batch, seq_len*seq_len]
            head_j = attention_weights[:, j].flatten(1)
            
            # Cosine similarity
            similarity = F.cosine_similarity(head_i, head_j, dim=1)
            similarities.append(similarity)
            
    if similarities:
        avg_similarity = torch.stack(similarities).mean(dim=0)
        diversity = 1 - avg_similarity  # Higher diversity = lower similarity
    else:
        diversity = torch.zeros(batch_size)
        
    return diversity


def extract_attention_statistics(attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Extract various statistics from attention weights"""
    
    stats = {}
    
    # Basic statistics
    stats['mean'] = attention_weights.mean()
    stats['std'] = attention_weights.std()
    stats['max'] = attention_weights.max()
    stats['min'] = attention_weights.min()
    
    # Entropy
    stats['entropy'] = compute_attention_entropy(attention_weights).mean()
    
    # Diversity
    stats['diversity'] = compute_attention_diversity(attention_weights).mean()
    
    # Diagonal attention (self-attention)
    if attention_weights.dim() == 4:  # [batch, heads, seq_len, seq_len]
        diagonal_attention = torch.diagonal(attention_weights, dim1=-2, dim2=-1)
        stats['diagonal_mean'] = diagonal_attention.mean()
        stats['diagonal_std'] = diagonal_attention.std()
        
    # Attention sparsity (fraction of near-zero weights)
    threshold = 0.01
    sparse_mask = attention_weights < threshold
    stats['sparsity'] = sparse_mask.float().mean()
    
    return stats


def compute_hidden_state_statistics(hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute statistics for hidden states"""
    
    stats = {}
    
    # Basic statistics
    stats['mean'] = hidden_states.mean()
    stats['std'] = hidden_states.std()
    stats['max'] = hidden_states.max()
    stats['min'] = hidden_states.min()
    
    # Magnitude
    stats['magnitude'] = torch.norm(hidden_states, dim=-1).mean()
    
    # Activation sparsity
    threshold = 0.1
    active_mask = torch.abs(hidden_states) > threshold
    stats['activation_rate'] = active_mask.float().mean()
    
    # Principal component analysis
    try:
        # Flatten spatial dimensions
        flattened = hidden_states.view(hidden_states.size(0), -1)
        
        # Compute covariance matrix
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        cov_matrix = torch.mm(centered.T, centered) / (centered.size(0) - 1)
        
        # Eigenvalues
        eigenvals = torch.linalg.eigvals(cov_matrix).real
        eigenvals = torch.sort(eigenvals, descending=True)[0]
        
        # Explained variance
        total_variance = eigenvals.sum()
        explained_variance = eigenvals / total_variance
        
        stats['top_eigenvalue'] = eigenvals[0]
        stats['explained_variance_top5'] = explained_variance[:5].sum()
        
    except Exception as e:
        warnings.warn(f"PCA computation failed: {e}")
        stats['top_eigenvalue'] = torch.tensor(0.0)
        stats['explained_variance_top5'] = torch.tensor(0.0)
        
    return stats


def compute_token_probability_statistics(token_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute statistics for token probability distributions"""
    
    stats = {}
    
    # Entropy
    entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10), dim=-1)
    stats['entropy'] = entropy.mean()
    stats['entropy_std'] = entropy.std()
    
    # Top-k probability mass
    for k in [1, 5, 10]:
        top_k_probs = torch.topk(token_probs, k=k, dim=-1)[0]
        top_k_mass = top_k_probs.sum(dim=-1)
        stats[f'top_{k}_mass'] = top_k_mass.mean()
        
    # Concentration measures
    stats['gini_coefficient'] = compute_gini_coefficient(token_probs)
    
    # Confidence (max probability)
    max_probs = token_probs.max(dim=-1)[0]
    stats['confidence'] = max_probs.mean()
    stats['confidence_std'] = max_probs.std()
    
    return stats


def compute_gini_coefficient(probs: torch.Tensor) -> torch.Tensor:
    """Compute Gini coefficient as concentration measure"""
    
    # Sort probabilities
    sorted_probs, _ = torch.sort(probs, dim=-1)
    n = sorted_probs.size(-1)
    
    # Compute Gini coefficient
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    gini = (n + 1 - 2 * torch.sum(cumsum, dim=-1) / cumsum[:, -1]) / n
    
    return gini.mean()


def create_attention_mask(seq_len: int, mask_type: str = 'causal') -> torch.Tensor:
    """Create attention mask"""
    
    if mask_type == 'causal':
        # Lower triangular mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
    elif mask_type == 'padding':
        # All ones (no masking)
        mask = torch.ones(seq_len, seq_len)
    elif mask_type == 'random':
        # Random mask
        mask = torch.rand(seq_len, seq_len) > 0.5
        mask = mask.float()
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
        
    return mask


def apply_attention_mask(attention_weights: torch.Tensor, 
                        mask: torch.Tensor) -> torch.Tensor:
    """Apply attention mask to attention weights"""
    
    # Add large negative value to masked positions
    masked_weights = attention_weights.masked_fill(mask == 0, -1e9)
    
    return masked_weights


def compute_temporal_coherence(trajectory: List[torch.Tensor]) -> float:
    """Compute temporal coherence of a trajectory"""
    
    if len(trajectory) < 2:
        return 0.0
        
    # Compute correlations between consecutive timesteps
    correlations = []
    for i in range(len(trajectory) - 1):
        curr = trajectory[i].flatten()
        next_t = trajectory[i + 1].flatten()
        
        # Ensure same length
        min_len = min(len(curr), len(next_t))
        curr = curr[:min_len]
        next_t = next_t[:min_len]
        
        # Compute correlation
        correlation = torch.corrcoef(torch.stack([curr, next_t]))[0, 1]
        if not torch.isnan(correlation):
            correlations.append(correlation.item())
            
    return np.mean(correlations) if correlations else 0.0


def detect_anomalies(values: List[float], threshold: float = 2.0) -> List[int]:
    """Detect anomalous values using z-score"""
    
    if len(values) < 3:
        return []
        
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    
    if std == 0:
        return []
        
    z_scores = np.abs((values - mean) / std)
    anomalies = np.where(z_scores > threshold)[0].tolist()
    
    return anomalies


def smooth_trajectory(trajectory: List[torch.Tensor], 
                     window_size: int = 3) -> List[torch.Tensor]:
    """Apply moving average smoothing to trajectory"""
    
    if len(trajectory) < window_size:
        return trajectory
        
    smoothed = []
    
    for i in range(len(trajectory)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(trajectory), i + window_size // 2 + 1)
        
        window = trajectory[start_idx:end_idx]
        smoothed_tensor = torch.stack(window).mean(dim=0)
        smoothed.append(smoothed_tensor)
        
    return smoothed


def interpolate_trajectory(trajectory: List[torch.Tensor], 
                          target_length: int) -> List[torch.Tensor]:
    """Interpolate trajectory to target length"""
    
    if len(trajectory) == target_length:
        return trajectory
    elif len(trajectory) == 1:
        return trajectory * target_length
        
    # Create interpolation indices
    original_indices = torch.linspace(0, len(trajectory) - 1, len(trajectory))
    target_indices = torch.linspace(0, len(trajectory) - 1, target_length)
    
    # Stack trajectory tensors
    stacked = torch.stack(trajectory)
    
    # Interpolate along time dimension
    interpolated = F.interpolate(
        stacked.unsqueeze(0).transpose(1, 2),  # Add batch and channel dims
        size=target_length,
        mode='linear',
        align_corners=True
    ).squeeze(0).transpose(0, 1)
    
    return [interpolated[i] for i in range(target_length)]


def compute_trajectory_similarity(traj1: List[torch.Tensor], 
                                 traj2: List[torch.Tensor]) -> float:
    """Compute similarity between two trajectories"""
    
    if len(traj1) != len(traj2):
        # Interpolate to same length
        max_len = max(len(traj1), len(traj2))
        traj1 = interpolate_trajectory(traj1, max_len)
        traj2 = interpolate_trajectory(traj2, max_len)
        
    # Compute element-wise similarity
    similarities = []
    for t1, t2 in zip(traj1, traj2):
        # Flatten tensors
        flat1 = t1.flatten()
        flat2 = t2.flatten()
        
        # Ensure same length
        min_len = min(len(flat1), len(flat2))
        flat1 = flat1[:min_len]
        flat2 = flat2[:min_len]
        
        # Cosine similarity
        similarity = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
        similarities.append(similarity.item())
        
    return np.mean(similarities)


def save_trajectory_data(trajectories: List[Dict], path: str):
    """Save trajectory data to file"""
    
    # Convert tensors to numpy arrays for serialization
    serializable_trajectories = []
    
    for traj in trajectories:
        serializable_traj = {}
        
        for key, value in traj.items():
            if isinstance(value, torch.Tensor):
                serializable_traj[key] = value.cpu().numpy()
            elif isinstance(value, dict):
                serializable_traj[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list) and subvalue and isinstance(subvalue[0], torch.Tensor):
                        serializable_traj[key][subkey] = [v.cpu().numpy() for v in subvalue]
                    else:
                        serializable_traj[key][subkey] = subvalue
            else:
                serializable_traj[key] = value
                
        serializable_trajectories.append(serializable_traj)
        
    # Save as pickle for efficiency
    with open(path, 'wb') as f:
        pickle.dump(serializable_trajectories, f)


def load_trajectory_data(path: str) -> List[Dict]:
    """Load trajectory data from file"""
    
    with open(path, 'rb') as f:
        trajectories = pickle.load(f)
        
    # Convert numpy arrays back to tensors
    tensor_trajectories = []
    
    for traj in trajectories:
        tensor_traj = {}
        
        for key, value in traj.items():
            if isinstance(value, np.ndarray):
                tensor_traj[key] = torch.from_numpy(value)
            elif isinstance(value, dict):
                tensor_traj[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list) and subvalue and isinstance(subvalue[0], np.ndarray):
                        tensor_traj[key][subkey] = [torch.from_numpy(v) for v in subvalue]
                    else:
                        tensor_traj[key][subkey] = subvalue
            else:
                tensor_traj[key] = value
                
        tensor_trajectories.append(tensor_traj)
        
    return tensor_trajectories


def create_batch_from_trajectories(trajectories: List[Dict]) -> Dict[str, torch.Tensor]:
    """Create batch from list of trajectories"""
    
    batch = {
        'attention_patterns': defaultdict(list),
        'hidden_states': defaultdict(list),
        'token_probs': [],
        'labels': [],
        'categories': []
    }
    
    # Find maximum sequence length
    max_length = max(len(traj['generated_tokens']) for traj in trajectories)
    
    for traj in trajectories:
        # Pad sequences
        seq_len = len(traj['generated_tokens'])
        
        # Attention patterns
        for layer_idx, attn_list in traj['attention_patterns'].items():
            padded_attn = pad_sequences(attn_list, max_length)
            batch['attention_patterns'][layer_idx].append(padded_attn)
            
        # Hidden states
        for layer_idx, hidden_list in traj['hidden_states'].items():
            padded_hidden = pad_sequences(hidden_list, max_length)
            batch['hidden_states'][layer_idx].append(padded_hidden)
            
        # Token probabilities
        padded_probs = pad_sequences(traj['token_probs'], max_length)
        batch['token_probs'].append(padded_probs)
        
        # Labels
        label = 1 if traj['label'] == 'harmful' else 0
        batch['labels'].append(label)
        
        # Categories
        category = traj.get('category', 0)
        batch['categories'].append(category)
        
    # Stack tensors
    for layer_idx in batch['attention_patterns']:
        batch['attention_patterns'][layer_idx] = torch.stack(batch['attention_patterns'][layer_idx])
        
    for layer_idx in batch['hidden_states']:
        batch['hidden_states'][layer_idx] = torch.stack(batch['hidden_states'][layer_idx])
        
    batch['token_probs'] = torch.stack(batch['token_probs'])
    batch['labels'] = torch.tensor(batch['labels'])
    batch['categories'] = torch.tensor(batch['categories'])
    
    return batch


def compute_model_complexity(model: nn.Module) -> Dict[str, Any]:
    """Compute model complexity metrics"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers
    num_layers = len(list(model.modules()))
    
    # Model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'num_layers': num_layers,
        'model_size_mb': model_size_mb
    }


def benchmark_inference_speed(model: nn.Module, input_shapes: Dict[str, tuple], 
                            num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model inference speed"""
    
    model.eval()
    
    # Create dummy inputs
    dummy_inputs = {}
    for key, shape in input_shapes.items():
        dummy_inputs[key] = torch.randn(shape)
        
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**dummy_inputs)
            
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(**dummy_inputs)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = input_shapes[list(input_shapes.keys())[0]][0] / avg_time
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'throughput_samples_per_second': throughput
    }
