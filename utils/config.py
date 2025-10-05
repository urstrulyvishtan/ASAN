"""
Configuration and Helper Utilities for ASAN

Centralized configuration management and utility functions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime


@dataclass
class ASANConfig:
    """Main configuration for ASAN system"""
    
    # Model architecture
    num_layers: int = 12
    num_heads: int = 8
    hidden_dim: int = 768
    vocab_size: int = 50257
    max_seq_len: int = 1024
    
    # ASAN specific
    encoding_dim: int = 256
    attention_dim_internal: int = 128
    attention_heads: int = 8
    decomposition_levels: int = 4
    wavelet: str = 'db4'
    num_harm_categories: int = 5
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    dropout: float = 0.1
    
    # Monitoring
    intervention_threshold: float = 0.7
    confidence_threshold: float = 0.8
    prediction_frequency: str = 'every_token'
    
    # Paths
    data_path: str = "data"
    model_path: str = "models"
    log_path: str = "logs"
    results_path: str = "results"
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Create directories
        for path in [self.data_path, self.model_path, self.log_path, self.results_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
            
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.log_path) / f"asan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        
    def save_config(self, path: str):
        """Save configuration to file"""
        config_dict = {
            'model_architecture': {
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'hidden_dim': self.hidden_dim,
                'vocab_size': self.vocab_size,
                'max_seq_len': self.max_seq_len
            },
            'asan_specific': {
                'encoding_dim': self.encoding_dim,
                'attention_dim_internal': self.attention_dim_internal,
                'attention_heads': self.attention_heads,
                'decomposition_levels': self.decomposition_levels,
                'wavelet': self.wavelet,
                'num_harm_categories': self.num_harm_categories
            },
            'training': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'dropout': self.dropout
            },
            'monitoring': {
                'intervention_threshold': self.intervention_threshold,
                'confidence_threshold': self.confidence_threshold,
                'prediction_frequency': self.prediction_frequency
            },
            'paths': {
                'data_path': self.data_path,
                'model_path': self.model_path,
                'log_path': self.log_path,
                'results_path': self.results_path
            },
            'device': self.device,
            'log_level': self.log_level
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    @classmethod
    def load_config(cls, path: str):
        """Load configuration from file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
        # Create config object
        config = cls()
        
        # Update with loaded values
        if 'model_architecture' in config_dict:
            for key, value in config_dict['model_architecture'].items():
                setattr(config, key, value)
                
        if 'asan_specific' in config_dict:
            for key, value in config_dict['asan_specific'].items():
                setattr(config, key, value)
                
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                setattr(config, key, value)
                
        if 'monitoring' in config_dict:
            for key, value in config_dict['monitoring'].items():
                setattr(config, key, value)
                
        if 'paths' in config_dict:
            for key, value in config_dict['paths'].items():
                setattr(config, key, value)
                
        if 'device' in config_dict:
            config.device = config_dict['device']
            
        if 'log_level' in config_dict:
            config.log_level = config_dict['log_level']
            
        return config


class ConfigManager:
    """Configuration manager for different experiment types"""
    
    @staticmethod
    def get_synthetic_experiment_config() -> ASANConfig:
        """Get configuration for synthetic experiments"""
        config = ASANConfig()
        
        # Optimize for synthetic data
        config.num_epochs = 50
        config.batch_size = 64
        config.learning_rate = 2e-4
        
        return config
        
    @staticmethod
    def get_real_llm_config() -> ASANConfig:
        """Get configuration for real LLM experiments"""
        config = ASANConfig()
        
        # Optimize for real data
        config.num_epochs = 100
        config.batch_size = 16  # Smaller batch for real data
        config.learning_rate = 1e-4
        
        return config
        
    @staticmethod
    def get_production_config() -> ASANConfig:
        """Get configuration for production deployment"""
        config = ASANConfig()
        
        # Optimize for production
        config.intervention_threshold = 0.8  # Higher threshold for production
        config.confidence_threshold = 0.9
        config.prediction_frequency = 'adaptive'
        
        return config


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: torch.nn.Module, path: str, metadata: Optional[Dict] = None):
    """Save model with metadata"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        save_dict['metadata'] = metadata
        
    torch.save(save_dict, path)


def load_model(model: torch.nn.Module, path: str) -> Dict[str, Any]:
    """Load model and return metadata"""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'model_class': checkpoint.get('model_class', 'Unknown'),
        'timestamp': checkpoint.get('timestamp', 'Unknown'),
        'metadata': checkpoint.get('metadata', {})
    }


class MetricsTracker:
    """Track and log metrics during training/evaluation"""
    
    def __init__(self, name: str = "metrics"):
        self.name = name
        self.metrics = {}
        self.history = []
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.copy()
        })
        
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
        
    def get_average(self, metric_name: str, last_n: int = 10) -> Optional[float]:
        """Get average of last N values for a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            values = self.metrics[metric_name][-last_n:]
            return sum(values) / len(values)
        return None
        
    def save_history(self, path: str):
        """Save metrics history to file"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot metrics over time"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        metric_names = list(self.metrics.keys())
        
        for i, metric_name in enumerate(metric_names[:4]):  # Plot first 4 metrics
            if i < len(axes):
                axes[i].plot(self.metrics[metric_name])
                axes[i].set_title(metric_name)
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel('Value')
                axes[i].grid(True)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ProgressTracker:
    """Track progress of long-running tasks"""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, step: int = 1):
        """Update progress"""
        self.current_step += step
        
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        progress_percent = (self.current_step / self.total_steps) * 100
        elapsed_time = datetime.now() - self.start_time
        
        if self.current_step > 0:
            estimated_total_time = elapsed_time * (self.total_steps / self.current_step)
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = None
            
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percent': progress_percent,
            'elapsed_time': str(elapsed_time),
            'remaining_time': str(remaining_time) if remaining_time else "Unknown"
        }
        
    def print_progress(self):
        """Print current progress"""
        progress = self.get_progress()
        print(f"{self.description}: {progress['current_step']}/{progress['total_steps']} "
              f"({progress['progress_percent']:.1f}%) - "
              f"Elapsed: {progress['elapsed_time']}, "
              f"Remaining: {progress['remaining_time']}")


def create_experiment_directory(experiment_name: str, base_path: str = "experiments") -> Path:
    """Create directory for experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(base_path) / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "models").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    
    return experiment_dir


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_memory(bytes: int) -> str:
    """Format memory in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}PB"


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if early stopping should be triggered"""
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


def validate_config(config: ASANConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    
    # Check model architecture
    if config.num_layers <= 0:
        issues.append("num_layers must be positive")
        
    if config.num_heads <= 0:
        issues.append("num_heads must be positive")
        
    if config.hidden_dim <= 0:
        issues.append("hidden_dim must be positive")
        
    # Check ASAN specific
    if config.encoding_dim <= 0:
        issues.append("encoding_dim must be positive")
        
    if config.decomposition_levels < 0:
        issues.append("decomposition_levels must be non-negative")
        
    if config.num_harm_categories <= 0:
        issues.append("num_harm_categories must be positive")
        
    # Check training parameters
    if config.batch_size <= 0:
        issues.append("batch_size must be positive")
        
    if config.learning_rate <= 0:
        issues.append("learning_rate must be positive")
        
    if config.num_epochs <= 0:
        issues.append("num_epochs must be positive")
        
    # Check monitoring parameters
    if not 0 <= config.intervention_threshold <= 1:
        issues.append("intervention_threshold must be between 0 and 1")
        
    if not 0 <= config.confidence_threshold <= 1:
        issues.append("confidence_threshold must be between 0 and 1")
        
    return issues


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def benchmark_model(model: torch.nn.Module, input_shape: tuple, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model inference speed"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = input_shape[0] / avg_time  # samples per second
    
    return {
        'avg_inference_time': avg_time,
        'throughput_samples_per_second': throughput
    }
