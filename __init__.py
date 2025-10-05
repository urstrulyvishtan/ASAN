"""
ASAN: Adaptive Spectral Alignment Networks for Predictive AI Safety

A novel framework that analyzes temporal patterns in LLM internal states
(attention patterns, hidden representations, token generation sequences)
to predict and prevent problematic outputs BEFORE they are generated.

This transforms AI safety from reactive output monitoring to proactive behavioral prediction.
"""

__version__ = "0.1.0"
__author__ = "Sibi Vishtan Thirukonda"
__email__ = "sibivishtan@gmail.com"

# Core imports
from .models.asan_predictor import ASANPredictor, ASANConfig
from .models.trajectory_encoder import TrajectoryEncoder
from .models.wavelets import TemporalWaveletTransform

from .data.llm_trajectory_collector import LLMTrajectoryCollector
from .data.synthetic_llm_simulator import SyntheticLLMSimulator, SimulatorConfig
from .data.harmful_patterns_dataset import HarmfulPatternsDataset, SafePatternsDataset

from .llm_integration.real_time_monitor import RealTimeASANMonitor, MonitoringConfig

from .training.train_predictor import ASANTrainingPipeline, TrainingConfig

from .evaluation.metrics import comprehensive_evaluation, EvaluationConfig

from .utils.config import ASANConfig as Config, ConfigManager
from .utils.helpers import *

# Safety benchmarks
from .safety_benchmarks.harmful_instruction_following import run_all_safety_benchmarks, BenchmarkConfig

# Visualization
from .visualization.prediction_dashboard import ASANDashboard

# Experiments
from .experiments.synthetic_llm_experiments import run_all_experiments, ExperimentConfig

__all__ = [
    # Core models
    'ASANPredictor',
    'ASANConfig', 
    'TrajectoryEncoder',
    'TemporalWaveletTransform',
    
    # Data components
    'LLMTrajectoryCollector',
    'SyntheticLLMSimulator',
    'SimulatorConfig',
    'HarmfulPatternsDataset',
    'SafePatternsDataset',
    
    # Integration
    'RealTimeASANMonitor',
    'MonitoringConfig',
    
    # Training
    'ASANTrainingPipeline',
    'TrainingConfig',
    
    # Evaluation
    'comprehensive_evaluation',
    'EvaluationConfig',
    
    # Configuration
    'Config',
    'ConfigManager',
    
    # Benchmarks
    'run_all_safety_benchmarks',
    'BenchmarkConfig',
    
    # Visualization
    'ASANDashboard',
    
    # Experiments
    'run_all_experiments',
    'ExperimentConfig',
]
