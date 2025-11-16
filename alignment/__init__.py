"""
Inference-Time Alignment and Steering

Use ASAN predictions to actively steer model behavior during generation.
"""

from .spectral_steering import SpectralSteeringController, InferenceTimeSteering
from .attention_modulation import AttentionModulator
from .hidden_state_correction import HiddenStateCorrector
from .trajectory_optimization import TrajectoryOptimizer
from .steering_strategies import SteeringStrategy

__all__ = [
    'SpectralSteeringController',
    'InferenceTimeSteering',
    'AttentionModulator',
    'HiddenStateCorrector',
    'TrajectoryOptimizer',
    'SteeringStrategy'
]
