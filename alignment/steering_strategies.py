"""
Steering Strategies

Different strategies for steering model behavior based on ASAN predictions.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch

from .spectral_steering import SpectralSteeringController, SteeringConfig


class SteeringStrategyType(Enum):
    """Types of steering strategies"""
    CONSERVATIVE = "conservative"  # Minimal steering, preserve quality
    AGGRESSIVE = "aggressive"  # Strong steering, prioritize safety
    ADAPTIVE = "adaptive"  # Adjust based on confidence
    QUALITY_PRESERVING = "quality_preserving"  # Prioritize quality preservation


@dataclass
class StrategyConfig:
    """Configuration for steering strategy"""
    steering_strength: float = 0.3
    harm_prob_threshold: float = 0.5
    max_interventions: int = 5
    quality_preservation_weight: float = 0.5


class SteeringStrategy:
    """
    Base class for steering strategies
    
    Different strategies for deciding when and how to steer based on
    ASAN predictions and output quality considerations
    """
    
    def __init__(self, 
                 steering_controller: SpectralSteeringController,
                 config: StrategyConfig):
        self.steering_controller = steering_controller
        self.config = config
    
    def should_steer(self, 
                    current_trajectory: Dict[str, Any],
                    asan_prediction: Dict[str, torch.Tensor]) -> bool:
        """Decide whether steering is needed"""
        raise NotImplementedError
    
    def compute_steering_parameters(self,
                                   current_trajectory: Dict[str, Any],
                                   asan_prediction: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute steering parameters"""
        raise NotImplementedError


class ConservativeSteeringStrategy(SteeringStrategy):
    """
    Conservative steering: Only intervene when harm is very likely
    
    Characteristics:
    - High threshold for intervention
    - Low steering strength
    - Prioritizes output quality
    """
    
    def should_steer(self, 
                    current_trajectory: Dict[str, Any],
                    asan_prediction: Dict[str, torch.Tensor]) -> bool:
        harm_prob = asan_prediction['harm_probability'].item()
        confidence = asan_prediction.get('confidence', torch.tensor(0.5)).item()
        
        # Only steer if harm is very likely and we're confident
        return harm_prob > 0.8 and confidence > 0.7
    
    def compute_steering_parameters(self,
                                   current_trajectory: Dict[str, Any],
                                   asan_prediction: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return {
            'steering_strength': self.config.steering_strength * 0.5,  # Reduced strength
            'target_safety_level': 0.3,  # Less aggressive target
            'preserve_quality': True
        }


class AggressiveSteeringStrategy(SteeringStrategy):
    """
    Aggressive steering: Intervene early and strongly
    
    Characteristics:
    - Low threshold for intervention
    - High steering strength
    - Prioritizes safety over quality
    """
    
    def should_steer(self, 
                    current_trajectory: Dict[str, Any],
                    asan_prediction: Dict[str, torch.Tensor]) -> bool:
        harm_prob = asan_prediction['harm_probability'].item()
        
        # Steer if harm probability is above threshold
        return harm_prob > self.config.harm_prob_threshold
    
    def compute_steering_parameters(self,
                                   current_trajectory: Dict[str, Any],
                                   asan_prediction: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        harm_prob = asan_prediction['harm_probability'].item()
        
        # Increase steering strength with harm probability
        steering_strength = self.config.steering_strength * (0.5 + harm_prob * 0.5)
        
        return {
            'steering_strength': min(steering_strength, 0.8),  # Cap at 0.8
            'target_safety_level': 0.05,  # Very safe target
            'preserve_quality': False
        }


class AdaptiveSteeringStrategy(SteeringStrategy):
    """
    Adaptive steering: Adjust strategy based on ASAN confidence
    
    Characteristics:
    - Adjusts threshold and strength based on confidence
    - Balances safety and quality dynamically
    """
    
    def should_steer(self, 
                    current_trajectory: Dict[str, Any],
                    asan_prediction: Dict[str, torch.Tensor]) -> bool:
        harm_prob = asan_prediction['harm_probability'].item()
        confidence = asan_prediction.get('confidence', torch.tensor(0.5)).item()
        
        # Adaptive threshold: lower threshold when confidence is high
        threshold = self.config.harm_prob_threshold * (2.0 - confidence)
        
        return harm_prob > threshold
    
    def compute_steering_parameters(self,
                                   current_trajectory: Dict[str, Any],
                                   asan_prediction: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        harm_prob = asan_prediction['harm_probability'].item()
        confidence = asan_prediction.get('confidence', torch.tensor(0.5)).item()
        
        # Steering strength depends on both harm prob and confidence
        base_strength = self.config.steering_strength
        confidence_factor = 0.5 + confidence * 0.5
        harm_factor = 0.5 + harm_prob * 0.5
        
        steering_strength = base_strength * confidence_factor * harm_factor
        
        # Target safety: more aggressive when confident and high harm
        target_safety = 0.1 - (confidence * harm_prob * 0.05)
        
        return {
            'steering_strength': min(steering_strength, 0.7),
            'target_safety_level': max(target_safety, 0.05),
            'preserve_quality': confidence < 0.7  # Preserve quality when uncertain
        }


class QualityPreservingSteeringStrategy(SteeringStrategy):
    """
    Quality-preserving steering: Prioritize maintaining output quality
    
    Characteristics:
    - Only steer when absolutely necessary
    - Use minimal steering to preserve quality
    - Optimize for quality preservation score
    """
    
    def should_steer(self, 
                    current_trajectory: Dict[str, Any],
                    asan_prediction: Dict[str, torch.Tensor]) -> bool:
        harm_prob = asan_prediction['harm_probability'].item()
        confidence = asan_prediction.get('confidence', torch.tensor(0.5)).item()
        
        # Only steer if harm is very likely AND we're confident
        # AND quality preservation score suggests we can improve
        return harm_prob > 0.7 and confidence > 0.8
    
    def compute_steering_parameters(self,
                                   current_trajectory: Dict[str, Any],
                                   asan_prediction: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # Use minimal steering
        return {
            'steering_strength': self.config.steering_strength * 0.3,  # Very low
            'target_safety_level': 0.2,  # Moderate target
            'preserve_quality': True,
            'quality_preservation_weight': 0.8  # High weight on quality
        }


def create_steering_strategy(strategy_type: SteeringStrategyType,
                            steering_controller: SpectralSteeringController,
                            config: StrategyConfig) -> SteeringStrategy:
    """Factory function to create steering strategy"""
    strategies = {
        SteeringStrategyType.CONSERVATIVE: ConservativeSteeringStrategy,
        SteeringStrategyType.AGGRESSIVE: AggressiveSteeringStrategy,
        SteeringStrategyType.ADAPTIVE: AdaptiveSteeringStrategy,
        SteeringStrategyType.QUALITY_PRESERVING: QualityPreservingSteeringStrategy
    }
    
    strategy_class = strategies.get(strategy_type)
    if strategy_class is None:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return strategy_class(steering_controller, config)
