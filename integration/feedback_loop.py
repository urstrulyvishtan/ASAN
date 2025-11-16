"""
Feedback Loop

Close the loop between prediction, steering, and learning to enable
continual improvement.
"""

import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import numpy as np

try:
    from models.asan_predictor import ASANPredictor
    from alignment.spectral_steering import SpectralSteeringController
except ImportError:
    from ..models.asan_predictor import ASANPredictor
    from ..alignment.spectral_steering import SpectralSteeringController


@dataclass
class FeedbackLoopConfig:
    """Configuration for feedback loop"""
    enable_online_learning: bool = True
    enable_steering_adjustment: bool = True
    feedback_window: int = 100  # Number of interactions to consider
    adaptation_rate: float = 0.1  # How quickly to adapt


class FeedbackLoop:
    """
    Close loop between prediction, steering, and learning
    
    Process:
    1. ASAN predicts harm probability
    2. Steering system intervenes if needed
    3. User/Moderator provides feedback
    4. System learns from feedback
    5. Adjusts prediction thresholds and steering parameters
    6. Repeat
    """
    
    def __init__(self,
                 asan_predictor: ASANPredictor,
                 steering_controller: Optional[SpectralSteeringController],
                 config: FeedbackLoopConfig = None):
        """Initialize feedback loop"""
        self.asan_predictor = asan_predictor
        self.steering_controller = steering_controller
        self.config = config or FeedbackLoopConfig()
        
        # Feedback history
        self.feedback_history = deque(maxlen=self.config.feedback_window)
        
        # Performance metrics
        self.prediction_accuracy = deque(maxlen=1000)
        self.steering_effectiveness = deque(maxlen=1000)
        
        # Adaptive thresholds
        self.harm_threshold = 0.5
        self.steering_strength = 0.3
    
    def process_feedback(self, feedback: Dict[str, Any]):
        """
        Process feedback from user or moderator
        
        Feedback can include:
        - Correctness of prediction (true/false positive/negative)
        - Quality of generated output
        - Effectiveness of steering
        - User satisfaction
        """
        # Store feedback
        self.feedback_history.append(feedback)
        
        # Update metrics
        if 'prediction_correct' in feedback:
            self.prediction_accuracy.append(feedback['prediction_correct'])
        
        if 'steering_effective' in feedback:
            self.steering_effectiveness.append(feedback['steering_effective'])
        
        # Adapt thresholds and parameters
        if len(self.feedback_history) >= 10:
            self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adapt prediction thresholds and steering parameters based on feedback"""
        if not self.feedback_history:
            return
        
        # Analyze recent feedback
        recent_feedback = list(self.feedback_history)[-50:]
        
        # Compute false positive and false negative rates
        false_positives = sum(
            1 for f in recent_feedback
            if f.get('prediction_type') == 'false_positive'
        )
        false_negatives = sum(
            1 for f in recent_feedback
            if f.get('prediction_type') == 'false_negative'
        )
        total_predictions = len(recent_feedback)
        
        if total_predictions == 0:
            return
        
        fp_rate = false_positives / total_predictions
        fn_rate = false_negatives / total_predictions
        
        # Adjust harm threshold
        # If too many false positives, raise threshold (be less sensitive)
        if fp_rate > 0.2:
            self.harm_threshold = min(0.8, self.harm_threshold + self.config.adaptation_rate * 0.1)
        
        # If too many false negatives, lower threshold (be more sensitive)
        elif fn_rate > 0.2:
            self.harm_threshold = max(0.2, self.harm_threshold - self.config.adaptation_rate * 0.1)
        
        # Adjust steering strength
        if self.steering_controller is not None:
            steering_effective_rate = sum(
                1 for f in recent_feedback
                if f.get('steering_effective', False)
            ) / total_predictions
            
            # If steering is too weak, increase strength
            if steering_effective_rate < 0.5:
                self.steering_strength = min(
                    0.8,
                    self.steering_strength + self.config.adaptation_rate * 0.05
                )
                self.steering_controller.config.steering_strength = self.steering_strength
            
            # If steering is too strong (causing quality issues), decrease strength
            elif steering_effective_rate > 0.9:
                quality_issues = sum(
                    1 for f in recent_feedback
                    if f.get('quality_degraded', False)
                ) / total_predictions
                
                if quality_issues > 0.3:
                    self.steering_strength = max(
                        0.1,
                        self.steering_strength - self.config.adaptation_rate * 0.05
                    )
                    self.steering_controller.config.steering_strength = self.steering_strength
    
    def get_adaptive_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds"""
        return {
            'harm_threshold': self.harm_threshold,
            'steering_strength': self.steering_strength
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics from feedback loop"""
        if not self.prediction_accuracy:
            return {
                'prediction_accuracy': 0.0,
                'steering_effectiveness': 0.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0
            }
        
        accuracy = np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0.0
        effectiveness = np.mean(self.steering_effectiveness) if self.steering_effectiveness else 0.0
        
        # Compute rates from recent feedback
        recent_feedback = list(self.feedback_history)[-100:]
        if recent_feedback:
            fp_count = sum(1 for f in recent_feedback if f.get('prediction_type') == 'false_positive')
            fn_count = sum(1 for f in recent_feedback if f.get('prediction_type') == 'false_negative')
            total = len(recent_feedback)
            
            fp_rate = fp_count / total if total > 0 else 0.0
            fn_rate = fn_count / total if total > 0 else 0.0
        else:
            fp_rate = 0.0
            fn_rate = 0.0
        
        return {
            'prediction_accuracy': accuracy,
            'steering_effectiveness': effectiveness,
            'false_positive_rate': fp_rate,
            'false_negative_rate': fn_rate
        }
    
    def should_trigger_retraining(self) -> bool:
        """Check if performance has degraded enough to trigger retraining"""
        metrics = self.get_performance_metrics()
        
        # Trigger retraining if:
        # - Accuracy drops below threshold
        # - False negative rate too high (safety critical)
        if metrics['prediction_accuracy'] < 0.7:
            return True
        
        if metrics['false_negative_rate'] > 0.15:  # Too many missed harmful outputs
            return True
        
        return False
