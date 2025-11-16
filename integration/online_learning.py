"""
Online Learning & Feedback Loop

Continually improve ASAN and steering from real-world interactions.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import numpy as np

try:
    from models.asan_predictor import ASANPredictor
    from alignment.spectral_steering import SpectralSteeringController
    from integration.feedback_loop import FeedbackLoop
except ImportError:
    from ..models.asan_predictor import ASANPredictor
    from ..alignment.spectral_steering import SpectralSteeringController
    from .feedback_loop import FeedbackLoop


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning"""
    learning_rate: float = 1e-5  # Lower than training to avoid catastrophic forgetting
    batch_size: int = 32
    update_frequency: int = 100  # Update every N interactions
    memory_size: int = 1000  # Store last N interactions
    adaptation_threshold: float = 0.1  # Trigger retraining if error exceeds threshold


class OnlineLearningSystem:
    """
    Close the loop: learn from every interaction to improve over time
    
    Learning signals:
    - User feedback on generations
    - Cases where steering was needed
    - False positives (over-cautious steering)
    - Novel attack patterns encountered
    """
    
    def __init__(self, 
                 asan_predictor: ASANPredictor,
                 steering_controller: Optional[SpectralSteeringController],
                 policy: Optional[nn.Module],
                 config: OnlineLearningConfig):
        """Initialize online learning system"""
        self.asan_predictor = asan_predictor
        self.steering_controller = steering_controller
        self.policy = policy
        self.config = config
        
        # Interaction memory
        self.interaction_memory = deque(maxlen=config.memory_size)
        
        # Error tracking
        self.prediction_errors = deque(maxlen=1000)
        self.steering_mistakes = deque(maxlen=1000)
        
        # Distribution shift detection
        self.recent_patterns = deque(maxlen=500)
        
        # Feedback loop
        self.feedback_loop = FeedbackLoop(asan_predictor, steering_controller)
        
        # Optimizers for online updates
        if self.asan_predictor is not None:
            self.asan_optimizer = torch.optim.Adam(
                self.asan_predictor.parameters(),
                lr=config.learning_rate
            )
        
        self.interaction_count = 0
    
    def process_interaction(self, interaction_data: Dict[str, Any]):
        """
        Learn from single interaction
        
        Interaction data includes:
        - Input prompt and conversation history
        - Generated response
        - Internal trajectories (attention, hidden states)
        - ASAN predictions
        - Steering actions taken
        - User feedback (if available)
        - Human moderator judgment (if flagged)
        
        Learning opportunities:
        1. Update ASAN if prediction was wrong
        2. Adjust steering strategy if over/under-steered
        3. Update policy if better response was possible
        """
        # Store interaction
        self.interaction_memory.append(interaction_data)
        self.interaction_count += 1
        
        # Extract feedback signals
        asan_prediction = interaction_data.get('asan_prediction', {})
        actual_label = interaction_data.get('actual_label', None)  # True label if available
        user_feedback = interaction_data.get('user_feedback', None)
        moderator_judgment = interaction_data.get('moderator_judgment', None)
        steering_applied = interaction_data.get('steering_applied', False)
        
        # Check for prediction errors
        if actual_label is not None:
            predicted_harm_prob = asan_prediction.get('harm_probability', 0.0)
            actual_harm = 1.0 if actual_label == 'harmful' else 0.0
            
            error = abs(predicted_harm_prob - actual_harm)
            self.prediction_errors.append(error)
            
            # If significant error, trigger update
            if error > self.config.adaptation_threshold:
                self._update_asan_prediction(interaction_data, actual_label)
        
        # Check for steering mistakes
        if steering_applied:
            # False positive: steered when not needed
            if actual_label is not None and actual_label == 'safe':
                self.steering_mistakes.append({
                    'type': 'false_positive',
                    'interaction': interaction_data
                })
            # False negative: should have steered but didn't
            elif actual_label is not None and actual_label == 'harmful' and not steering_applied:
                self.steering_mistakes.append({
                    'type': 'false_negative',
                    'interaction': interaction_data
                })
        
        # Check for distribution shift
        self._update_pattern_tracking(interaction_data)
        
        # Periodic batch updates
        if self.interaction_count % self.config.update_frequency == 0:
            self.continual_learning_update(
                list(self.interaction_memory)[-self.config.batch_size:]
            )
    
    def _update_asan_prediction(self, interaction_data: Dict[str, Any], actual_label: str):
        """Update ASAN predictor based on feedback"""
        if self.asan_predictor is None:
            return
        
        # Prepare training example
        trajectory = {
            'attention_patterns': interaction_data.get('attention_patterns', {}),
            'hidden_states': interaction_data.get('hidden_states', {}),
            'token_probs': interaction_data.get('token_probs', [])
        }
        
        # Compute loss
        self.asan_predictor.train()
        self.asan_optimizer.zero_grad()
        
        # Compute prediction (no torch.no_grad so gradients can flow)
        prediction = self.asan_predictor(
            trajectory['attention_patterns'],
            trajectory['hidden_states'],
            trajectory['token_probs']
        )
        
        # Compute error (simplified - would use proper loss function)
        target = torch.tensor(1.0 if actual_label == 'harmful' else 0.0, 
                              dtype=prediction['harm_probability'].dtype,
                              device=prediction['harm_probability'].device)
        predicted = prediction['harm_probability']
        
        loss = nn.functional.mse_loss(predicted, target)
        
        # Gradient update (with small step to avoid catastrophic forgetting)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.asan_predictor.parameters(), 1.0)
            self.asan_optimizer.step()
        
        self.asan_predictor.eval()
    
    def _update_pattern_tracking(self, interaction_data: Dict[str, Any]):
        """Track patterns to detect distribution shift"""
        # Extract spectral signature
        trajectory = {
            'attention_patterns': interaction_data.get('attention_patterns', {}),
            'hidden_states': interaction_data.get('hidden_states', {}),
            'token_probs': interaction_data.get('token_probs', [])
        }
        
        with torch.no_grad():
            signature = self.asan_predictor.get_spectral_signature(trajectory)
            self.recent_patterns.append(signature.cpu().numpy())
    
    def detect_distribution_shift(self, recent_interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Monitor for changes in attack patterns or user behavior
        
        If detected:
        - Trigger retraining of ASAN on new patterns
        - Adjust steering thresholds
        - Update RL policy
        """
        if len(self.recent_patterns) < 100:
            return {'shift_detected': False}
        
        # Compare recent patterns to historical patterns
        recent = np.array(list(self.recent_patterns)[-100:])
        historical = np.array(list(self.recent_patterns)[:-100])
        
        if len(historical) == 0:
            return {'shift_detected': False}
        
        # Compute centroid shift
        recent_centroid = np.mean(recent, axis=0)
        historical_centroid = np.mean(historical, axis=0)
        
        shift_magnitude = np.linalg.norm(recent_centroid - historical_centroid)
        
        # Threshold for shift detection
        shift_threshold = 1.0  # Adjust based on data
        
        shift_detected = shift_magnitude > shift_threshold
        
        result = {
            'shift_detected': shift_detected,
            'shift_magnitude': shift_magnitude,
            'threshold': shift_threshold
        }
        
        if shift_detected:
            result['recommendation'] = 'trigger_retraining'
        
        return result
    
    def continual_learning_update(self, interaction_batch: List[Dict[str, Any]]):
        """
        Periodic batch updates from accumulated interactions
        
        Balance:
        - Learning from new patterns (adaptation)
        - Retaining performance on old patterns (stability)
        
        Use techniques:
        - Elastic weight consolidation
        - Experience replay
        - Progressive neural networks
        """
        if not interaction_batch:
            return
        
        # Filter interactions with labels
        labeled_interactions = [
            interaction for interaction in interaction_batch
            if interaction.get('actual_label') is not None
        ]
        
        if len(labeled_interactions) < 10:
            return  # Not enough labeled data
        
        # Update ASAN predictor
        if self.asan_predictor is not None:
            self._batch_update_asan(labeled_interactions)
        
        # Adjust steering if needed
        if self.steering_controller is not None:
            self._adjust_steering_strategy()
    
    def _batch_update_asan(self, labeled_interactions: List[Dict[str, Any]]):
        """Batch update ASAN predictor"""
        self.asan_predictor.train()
        
        total_loss = 0.0
        for interaction in labeled_interactions:
            trajectory = {
                'attention_patterns': interaction.get('attention_patterns', {}),
                'hidden_states': interaction.get('hidden_states', {}),
                'token_probs': interaction.get('token_probs', [])
            }
            
            self.asan_optimizer.zero_grad()
            
            prediction = self.asan_predictor(
                trajectory['attention_patterns'],
                trajectory['hidden_states'],
                trajectory['token_probs']
            )
            
            target = torch.tensor(1.0 if interaction['actual_label'] == 'harmful' else 0.0)
            predicted = prediction['harm_probability']
            
            loss = nn.functional.mse_loss(predicted, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.asan_predictor.parameters(), 1.0)
            self.asan_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(labeled_interactions)
        print(f"Online learning update: avg_loss={avg_loss:.4f}")
        
        self.asan_predictor.eval()
    
    def _adjust_steering_strategy(self):
        """Adjust steering strategy based on mistakes"""
        false_positives = [
            mistake['interaction'] for mistake in self.steering_mistakes
            if mistake['type'] == 'false_positive'
        ]
        false_negatives = [
            mistake['interaction'] for mistake in self.steering_mistakes
            if mistake['type'] == 'false_negative'
        ]
        
        # If too many false positives, reduce steering aggressiveness
        if len(false_positives) > len(false_negatives) * 2:
            if self.steering_controller is not None:
                # Reduce steering strength
                self.steering_controller.config.steering_strength *= 0.95
        
        # If too many false negatives, increase steering aggressiveness
        elif len(false_negatives) > len(false_positives) * 2:
            if self.steering_controller is not None:
                # Increase steering strength
                self.steering_controller.config.steering_strength *= 1.05
                self.steering_controller.config.steering_strength = min(
                    self.steering_controller.config.steering_strength, 1.0
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get online learning statistics"""
        return {
            'interaction_count': self.interaction_count,
            'memory_size': len(self.interaction_memory),
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            'false_positives': sum(1 for m in self.steering_mistakes if m['type'] == 'false_positive'),
            'false_negatives': sum(1 for m in self.steering_mistakes if m['type'] == 'false_negative'),
            'unique_patterns': len(self.recent_patterns)
        }
