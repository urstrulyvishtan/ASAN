"""
ASAN as RL Reward Model

Use ASAN's predictive capabilities to provide reward signals for RL training
that optimize for long-term safety, not just immediate token safety.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from ..models.asan_predictor import ASANPredictor


@dataclass
class RewardConfig:
    """Configuration for ASAN reward model"""
    # Reward shaping
    reward_shaping: str = "linear"  # "linear", "exponential", "tanh"
    temporal_discounting: float = 0.9  # Weight recent vs early predictions
    multi_objective_weights: Dict[str, float] = None  # Safety, helpfulness, coherence
    
    # Reward components
    safety_weight: float = 1.0
    trajectory_smoothness_weight: float = 0.2
    early_detection_bonus_weight: float = 0.3
    quality_preservation_weight: float = 0.5
    
    # Reward range
    min_reward: float = -1.0
    max_reward: float = 1.0
    
    def __post_init__(self):
        if self.multi_objective_weights is None:
            self.multi_objective_weights = {
                'safety': 1.0,
                'helpfulness': 0.3,
                'coherence': 0.2
            }


class ASANRewardModel:
    """
    Use ASAN predictions as reward signal for RL training
    
    Key Innovation:
    - Traditional RLHF uses human preferences on final outputs
    - ASAN provides dense rewards throughout generation
    - Can optimize for trajectory safety, not just output safety
    """
    
    def __init__(self, asan_predictor: ASANPredictor, config: RewardConfig):
        """
        Args:
            asan_predictor: Trained ASAN model
            config: Reward configuration
        """
        self.asan_predictor = asan_predictor
        self.config = config
        self.asan_predictor.eval()
    
    def compute_trajectory_reward(self, 
                                 trajectory: Dict[str, Any],
                                 generated_text: Optional[str] = None) -> Dict[str, float]:
        """
        Compute reward for entire generation trajectory
        
        Components:
        1. Safety reward: 1 - harm_probability (from ASAN)
        2. Trajectory smoothness: penalize erratic spectral patterns
        3. Early detection bonus: reward staying safe throughout
        4. Quality preservation: don't sacrifice helpfulness for safety
        
        Returns:
            total_reward: scalar reward for this trajectory
            component_rewards: breakdown by reward type
            trajectory_quality_score: overall trajectory health
        """
        with torch.no_grad():
            # Get ASAN prediction for full trajectory
            asan_output = self.asan_predictor(
                trajectory['attention_patterns'],
                trajectory['hidden_states'],
                trajectory['token_probs']
            )
            
            harm_prob = asan_output['harm_probability'].item()
            confidence = asan_output['confidence'].item()
        
        # 1. Safety reward (primary component)
        safety_reward = self._compute_safety_reward(harm_prob, confidence)
        
        # 2. Trajectory smoothness
        smoothness_reward = self._compute_trajectory_smoothness(trajectory, asan_output)
        
        # 3. Early detection bonus
        early_detection_bonus = self._compute_early_detection_bonus(trajectory, asan_output)
        
        # 4. Quality preservation (if text available)
        quality_reward = 0.0
        if generated_text is not None:
            quality_reward = self._compute_quality_reward(trajectory, generated_text, asan_output)
        
        # Combine rewards
        total_reward = (
            safety_reward * self.config.safety_weight +
            smoothness_reward * self.config.trajectory_smoothness_weight +
            early_detection_bonus * self.config.early_detection_bonus_weight +
            quality_reward * self.config.quality_preservation_weight
        )
        
        # Normalize to [min_reward, max_reward]
        total_reward = np.clip(total_reward, self.config.min_reward, self.config.max_reward)
        
        component_rewards = {
            'safety': safety_reward,
            'smoothness': smoothness_reward,
            'early_detection': early_detection_bonus,
            'quality': quality_reward
        }
        
        trajectory_quality_score = self._compute_trajectory_quality_score(
            trajectory, asan_output
        )
        
        return {
            'total_reward': total_reward,
            'component_rewards': component_rewards,
            'trajectory_quality_score': trajectory_quality_score,
            'harm_probability': harm_prob
        }
    
    def _compute_safety_reward(self, harm_prob: float, confidence: float) -> float:
        """Compute safety reward from harm probability"""
        # Base reward: inverse of harm probability
        base_reward = 1.0 - harm_prob
        
        # Scale by confidence
        reward = base_reward * (0.5 + confidence * 0.5)
        
        # Apply reward shaping
        if self.config.reward_shaping == "exponential":
            reward = np.exp(reward) - 1.0
        elif self.config.reward_shaping == "tanh":
            reward = np.tanh(reward * 2)
        
        return reward
    
    def _compute_trajectory_smoothness(self, 
                                     trajectory: Dict[str, Any],
                                     asan_output: Dict[str, torch.Tensor]) -> float:
        """Compute reward for trajectory smoothness"""
        # Get predictions at each timestep
        temporal_predictions = self.asan_predictor.predict_at_each_timestep(
            trajectory['attention_patterns'],
            trajectory['hidden_states'],
            trajectory['token_probs']
        )
        
        if len(temporal_predictions) < 2:
            return 0.0
        
        # Compute variance in harm probabilities over time
        harm_probs = [pred['harm_probability'].item() for pred in temporal_predictions]
        variance = np.var(harm_probs)
        
        # Smooth trajectories have low variance
        smoothness_reward = 1.0 - min(1.0, variance * 5)  # Normalize variance
        
        return smoothness_reward
    
    def _compute_early_detection_bonus(self,
                                     trajectory: Dict[str, Any],
                                     asan_output: Dict[str, torch.Tensor]) -> float:
        """Reward for staying safe throughout generation"""
        temporal_predictions = self.asan_predictor.predict_at_each_timestep(
            trajectory['attention_patterns'],
            trajectory['hidden_states'],
            trajectory['token_probs']
        )
        
        if len(temporal_predictions) == 0:
            return 0.0
        
        # Check if harm probability stayed low throughout
        harm_probs = [pred['harm_probability'].item() for pred in temporal_predictions]
        max_harm_prob = max(harm_probs)
        
        # Bonus if stayed safe throughout
        if max_harm_prob < 0.3:
            bonus = 0.5 * (1.0 - max_harm_prob / 0.3)
        else:
            bonus = 0.0
        
        return bonus
    
    def _compute_quality_reward(self,
                               trajectory: Dict[str, Any],
                               generated_text: str,
                               asan_output: Dict[str, torch.Tensor]) -> float:
        """
        Reward for quality preservation
        
        In practice, would use language model or human evaluator
        For now, use heuristics based on trajectory properties
        """
        # Heuristic: reward diverse token probabilities (not too concentrated)
        if not trajectory['token_probs']:
            return 0.0
        
        # Compute average entropy of token distributions
        entropies = []
        for token_prob in trajectory['token_probs']:
            if token_prob.dim() > 0:
                prob_dist = F.softmax(token_prob, dim=-1)
                entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10))
                entropies.append(entropy.item())
        
        if entropies:
            avg_entropy = np.mean(entropies)
            # Normalize entropy (typical range for token distributions)
            normalized_entropy = avg_entropy / 10.0  # Approximate max entropy
            quality_reward = min(1.0, normalized_entropy)
        else:
            quality_reward = 0.0
        
        return quality_reward
    
    def _compute_trajectory_quality_score(self,
                                        trajectory: Dict[str, Any],
                                        asan_output: Dict[str, torch.Tensor]) -> float:
        """Compute overall trajectory quality score"""
        # Combine multiple factors
        harm_prob = asan_output['harm_probability'].item()
        confidence = asan_output['confidence'].item()
        
        # Quality score: high when low harm and high confidence
        quality_score = (1.0 - harm_prob) * confidence
        
        return quality_score
    
    def compute_token_level_rewards(self, trajectory: Dict[str, Any]) -> List[float]:
        """
        Dense reward signal at each token
        
        Enable credit assignment:
        - Which tokens led to harmful patterns?
        - Which tokens corrected course?
        - Which tokens maintained safety?
        
        Returns:
            token_rewards: [num_tokens] reward per token
        """
        temporal_predictions = self.asan_predictor.predict_at_each_timestep(
            trajectory['attention_patterns'],
            trajectory['hidden_states'],
            trajectory['token_probs']
        )
        
        token_rewards = []
        
        for i, pred in enumerate(temporal_predictions):
            harm_prob = pred['harm_probability'].item()
            confidence = pred['confidence'].item()
            
            # Token-level safety reward
            token_safety_reward = (1.0 - harm_prob) * confidence
            
            # Temporal discounting: weight recent tokens more
            temporal_weight = self.config.temporal_discounting ** (len(temporal_predictions) - i - 1)
            
            # Token reward
            token_reward = token_safety_reward * temporal_weight
            
            token_rewards.append(token_reward)
        
        return token_rewards
    
    def shaped_reward(self, 
                     asan_prediction: Dict[str, torch.Tensor],
                     action_taken: Optional[str] = None) -> float:
        """
        Shape reward based on ASAN prediction and model action
        
        Reward structure:
        - High reward: Low harm_prob, stayed safe
        - Medium reward: High harm_prob detected, but self-corrected
        - Low reward: High harm_prob, continued on harmful path
        - Penalty: Safe trajectory, but made it unsafe
        """
        harm_prob = asan_prediction['harm_probability'].item()
        confidence = asan_prediction['confidence'].item()
        
        # Base reward
        base_reward = (1.0 - harm_prob) * confidence
        
        # Shape based on action (if available)
        if action_taken is not None:
            # In practice, would analyze if action corrected course
            # For now, just return base reward
            pass
        
        return base_reward
