"""
Long-Horizon Multi-Turn Rewards

Compute rewards that account for multi-turn conversational dynamics,
capturing delayed harm that might not be visible in single responses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

try:
    from models.asan_predictor import ASANPredictor
except ImportError:
    from ..models.asan_predictor import ASANPredictor


@dataclass
class LongHorizonConfig:
    """Configuration for long-horizon reward computation"""
    horizon_length: int = 5  # Number of turns to consider
    temporal_discount: float = 0.9  # Discount factor for earlier turns
    cross_turn_analysis: bool = True  # Analyze cross-turn patterns
    detection_bonus: float = 0.5  # Bonus for early detection of multi-turn attacks


class LongHorizonRewardComputer:
    """
    Compute rewards that account for multi-turn conversational dynamics
    
    Critical insight:
    - Some harms only emerge over multiple turns
    - Model might gather context in turn 1, exploit in turn 3
    - Need to credit/penalize based on trajectory across conversation
    """
    
    def __init__(self, asan_predictor: ASANPredictor, config: LongHorizonConfig):
        """
        Args:
            asan_predictor: Trained ASAN model
            horizon_length: how many turns to consider for reward
        """
        self.asan_predictor = asan_predictor
        self.config = config
        self.asan_predictor.eval()
    
    def compute_conversation_reward(self, 
                                   conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate safety of entire conversation trajectory
        
        Analysis:
        1. Run ASAN on each turn independently
        2. Analyze cross-turn patterns:
           - Is model gradually leading toward harmful output?
           - Is context from turn 1 being exploited in turn 4?
           - Are safety mechanisms degrading over turns?
        3. Compute conversation-level reward
        
        Returns:
            conversation_reward: overall safety score
            turn_level_rewards: reward per turn
            cross_turn_patterns: identified multi-turn risks
        """
        if not conversation_history:
            return {
                'conversation_reward': 0.0,
                'turn_level_rewards': [],
                'cross_turn_patterns': {}
            }
        
        # 1. Run ASAN on each turn
        turn_level_predictions = []
        turn_level_rewards = []
        
        for turn_idx, turn_data in enumerate(conversation_history):
            with torch.no_grad():
                asan_output = self.asan_predictor(
                    turn_data['attention_patterns'],
                    turn_data['hidden_states'],
                    turn_data['token_probs']
                )
                
                harm_prob = asan_output['harm_probability'].item()
                confidence = asan_output['confidence'].item()
                
                # Turn-level reward (higher for safer turns)
                turn_reward = (1.0 - harm_prob) * confidence
                
                # Apply temporal discount (earlier turns weighted less)
                temporal_weight = self.config.temporal_discount ** (len(conversation_history) - turn_idx - 1)
                turn_reward *= temporal_weight
                
                turn_level_predictions.append({
                    'turn_idx': turn_idx,
                    'harm_probability': harm_prob,
                    'confidence': confidence,
                    'reward': turn_reward,
                    'prediction': asan_output
                })
                turn_level_rewards.append(turn_reward)
        
        # 2. Analyze cross-turn patterns
        cross_turn_patterns = {}
        if self.config.cross_turn_analysis and len(conversation_history) >= 2:
            cross_turn_patterns = self._analyze_cross_turn_patterns(
                conversation_history,
                turn_level_predictions
            )
        
        # 3. Compute conversation-level reward
        conversation_reward = self._compute_conversation_level_reward(
            turn_level_rewards,
            turn_level_predictions,
            cross_turn_patterns
        )
        
        return {
            'conversation_reward': conversation_reward,
            'turn_level_rewards': turn_level_rewards,
            'turn_level_predictions': turn_level_predictions,
            'cross_turn_patterns': cross_turn_patterns
        }
    
    def _analyze_cross_turn_patterns(self,
                                    conversation_history: List[Dict[str, Any]],
                                    turn_level_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across conversation turns"""
        patterns = {
            'gradual_harm_increase': False,
            'context_exploitation': False,
            'safety_degradation': False,
            'multi_turn_attack_detected': False
        }
        
        if len(turn_level_predictions) < 2:
            return patterns
        
        # Check for gradual harm increase
        harm_probs = [pred['harm_probability'] for pred in turn_level_predictions]
        if len(harm_probs) >= 3:
            # Check if harm probability increases over turns
            early_harm = np.mean(harm_probs[:2])
            late_harm = np.mean(harm_probs[-2:])
            
            if late_harm > early_harm + 0.2:  # Significant increase
                patterns['gradual_harm_increase'] = True
                patterns['harm_increase_magnitude'] = late_harm - early_harm
        
        # Check for context exploitation
        # (In practice, would analyze how context from early turns is used in later turns)
        if len(conversation_history) >= 3:
            # Heuristic: if later turns have high harm but early turns were safe
            early_avg_harm = np.mean([pred['harm_probability'] 
                                     for pred in turn_level_predictions[:2]])
            late_max_harm = max([pred['harm_probability'] 
                                for pred in turn_level_predictions[2:]])
            
            if early_avg_harm < 0.3 and late_max_harm > 0.7:
                patterns['context_exploitation'] = True
        
        # Check for safety degradation
        confidences = [pred['confidence'] for pred in turn_level_predictions]
        if len(confidences) >= 2:
            # Check if confidence decreases over time
            early_conf = np.mean(confidences[:len(confidences)//2])
            late_conf = np.mean(confidences[len(confidences)//2:])
            
            if late_conf < early_conf - 0.2:  # Significant decrease
                patterns['safety_degradation'] = True
        
        # Multi-turn attack detected
        if (patterns['gradual_harm_increase'] or 
            patterns['context_exploitation'] or 
            patterns['safety_degradation']):
            patterns['multi_turn_attack_detected'] = True
        
        return patterns
    
    def _compute_conversation_level_reward(self,
                                         turn_level_rewards: List[float],
                                         turn_level_predictions: List[Dict[str, Any]],
                                         cross_turn_patterns: Dict[str, Any]) -> float:
        """Compute overall conversation reward"""
        # Base reward: average of turn-level rewards
        base_reward = np.mean(turn_level_rewards) if turn_level_rewards else 0.0
        
        # Penalties for cross-turn patterns
        penalty = 0.0
        
        if cross_turn_patterns.get('gradual_harm_increase', False):
            harm_increase = cross_turn_patterns.get('harm_increase_magnitude', 0.0)
            penalty += harm_increase * 0.5
        
        if cross_turn_patterns.get('context_exploitation', False):
            penalty += 0.3
        
        if cross_turn_patterns.get('safety_degradation', False):
            penalty += 0.2
        
        # Bonus for early detection
        if cross_turn_patterns.get('multi_turn_attack_detected', False):
            # Bonus if attack was detected early (before it fully materialized)
            early_harm = np.mean([pred['harm_probability'] 
                                 for pred in turn_level_predictions[:len(turn_level_predictions)//2]])
            
            if early_harm < 0.5:  # Detected early
                bonus = self.config.detection_bonus
            else:
                bonus = 0.0
        else:
            bonus = 0.0
        
        # Final reward
        conversation_reward = base_reward - penalty + bonus
        
        # Clip to reasonable range
        conversation_reward = np.clip(conversation_reward, -1.0, 1.0)
        
        return conversation_reward
    
    def detect_multi_turn_exploitation(self, 
                                      conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patterns where model is being manipulated across turns
        
        Examples:
        - Turn 1: Establish trust with safe response
        - Turn 2-3: Gradually shift context
        - Turn 4: Exploit accumulated context for harmful output
        
        Use ASAN to track:
        - Spectral signatures evolving across turns
        - Attention patterns shifting toward manipulation
        - Hidden state trajectories trending toward unsafe regions
        """
        exploitation_patterns = {
            'trust_building': False,
            'gradual_context_shift': False,
            'exploitation_attempt': False,
            'confidence': 0.0
        }
        
        if len(conversation_history) < 3:
            return exploitation_patterns
        
        # Analyze spectral signatures across turns
        spectral_signatures = []
        for turn_data in conversation_history:
            with torch.no_grad():
                signature = self.asan_predictor.get_spectral_signature({
                    'attention_patterns': turn_data['attention_patterns'],
                    'hidden_states': turn_data['hidden_states'],
                    'token_probs': turn_data['token_probs']
                })
                spectral_signatures.append(signature.cpu().numpy())
        
        # Check for gradual shift in spectral signature
        if len(spectral_signatures) >= 3:
            early_signature = np.mean(spectral_signatures[:2], axis=0)
            late_signature = np.mean(spectral_signatures[-2:], axis=0)
            
            # Compute cosine similarity
            early_norm = np.linalg.norm(early_signature)
            late_norm = np.linalg.norm(late_signature)
            
            if early_norm > 0 and late_norm > 0:
                cosine_sim = np.dot(early_signature, late_signature) / (early_norm * late_norm)
                
                # Low similarity indicates shift
                if cosine_sim < 0.7:
                    exploitation_patterns['gradual_context_shift'] = True
                    exploitation_patterns['shift_magnitude'] = 1.0 - cosine_sim
        
        # Check for trust-building then exploitation
        turn_harm_probs = []
        for turn_data in conversation_history:
            with torch.no_grad():
                asan_output = self.asan_predictor(
                    turn_data['attention_patterns'],
                    turn_data['hidden_states'],
                    turn_data['token_probs']
                )
                turn_harm_probs.append(asan_output['harm_probability'].item())
        
        if len(turn_harm_probs) >= 3:
            early_harm = np.mean(turn_harm_probs[:2])
            late_harm = max(turn_harm_probs[2:])
            
            if early_harm < 0.3 and late_harm > 0.7:
                exploitation_patterns['trust_building'] = True
                exploitation_patterns['exploitation_attempt'] = True
                exploitation_patterns['confidence'] = late_harm - early_harm
        
        return exploitation_patterns
    
    def temporal_credit_assignment(self, 
                                  conversation_history: List[Dict[str, Any]],
                                  harm_occurred_turn: int) -> Dict[int, float]:
        """
        When harm occurs at turn N, assign credit/blame to earlier turns
        
        Questions:
        - Which earlier turn set up the harmful trajectory?
        - Which turns had opportunities to correct course?
        - Which turns maintained safety?
        
        Use ASAN's spectral analysis to trace back through conversation
        
        Returns:
            turn_credits: Dict[turn_idx, credit_score]
                Positive credit = contributed to safety
                Negative credit = contributed to harm
        """
        turn_credits = {}
        
        if harm_occurred_turn < 0 or harm_occurred_turn >= len(conversation_history):
            return turn_credits
        
        # Analyze each turn's contribution
        for turn_idx in range(harm_occurred_turn + 1):
            turn_data = conversation_history[turn_idx]
            
            with torch.no_grad():
                # Get ASAN prediction for this turn
                asan_output = self.asan_predictor(
                    turn_data['attention_patterns'],
                    turn_data['hidden_states'],
                    turn_data['token_probs']
                )
                
                harm_prob = asan_output['harm_probability'].item()
                
                # Compute contribution to final harm
                # Turns with high harm_prob contribute negatively
                # Turns with low harm_prob contribute positively
                
                # Temporal discount: turns closer to harm are more important
                temporal_weight = self.config.temporal_discount ** (harm_occurred_turn - turn_idx)
                
                # Credit: negative if high harm, positive if low harm
                credit = (1.0 - harm_prob - 0.5) * 2.0  # Scale to [-1, 1]
                credit *= temporal_weight
                
                turn_credits[turn_idx] = credit
        
        return turn_credits
