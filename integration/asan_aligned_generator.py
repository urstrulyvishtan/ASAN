"""
Unified ASAN-Aligned Generator

Single unified interface that combines ASAN prediction, steering, and RL-trained policy.
"""

import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..models.asan_predictor import ASANPredictor
from ..alignment.spectral_steering import SpectralSteeringController, SteeringConfig
from ..alignment.steering_strategies import SteeringStrategyType, create_steering_strategy, StrategyConfig


@dataclass
class GeneratorConfig:
    """Configuration for ASAN-aligned generator"""
    enable_steering: bool = True
    enable_rl_policy: bool = True
    steering_strategy: SteeringStrategyType = SteeringStrategyType.ADAPTIVE
    harm_threshold: float = 0.5
    max_length: int = 50
    max_turns: int = 10


class ASANAlignedGenerator:
    """
    Complete system: ASAN prediction + steering + RL-optimized policy
    
    Three-layer safety:
    1. Policy pre-trained with ASAN-RL (long-term safe behavior)
    2. Real-time ASAN monitoring (detect emerging issues)
    3. Inference-time steering (correct course if needed)
    """
    
    def __init__(self, 
                 rl_trained_policy: Optional[Any],
                 asan_predictor: ASANPredictor,
                 steering_controller: Optional[SpectralSteeringController],
                 config: GeneratorConfig):
        """
        Initialize complete aligned system
        
        Args:
            rl_trained_policy: Policy model trained with ASAN-RL (optional)
            asan_predictor: Trained ASAN model
            steering_controller: Steering controller (optional if steering disabled)
            config: Generator configuration
        """
        self.rl_trained_policy = rl_trained_policy
        self.asan_predictor = asan_predictor
        self.steering_controller = steering_controller
        self.config = config
        
        # Set up steering strategy
        if self.config.enable_steering and self.steering_controller is not None:
            steering_config = StrategyConfig()
            self.steering_strategy = create_steering_strategy(
                self.config.steering_strategy,
                self.steering_controller,
                steering_config
            )
        else:
            self.steering_strategy = None
        
        self.asan_predictor.eval()
        if self.rl_trained_policy is not None:
            self.rl_trained_policy.eval()
    
    def generate(self, 
                 prompt: str,
                 max_length: Optional[int] = None,
                 conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate with full ASAN alignment system
        
        Generation process:
        1. Use RL-trained policy (already optimized for safety)
        2. Monitor with ASAN during generation
        3. Apply steering if harmful patterns detected
        4. Log everything for continual learning
        
        Returns:
            generated_text: final output
            safety_metrics: detailed safety analysis
            steering_log: when/how steering was applied
            confidence: system confidence in output safety
        """
        max_length = max_length or self.config.max_length
        
        # Track trajectory
        trajectory = {
            'attention_patterns': {},
            'hidden_states': {},
            'token_probs': []
        }
        steering_log = []
        safety_metrics = []
        
        # Generate with RL policy (simplified - would use actual generation)
        if self.config.enable_rl_policy and self.rl_trained_policy is not None:
            # Use RL-trained policy for generation
            generated_text, trajectory = self._generate_with_rl_policy(
                prompt, max_length, trajectory
            )
        else:
            # Fallback: standard generation (would use base model)
            generated_text, trajectory = self._generate_standard(
                prompt, max_length, trajectory
            )
        
        # Monitor with ASAN
        with torch.no_grad():
            asan_output = self.asan_predictor(
                trajectory['attention_patterns'],
                trajectory['hidden_states'],
                trajectory['token_probs']
            )
            
            harm_prob = asan_output['harm_probability'].item()
            confidence = asan_output['confidence'].item()
        
        safety_metrics.append({
            'harm_probability': harm_prob,
            'confidence': confidence,
            'final_prediction': asan_output
        })
        
        # Apply steering if needed
        if (self.config.enable_steering and 
            self.steering_strategy is not None and
            harm_prob > self.config.harm_threshold):
            
            # Check if steering is needed
            if self.steering_strategy.should_steer(trajectory, asan_output):
                # Apply steering
                steering_params = self.steering_strategy.compute_steering_parameters(
                    trajectory, asan_output
                )
                
                # Log steering action
                steering_log.append({
                    'harm_probability': harm_prob,
                    'steering_params': steering_params,
                    'before_steering': generated_text
                })
                
                # Apply steering (simplified - would actually modify generation)
                # For now, just log the steering decision
        
        return {
            'generated_text': generated_text,
            'safety_metrics': safety_metrics,
            'steering_log': steering_log,
            'confidence': confidence,
            'harm_probability': harm_prob,
            'trajectory': trajectory
        }
    
    def multi_turn_generate(self, 
                           conversation_history: List[Dict[str, Any]],
                           max_turns: Optional[int] = None) -> Dict[str, Any]:
        """
        Handle multi-turn conversation with full safety stack
        
        Across turns:
        - Maintain conversation-level safety tracking
        - Detect cross-turn exploitation attempts
        - Adjust policy based on conversational dynamics
        
        Returns:
            conversation: full multi-turn dialogue
            safety_analysis: turn-by-turn and conversation-level safety
        """
        max_turns = max_turns or self.config.max_turns
        
        conversation = conversation_history.copy() if conversation_history else []
        turn_safety_metrics = []
        
        for turn_idx in range(max_turns):
            # Get current prompt (would come from user)
            if turn_idx < len(conversation):
                current_prompt = conversation[turn_idx].get('user_prompt', '')
            else:
                # End of conversation
                break
            
            # Generate response
            response = self.generate(
                current_prompt,
                conversation_history=conversation
            )
            
            # Store in conversation
            conversation.append({
                'turn': turn_idx,
                'user_prompt': current_prompt,
                'model_response': response['generated_text'],
                'safety_metrics': response['safety_metrics'],
                'harm_probability': response['harm_probability']
            })
            
            turn_safety_metrics.append({
                'turn': turn_idx,
                'harm_probability': response['harm_probability'],
                'confidence': response['confidence'],
                'steering_applied': len(response['steering_log']) > 0
            })
            
            # Check for cross-turn exploitation
            if turn_idx >= 2:
                cross_turn_analysis = self._detect_cross_turn_exploitation(
                    conversation
                )
                
                if cross_turn_analysis.get('exploitation_detected', False):
                    # Log and potentially adjust
                    conversation[-1]['cross_turn_risk'] = cross_turn_analysis
        
        # Overall conversation safety analysis
        overall_safety = self._compute_conversation_safety(conversation)
        
        return {
            'conversation': conversation,
            'turn_safety_metrics': turn_safety_metrics,
            'overall_safety': overall_safety,
            'total_turns': len(conversation)
        }
    
    def _generate_with_rl_policy(self,
                                 prompt: str,
                                 max_length: int,
                                 trajectory: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate with RL-trained policy (simplified placeholder)"""
        # In practice, would use RL-trained policy for generation
        # For now, return placeholder
        generated_text = f"RL-generated response to: {prompt}"
        
        # Simulate trajectory (would come from actual generation)
        trajectory['token_probs'] = [torch.randn(50257) for _ in range(min(10, max_length))]
        
        return generated_text, trajectory
    
    def _generate_standard(self,
                          prompt: str,
                          max_length: int,
                          trajectory: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Standard generation (simplified placeholder)"""
        # In practice, would use base model for generation
        generated_text = f"Standard response to: {prompt}"
        
        # Simulate trajectory
        trajectory['token_probs'] = [torch.randn(50257) for _ in range(min(10, max_length))]
        
        return generated_text, trajectory
    
    def _detect_cross_turn_exploitation(self,
                                       conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect cross-turn exploitation patterns"""
        if len(conversation) < 3:
            return {'exploitation_detected': False}
        
        # Analyze harm probabilities across turns
        harm_probs = [turn.get('harm_probability', 0.0) for turn in conversation]
        
        # Check for gradual increase
        early_harm = sum(harm_probs[:2]) / len(harm_probs[:2])
        late_harm = max(harm_probs[2:]) if len(harm_probs) > 2 else 0.0
        
        if late_harm > early_harm + 0.3:
            return {
                'exploitation_detected': True,
                'pattern': 'gradual_increase',
                'early_harm': early_harm,
                'late_harm': late_harm
            }
        
        return {'exploitation_detected': False}
    
    def _compute_conversation_safety(self,
                                    conversation: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall conversation safety metrics"""
        if not conversation:
            return {
                'overall_harm_prob': 0.0,
                'max_harm_prob': 0.0,
                'avg_confidence': 0.0,
                'safety_score': 0.0
            }
        
        harm_probs = [turn.get('harm_probability', 0.0) for turn in conversation]
        confidences = []
        for turn in conversation:
            metrics = turn.get('safety_metrics', [])
            if metrics:
                confidences.append(metrics[0].get('confidence', 0.5))
        
        overall_harm_prob = sum(harm_probs) / len(harm_probs) if harm_probs else 0.0
        max_harm_prob = max(harm_probs) if harm_probs else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Safety score: high when low harm and high confidence
        safety_score = (1.0 - overall_harm_prob) * avg_confidence
        
        return {
            'overall_harm_prob': overall_harm_prob,
            'max_harm_prob': max_harm_prob,
            'avg_confidence': avg_confidence,
            'safety_score': safety_score
        }
