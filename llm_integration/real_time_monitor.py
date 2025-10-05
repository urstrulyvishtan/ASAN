"""
Real-Time ASAN Monitoring System

Monitor LLM inference in real-time and intervene when harmful patterns detected.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import json
from dataclasses import dataclass
from collections import deque
import numpy as np

from ..models.asan_predictor import ASANPredictor, ASANConfig
from ..data.llm_trajectory_collector import LLMTrajectoryCollector


@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring"""
    intervention_threshold: float = 0.7
    prediction_frequency: str = 'every_token'  # 'every_token', 'every_5_tokens', 'adaptive'
    max_trajectory_length: int = 100
    confidence_threshold: float = 0.8
    intervention_strategy: str = 'stop_generation'  # 'stop', 'backtrack', 'filter', 'human_review'
    backtrack_tokens: int = 5
    save_interventions: bool = True
    intervention_log_path: str = 'logs/interventions.json'


class RealTimeASANMonitor:
    """
    Monitor LLM inference in real-time and intervene when harmful patterns detected
    
    Integration with existing LLM:
    1. Wrap model with ASAN monitoring hooks
    2. Collect trajectories during generation  
    3. Run ASAN predictor at each token
    4. Trigger intervention if harm probability exceeds threshold
    """
    
    def __init__(self, 
                 llm_model,
                 asan_predictor: ASANPredictor,
                 config: MonitoringConfig):
        """
        Args:
            llm_model: The language model to monitor
            asan_predictor: Trained ASAN model
            config: Monitoring configuration
        """
        self.llm = llm_model
        self.asan = asan_predictor
        self.config = config
        
        # Initialize trajectory collector
        self.trajectory_collector = LLMTrajectoryCollector(llm_model)
        
        # Monitoring state
        self.is_monitoring = False
        self.current_generation_id = None
        self.intervention_log = []
        self.false_positive_log = []
        
        # Prediction cache for efficiency
        self.prediction_cache = deque(maxlen=10)
        self.last_prediction_time = 0
        
        # Intervention statistics
        self.stats = {
            'total_generations': 0,
            'interventions_triggered': 0,
            'false_positives': 0,
            'true_positives': 0,
            'average_prediction_time': 0.0,
            'average_intervention_time': 0.0
        }
        
    def generate_with_monitoring(self, input_ids: torch.Tensor, max_length: int = 50,
                               temperature: float = 1.0, do_sample: bool = True,
                               pad_token_id: Optional[int] = None) -> Tuple[str, bool, Dict]:
        """
        Generate text while monitoring for harmful patterns
        
        Process:
        1. Generate token
        2. Collect internal states
        3. Run ASAN prediction
        4. If harm_probability > threshold:
           - Log intervention
           - Either: stop generation, regenerate, or apply safety filter
        5. Continue until max_length or intervention
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            
        Returns:
            generated_text: the generated text (possibly truncated)
            intervention_occurred: bool
            intervention_details: dict with:
                - timestep of intervention
                - harm probability at intervention
                - predicted harm category
                - full trajectory
        """
        self.is_monitoring = True
        self.current_generation_id = f"gen_{int(time.time())}"
        
        generated_tokens = []
        intervention_occurred = False
        intervention_details = {}
        
        # Start generation
        current_ids = input_ids.clone()
        start_time = time.time()
        
        try:
            for step in range(max_length):
                step_start = time.time()
                
                # Generate next token
                with torch.no_grad():
                    outputs = self.llm(current_ids)
                    logits = outputs.logits
                    
                    # Sample next token
                    if do_sample:
                        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
                # Check for end token
                if pad_token_id is not None and next_token.item() == pad_token_id:
                    break
                    
                generated_tokens.append(next_token.item())
                
                # Collect trajectory up to this point
                trajectory = self.trajectory_collector.collect_during_generation(
                    current_ids, max_length=1, temperature=temperature, do_sample=do_sample
                )
                
                # Run ASAN prediction
                should_predict = self._should_run_prediction(step, trajectory)
                
                if should_predict:
                    prediction_start = time.time()
                    
                    prediction = self.asan(
                        trajectory['attention_patterns'],
                        trajectory['hidden_states'],
                        trajectory['token_probs'],
                        current_timestep=step
                    )
                    
                    prediction_time = time.time() - prediction_start
                    self.stats['average_prediction_time'] = (
                        self.stats['average_prediction_time'] * 0.9 + prediction_time * 0.1
                    )
                    
                    # Check for intervention
                    harm_prob = prediction['harm_probability'].item()
                    confidence = prediction['confidence'].item()
                    
                    if (harm_prob > self.config.intervention_threshold and 
                        confidence > self.config.confidence_threshold):
                        
                        intervention_start = time.time()
                        
                        # Trigger intervention
                        intervention_result = self._trigger_intervention(
                            prediction, trajectory, step, generated_tokens
                        )
                        
                        intervention_time = time.time() - intervention_start
                        self.stats['average_intervention_time'] = (
                            self.stats['average_intervention_time'] * 0.9 + intervention_time * 0.1
                        )
                        
                        if intervention_result['intervention_triggered']:
                            intervention_occurred = True
                            intervention_details = {
                                'timestep': step,
                                'harm_probability': harm_prob,
                                'confidence': confidence,
                                'predicted_category': prediction['harm_category'].argmax().item(),
                                'intervention_type': intervention_result['type'],
                                'generated_tokens': generated_tokens.copy(),
                                'trajectory': trajectory,
                                'prediction': prediction
                            }
                            
                            # Log intervention
                            self._log_intervention(intervention_details)
                            
                            # Apply intervention strategy
                            if self.config.intervention_strategy == 'stop_generation':
                                break
                            elif self.config.intervention_strategy == 'backtrack':
                                # Backtrack and regenerate
                                current_ids = self._backtrack_and_regenerate(
                                    current_ids, self.config.backtrack_tokens
                                )
                                continue
                                
                    # Cache prediction for efficiency
                    self.prediction_cache.append({
                        'timestep': step,
                        'prediction': prediction,
                        'timestamp': time.time()
                    })
                
                # Update current_ids for next iteration
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Limit context length
                if current_ids.size(-1) > 1024:
                    current_ids = current_ids[:, -512:]
                    
        except Exception as e:
            print(f"Error during monitored generation: {e}")
            intervention_occurred = True
            intervention_details = {
                'error': str(e),
                'generated_tokens': generated_tokens
            }
            
        finally:
            self.is_monitoring = False
            self.stats['total_generations'] += 1
            
        # Convert tokens to text
        try:
            generated_text = self.trajectory_collector.tokenizer.decode(generated_tokens)
        except:
            generated_text = f"Generated {len(generated_tokens)} tokens"
            
        return generated_text, intervention_occurred, intervention_details
        
    def _should_run_prediction(self, timestep: int, trajectory: Dict) -> bool:
        """Decide when to run ASAN prediction"""
        
        if self.config.prediction_frequency == 'every_token':
            return True
        elif self.config.prediction_frequency == 'every_5_tokens':
            return timestep % 5 == 0
        elif self.config.prediction_frequency == 'adaptive':
            return self._adaptive_prediction_schedule(timestep, trajectory)
        else:
            return True
            
    def _adaptive_prediction_schedule(self, timestep: int, trajectory: Dict) -> bool:
        """
        Decide when to run ASAN prediction based on trajectory characteristics
        
        Strategy:
        - Run every token initially
        - If patterns stable, reduce frequency
        - If detecting concerning patterns, increase frequency
        """
        
        # Always predict for first few tokens
        if timestep < 5:
            return True
            
        # Check recent predictions for concerning patterns
        recent_predictions = list(self.prediction_cache)[-3:]
        if recent_predictions:
            avg_harm_prob = np.mean([
                pred['prediction']['harm_probability'].item() 
                for pred in recent_predictions
            ])
            
            # If harm probability is increasing, predict more frequently
            if avg_harm_prob > 0.3:
                return timestep % 2 == 0
            elif avg_harm_prob > 0.1:
                return timestep % 3 == 0
            else:
                return timestep % 5 == 0
                
        return timestep % 5 == 0
        
    def _trigger_intervention(self, prediction: Dict, trajectory: Dict, 
                            timestep: int, generated_tokens: List[int]) -> Dict:
        """Trigger intervention based on prediction"""
        
        intervention_result = {
            'intervention_triggered': True,
            'type': self.config.intervention_strategy,
            'timestep': timestep,
            'harm_probability': prediction['harm_probability'].item(),
            'confidence': prediction['confidence'].item()
        }
        
        # Update statistics
        self.stats['interventions_triggered'] += 1
        
        return intervention_result
        
    def _backtrack_and_regenerate(self, current_ids: torch.Tensor, 
                                 backtrack_tokens: int) -> torch.Tensor:
        """Backtrack N tokens and regenerate"""
        
        if current_ids.size(-1) > backtrack_tokens:
            # Remove last N tokens
            backtracked_ids = current_ids[:, :-backtrack_tokens]
        else:
            # If not enough tokens, go back to beginning
            backtracked_ids = current_ids[:, :1]
            
        return backtracked_ids
        
    def _log_intervention(self, intervention_details: Dict):
        """Log intervention details"""
        
        log_entry = {
            'generation_id': self.current_generation_id,
            'timestamp': time.time(),
            'intervention_details': intervention_details,
            'stats': self.stats.copy()
        }
        
        self.intervention_log.append(log_entry)
        
        # Save to file if configured
        if self.config.save_interventions:
            self._save_intervention_log(log_entry)
            
    def _save_intervention_log(self, log_entry: Dict):
        """Save intervention log to file"""
        import os
        
        log_dir = os.path.dirname(self.config.intervention_log_path)
        os.makedirs(log_dir, exist_ok=True)
        
        # Append to log file
        with open(self.config.intervention_log_path, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
            
    def get_monitoring_stats(self) -> Dict:
        """Get current monitoring statistics"""
        stats = self.stats.copy()
        
        # Add derived metrics
        if stats['total_generations'] > 0:
            stats['intervention_rate'] = stats['interventions_triggered'] / stats['total_generations']
            stats['false_positive_rate'] = stats['false_positives'] / max(stats['interventions_triggered'], 1)
            stats['true_positive_rate'] = stats['true_positives'] / max(stats['interventions_triggered'], 1)
        else:
            stats['intervention_rate'] = 0.0
            stats['false_positive_rate'] = 0.0
            stats['true_positive_rate'] = 0.0
            
        return stats
        
    def update_intervention_feedback(self, generation_id: str, was_harmful: bool, 
                                   was_false_positive: bool):
        """Update intervention statistics based on feedback"""
        
        if was_harmful:
            self.stats['true_positives'] += 1
        elif was_false_positive:
            self.stats['false_positives'] += 1
            
    def reset_stats(self):
        """Reset monitoring statistics"""
        self.stats = {
            'total_generations': 0,
            'interventions_triggered': 0,
            'false_positives': 0,
            'true_positives': 0,
            'average_prediction_time': 0.0,
            'average_intervention_time': 0.0
        }
        
    def cleanup(self):
        """Cleanup resources"""
        self.trajectory_collector.cleanup_hooks()
        self.is_monitoring = False


class BatchASANMonitor:
    """
    Monitor multiple LLM generations in batch for efficiency
    """
    
    def __init__(self, llm_model, asan_predictor: ASANPredictor, 
                 config: MonitoringConfig, batch_size: int = 4):
        super().__init__()
        
        self.llm = llm_model
        self.asan = asan_predictor
        self.config = config
        self.batch_size = batch_size
        
        # Individual monitors for each batch item
        self.monitors = [
            RealTimeASANMonitor(llm_model, asan_predictor, config)
            for _ in range(batch_size)
        ]
        
    def generate_batch_with_monitoring(self, input_ids_batch: List[torch.Tensor],
                                     max_length: int = 50) -> List[Tuple[str, bool, Dict]]:
        """Generate batch of texts with monitoring"""
        
        results = []
        
        for i, input_ids in enumerate(input_ids_batch):
            monitor = self.monitors[i % len(self.monitors)]
            
            result = monitor.generate_with_monitoring(
                input_ids, max_length=max_length
            )
            results.append(result)
            
        return results
        
    def cleanup(self):
        """Cleanup all monitors"""
        for monitor in self.monitors:
            monitor.cleanup()


class AdaptiveThresholdMonitor(RealTimeASANMonitor):
    """
    Monitor with adaptive intervention threshold based on context
    """
    
    def __init__(self, llm_model, asan_predictor: ASANPredictor, 
                 config: MonitoringConfig):
        super().__init__(llm_model, asan_predictor, config)
        
        # Adaptive threshold parameters
        self.base_threshold = config.intervention_threshold
        self.threshold_range = (0.3, 0.9)
        self.adaptation_rate = 0.1
        
        # Context tracking
        self.recent_interventions = deque(maxlen=10)
        self.recent_false_positives = deque(maxlen=10)
        
    def _get_adaptive_threshold(self, context: Dict) -> float:
        """Get adaptive threshold based on context"""
        
        # Adjust threshold based on recent performance
        if len(self.recent_false_positives) > 3:
            recent_fp_rate = sum(self.recent_false_positives) / len(self.recent_false_positives)
            
            if recent_fp_rate > 0.3:  # High false positive rate
                # Increase threshold (be more conservative)
                adaptive_threshold = min(
                    self.base_threshold + 0.1,
                    self.threshold_range[1]
                )
            elif recent_fp_rate < 0.1:  # Low false positive rate
                # Decrease threshold (be more sensitive)
                adaptive_threshold = max(
                    self.base_threshold - 0.1,
                    self.threshold_range[0]
                )
            else:
                adaptive_threshold = self.base_threshold
        else:
            adaptive_threshold = self.base_threshold
            
        return adaptive_threshold
        
    def _trigger_intervention(self, prediction: Dict, trajectory: Dict, 
                            timestep: int, generated_tokens: List[int]) -> Dict:
        """Trigger intervention with adaptive threshold"""
        
        # Get adaptive threshold
        context = {
            'timestep': timestep,
            'trajectory_length': len(generated_tokens),
            'recent_interventions': len(self.recent_interventions)
        }
        
        adaptive_threshold = self._get_adaptive_threshold(context)
        
        # Use adaptive threshold
        harm_prob = prediction['harm_probability'].item()
        confidence = prediction['confidence'].item()
        
        if harm_prob > adaptive_threshold and confidence > self.config.confidence_threshold:
            intervention_result = super()._trigger_intervention(
                prediction, trajectory, timestep, generated_tokens
            )
            
            # Track for adaptation
            self.recent_interventions.append(True)
            
            return intervention_result
        else:
            return {'intervention_triggered': False}
            
    def update_intervention_feedback(self, generation_id: str, was_harmful: bool, 
                                   was_false_positive: bool):
        """Update feedback with adaptive learning"""
        
        super().update_intervention_feedback(generation_id, was_harmful, was_false_positive)
        
        # Track false positives for threshold adaptation
        if was_false_positive:
            self.recent_false_positives.append(True)
        else:
            self.recent_false_positives.append(False)
