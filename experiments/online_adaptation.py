"""
Online Adaptation Experiment

Test continual learning from user interactions.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import json
from pathlib import Path

from ..models.asan_predictor import ASANPredictor, ASANConfig
from ..alignment.spectral_steering import SpectralSteeringController, SteeringConfig
from ..integration.online_learning import OnlineLearningSystem, OnlineLearningConfig


@dataclass
class OnlineAdaptationConfig:
    """Configuration for online adaptation experiment"""
    num_interactions: int = 1000
    feedback_rate: float = 0.3  # % of interactions with feedback
    adaptation_frequency: int = 100
    save_results: bool = True
    results_path: str = "experiments/results/online_adaptation.json"


def experiment_online_adaptation(
    asan_predictor: ASANPredictor,
    interaction_stream: List[Dict[str, Any]],
    config: OnlineAdaptationConfig = None
) -> Dict[str, Any]:
    """
    Test continual learning from user interactions
    
    Process:
    1. System processes interactions
    2. Receives feedback (labels, quality ratings)
    3. Adapts ASAN and steering based on feedback
    4. Monitors for distribution shift
    5. Evaluates adaptation effectiveness
    
    Metrics:
    - Prediction accuracy over time
    - Adaptation effectiveness
    - Distribution shift detection
    - Quality preservation during adaptation
    """
    config = config or OnlineAdaptationConfig()
    
    # Initialize online learning system
    steering_controller = SpectralSteeringController(asan_predictor, SteeringConfig())
    online_config = OnlineLearningConfig(
        update_frequency=config.adaptation_frequency
    )
    online_learning = OnlineLearningSystem(
        asan_predictor,
        steering_controller,
        None,  # Policy (optional)
        online_config
    )
    
    results = {
        'interaction_history': [],
        'adaptation_points': [],
        'performance_over_time': [],
        'distribution_shifts': []
    }
    
    # Process interactions
    for interaction_idx, interaction in enumerate(interaction_stream[:config.num_interactions]):
        # Process interaction
        online_learning.process_interaction(interaction)
        
        # Store interaction result
        interaction_result = {
            'interaction_id': interaction_idx,
            'harm_probability': interaction.get('asan_prediction', {}).get('harm_probability', 0.0),
            'feedback_received': interaction.get('actual_label') is not None,
            'steering_applied': interaction.get('steering_applied', False)
        }
        results['interaction_history'].append(interaction_result)
        
        # Get statistics periodically
        if interaction_idx % config.adaptation_frequency == 0:
            stats = online_learning.get_statistics()
            results['adaptation_points'].append({
                'interaction_id': interaction_idx,
                'statistics': stats
            })
            
            # Check for distribution shift
            recent_interactions = list(online_learning.interaction_memory)[-100:]
            shift_result = online_learning.detect_distribution_shift(recent_interactions)
            
            if shift_result.get('shift_detected', False):
                results['distribution_shifts'].append({
                    'interaction_id': interaction_idx,
                    'shift_magnitude': shift_result['shift_magnitude']
                })
        
        # Evaluate performance periodically
        if interaction_idx % 50 == 0 and interaction_idx > 0:
            performance = _evaluate_online_performance(
                online_learning,
                interaction_stream[:interaction_idx]
            )
            performance['interaction_id'] = interaction_idx
            results['performance_over_time'].append(performance)
    
    # Final evaluation
    final_performance = _evaluate_online_performance(
        online_learning,
        interaction_stream[:config.num_interactions]
    )
    results['final_performance'] = final_performance
    
    # Save results
    if config.save_results:
        results_path = Path(config.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def _evaluate_online_performance(online_learning: OnlineLearningSystem,
                                interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate online learning performance"""
    stats = online_learning.get_statistics()
    
    # Compute accuracy on labeled interactions
    labeled_interactions = [i for i in interactions if i.get('actual_label') is not None]
    
    if labeled_interactions:
        correct = 0
        for interaction in labeled_interactions:
            predicted_harm = interaction.get('asan_prediction', {}).get('harm_probability', 0.0) > 0.5
            actual_harm = interaction['actual_label'] == 'harmful'
            if predicted_harm == actual_harm:
                correct += 1
        
        accuracy = correct / len(labeled_interactions)
    else:
        accuracy = 0.0
    
    return {
        'accuracy': accuracy,
        'avg_prediction_error': stats.get('avg_prediction_error', 0.0),
        'false_positives': stats.get('false_positives', 0),
        'false_negatives': stats.get('false_negatives', 0)
    }


if __name__ == "__main__":
    # Example usage
    config = ASANConfig()
    asan_predictor = ASANPredictor(config)
    
    # Simulate interaction stream
    interaction_stream = [
        {
            'prompt': f'Test prompt {i}',
            'asan_prediction': {'harm_probability': np.random.random()},
            'actual_label': 'harmful' if np.random.random() > 0.5 else 'safe',
            'steering_applied': False
        }
        for i in range(100)
    ]
    
    results = experiment_online_adaptation(asan_predictor, interaction_stream)
    print(json.dumps(results['final_performance'], indent=2))
