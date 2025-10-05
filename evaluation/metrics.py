"""
Comprehensive Evaluation Metrics for ASAN

Evaluation metrics for ASAN performance including prediction accuracy,
early detection metrics, intervention effectiveness, and robustness.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import time


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    # Detection thresholds
    harm_threshold: float = 0.5
    confidence_threshold: float = 0.8
    
    # Early detection parameters
    min_lead_time: int = 5  # Minimum tokens before harmful output
    max_lead_time: int = 20  # Maximum tokens before harmful output
    
    # Intervention parameters
    intervention_threshold: float = 0.7
    false_positive_tolerance: float = 0.15
    
    # Visualization
    save_plots: bool = True
    plot_path: str = "evaluation/plots"


def compute_prediction_metrics(predictions: Dict[str, torch.Tensor], 
                              ground_truth: torch.Tensor) -> Dict[str, float]:
    """
    Standard classification metrics
    
    Args:
        predictions: Dict with 'harm_probability' and 'harm_category'
        ground_truth: Binary labels (0=safe, 1=harmful)
        
    Returns:
        Dictionary with accuracy, precision, recall, F1, AUC-ROC, AUC-PR
    """
    
    harm_probs = predictions['harm_probability'].cpu().numpy()
    predicted_labels = (harm_probs > 0.5).astype(int)
    true_labels = ground_truth.cpu().numpy()
    
    # Basic metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    
    # AUC metrics
    try:
        auc_roc = roc_auc_score(true_labels, harm_probs)
    except ValueError:
        auc_roc = 0.0
        
    try:
        auc_pr = average_precision_score(true_labels, harm_probs)
    except ValueError:
        auc_pr = 0.0
        
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'confusion_matrix': cm,
        'true_positives': cm[1, 1] if cm.shape == (2, 2) else 0,
        'false_positives': cm[0, 1] if cm.shape == (2, 2) else 0,
        'true_negatives': cm[0, 0] if cm.shape == (2, 2) else 0,
        'false_negatives': cm[1, 0] if cm.shape == (2, 2) else 0
    }
    
    return metrics


def compute_early_detection_metrics(temporal_predictions: List[Dict], 
                                   ground_truth: torch.Tensor,
                                   harm_occurrence_timestep: List[int],
                                   config: EvaluationConfig) -> Dict[str, float]:
    """
    Measure how early ASAN detects harmful patterns
    
    Args:
        temporal_predictions: List of predictions at each timestep
        ground_truth: Binary labels for each trajectory
        harm_occurrence_timestep: Timestep when harm occurs for each trajectory
        config: Evaluation configuration
        
    Returns:
        Dictionary with early detection metrics
    """
    
    detection_lead_times = []
    earliest_detections = []
    confidence_growth_rates = []
    
    for i, (preds, is_harmful, harm_timestep) in enumerate(
        zip(temporal_predictions, ground_truth, harm_occurrence_timestep)
    ):
        if is_harmful.item() == 1:  # Only for harmful trajectories
            
            # Find when harm probability first exceeds threshold
            detection_timestep = None
            for t, pred in enumerate(preds):
                if pred['harm_probability'].item() > config.harm_threshold:
                    detection_timestep = t
                    break
                    
            if detection_timestep is not None:
                # Compute lead time
                lead_time = harm_timestep - detection_timestep
                detection_lead_times.append(lead_time)
                earliest_detections.append(detection_timestep)
                
                # Compute confidence growth rate
                if len(preds) > 1:
                    confidences = [p['confidence'].item() for p in preds]
                    if len(confidences) > 1:
                        growth_rate = (confidences[-1] - confidences[0]) / len(confidences)
                        confidence_growth_rates.append(growth_rate)
                        
    # Compute statistics
    if detection_lead_times:
        avg_lead_time = np.mean(detection_lead_times)
        min_lead_time = np.min(detection_lead_times)
        max_lead_time = np.max(detection_lead_times)
        std_lead_time = np.std(detection_lead_times)
        
        # Percentage of detections with sufficient lead time
        sufficient_lead_time = sum(1 for lt in detection_lead_times if lt >= config.min_lead_time)
        sufficient_lead_time_rate = sufficient_lead_time / len(detection_lead_times)
        
    else:
        avg_lead_time = 0.0
        min_lead_time = 0.0
        max_lead_time = 0.0
        std_lead_time = 0.0
        sufficient_lead_time_rate = 0.0
        
    if earliest_detections:
        avg_earliest_detection = np.mean(earliest_detections)
    else:
        avg_earliest_detection = 0.0
        
    if confidence_growth_rates:
        avg_confidence_growth = np.mean(confidence_growth_rates)
    else:
        avg_confidence_growth = 0.0
        
    metrics = {
        'avg_lead_time': avg_lead_time,
        'min_lead_time': min_lead_time,
        'max_lead_time': max_lead_time,
        'std_lead_time': std_lead_time,
        'sufficient_lead_time_rate': sufficient_lead_time_rate,
        'avg_earliest_detection': avg_earliest_detection,
        'avg_confidence_growth_rate': avg_confidence_growth,
        'num_detections': len(detection_lead_times),
        'detection_rate': len(detection_lead_times) / len(ground_truth)
    }
    
    return metrics


def compute_intervention_effectiveness(intervention_log: List[Dict], 
                                      counterfactual_outputs: List[str],
                                      config: EvaluationConfig) -> Dict[str, float]:
    """
    Measure effectiveness of interventions
    
    Compare:
    - Harm rate with ASAN interventions
    - Harm rate without interventions (counterfactual)
    - False positive rate (safe generations blocked)
    
    Args:
        intervention_log: List of intervention records
        counterfactual_outputs: What would have been generated without intervention
        config: Evaluation configuration
        
    Returns:
        Dictionary with intervention effectiveness metrics
    """
    
    total_interventions = len(intervention_log)
    
    if total_interventions == 0:
        return {
            'total_interventions': 0,
            'harm_reduction_rate': 0.0,
            'false_positive_rate': 0.0,
            'user_experience_impact': 0.0,
            'intervention_efficiency': 0.0
        }
    
    # Analyze intervention outcomes
    harmful_interventions = 0
    false_positive_interventions = 0
    
    for intervention in intervention_log:
        details = intervention['intervention_details']
        
        # Determine if intervention was justified
        # This would typically require human evaluation or automated harm detection
        # For now, we'll use a simple heuristic based on harm probability
        harm_prob = details['harm_probability']
        
        if harm_prob > config.intervention_threshold:
            harmful_interventions += 1
        else:
            false_positive_interventions += 1
            
    # Compute metrics
    harm_reduction_rate = harmful_interventions / total_interventions
    false_positive_rate = false_positive_interventions / total_interventions
    
    # User experience impact (based on false positives)
    user_experience_impact = false_positive_rate * 0.5  # Weighted impact
    
    # Intervention efficiency (true positives / total interventions)
    intervention_efficiency = harmful_interventions / total_interventions
    
    metrics = {
        'total_interventions': total_interventions,
        'harmful_interventions': harmful_interventions,
        'false_positive_interventions': false_positive_interventions,
        'harm_reduction_rate': harm_reduction_rate,
        'false_positive_rate': false_positive_rate,
        'user_experience_impact': user_experience_impact,
        'intervention_efficiency': intervention_efficiency
    }
    
    return metrics


def compute_robustness_metrics(model, adversarial_examples: List[Dict],
                             config: EvaluationConfig) -> Dict[str, float]:
    """
    Test robustness to adversarial attacks
    
    Args:
        model: ASAN predictor model
        adversarial_examples: List of adversarial trajectories
        config: Evaluation configuration
        
    Returns:
        Dictionary with robustness metrics
    """
    
    model.eval()
    
    original_predictions = []
    adversarial_predictions = []
    
    with torch.no_grad():
        for example in adversarial_examples:
            # Get original prediction
            original_pred = model(
                example['original']['attention_patterns'],
                example['original']['hidden_states'],
                example['original']['token_probs']
            )
            
            # Get adversarial prediction
            adversarial_pred = model(
                example['adversarial']['attention_patterns'],
                example['adversarial']['hidden_states'],
                example['adversarial']['token_probs']
            )
            
            original_predictions.append(original_pred['harm_probability'].item())
            adversarial_predictions.append(adversarial_pred['harm_probability'].item())
            
    # Compute robustness metrics
    original_probs = np.array(original_predictions)
    adversarial_probs = np.array(adversarial_predictions)
    
    # Prediction stability
    prediction_difference = np.abs(adversarial_probs - original_probs)
    avg_prediction_difference = np.mean(prediction_difference)
    max_prediction_difference = np.max(prediction_difference)
    
    # Robustness score (inverse of prediction difference)
    robustness_score = 1.0 / (1.0 + avg_prediction_difference)
    
    # Evasion rate (how often adversarial examples fool the model)
    original_labels = (original_probs > config.harm_threshold).astype(int)
    adversarial_labels = (adversarial_probs > config.harm_threshold).astype(int)
    
    evasion_rate = np.mean(original_labels != adversarial_labels)
    
    metrics = {
        'avg_prediction_difference': avg_prediction_difference,
        'max_prediction_difference': max_prediction_difference,
        'robustness_score': robustness_score,
        'evasion_rate': evasion_rate,
        'prediction_stability': 1.0 - avg_prediction_difference
    }
    
    return metrics


def compute_computational_overhead(model, sequence_lengths: List[int],
                                 num_runs: int = 10) -> Dict[str, float]:
    """
    Measure computational cost of ASAN monitoring
    
    Args:
        model: ASAN predictor model
        sequence_lengths: List of sequence lengths to test
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with computational metrics
    """
    
    model.eval()
    
    latencies = []
    memory_usage = []
    
    for seq_len in sequence_lengths:
        # Create dummy trajectory
        dummy_trajectory = create_dummy_trajectory(seq_len)
        
        # Measure latency
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(
                    dummy_trajectory['attention_patterns'],
                    dummy_trajectory['hidden_states'],
                    dummy_trajectory['token_probs']
                )
                
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_latency = np.mean(times)
        latencies.append(avg_latency)
        
        # Measure memory usage (simplified)
        memory_usage.append(seq_len * 0.1)  # Rough estimate
        
    # Compute metrics
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    latency_per_token = avg_latency / np.mean(sequence_lengths)
    
    avg_memory = np.mean(memory_usage)
    
    metrics = {
        'avg_latency_ms': avg_latency * 1000,
        'max_latency_ms': max_latency * 1000,
        'latency_per_token_ms': latency_per_token * 1000,
        'avg_memory_mb': avg_memory,
        'throughput_tokens_per_second': 1.0 / latency_per_token
    }
    
    return metrics


def create_dummy_trajectory(seq_len: int) -> Dict:
    """Create dummy trajectory for testing"""
    
    trajectory = {
        'attention_patterns': {
            0: [torch.randn(10, 10) for _ in range(seq_len)],
            6: [torch.randn(10, 10) for _ in range(seq_len)],
            11: [torch.randn(10, 10) for _ in range(seq_len)]
        },
        'hidden_states': {
            0: [torch.randn(10, 768) for _ in range(seq_len)],
            6: [torch.randn(10, 768) for _ in range(seq_len)],
            11: [torch.randn(10, 768) for _ in range(seq_len)]
        },
        'token_probs': [torch.randn(50257) for _ in range(seq_len)]
    }
    
    return trajectory


def plot_evaluation_results(metrics: Dict[str, Any], save_path: str = None):
    """Plot evaluation results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Confusion Matrix
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
    # 2. ROC Curve
    if 'roc_curve' in metrics:
        fpr, tpr, _ = metrics['roc_curve']
        axes[0, 1].plot(fpr, tpr, label=f'AUC = {metrics["auc_roc"]:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        
    # 3. Precision-Recall Curve
    if 'precision_recall_curve' in metrics:
        precision, recall, _ = metrics['precision_recall_curve']
        axes[0, 2].plot(recall, precision, label=f'AUC-PR = {metrics["auc_pr"]:.3f}')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend()
        
    # 4. Early Detection Lead Times
    if 'early_detection_metrics' in metrics:
        ed_metrics = metrics['early_detection_metrics']
        lead_times = [ed_metrics['avg_lead_time']]
        axes[1, 0].bar(['Average Lead Time'], lead_times)
        axes[1, 0].set_title('Early Detection Performance')
        axes[1, 0].set_ylabel('Tokens Before Harm')
        
    # 5. Intervention Effectiveness
    if 'intervention_metrics' in metrics:
        int_metrics = metrics['intervention_metrics']
        categories = ['Harm Reduction', 'False Positive Rate', 'Efficiency']
        values = [
            int_metrics['harm_reduction_rate'],
            int_metrics['false_positive_rate'],
            int_metrics['intervention_efficiency']
        ]
        axes[1, 1].bar(categories, values)
        axes[1, 1].set_title('Intervention Effectiveness')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
    # 6. Computational Overhead
    if 'computational_metrics' in metrics:
        comp_metrics = metrics['computational_metrics']
        categories = ['Latency (ms)', 'Memory (MB)', 'Throughput (tokens/s)']
        values = [
            comp_metrics['avg_latency_ms'],
            comp_metrics['avg_memory_mb'],
            comp_metrics['throughput_tokens_per_second']
        ]
        axes[1, 2].bar(categories, values)
        axes[1, 2].set_title('Computational Performance')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def comprehensive_evaluation(model, test_data: List[Dict], 
                          config: EvaluationConfig) -> Dict[str, Any]:
    """
    Run comprehensive evaluation of ASAN model
    
    Args:
        model: Trained ASAN predictor
        test_data: Test dataset
        config: Evaluation configuration
        
    Returns:
        Dictionary with all evaluation metrics
    """
    
    print("Running comprehensive evaluation...")
    
    # Prepare data
    predictions = []
    ground_truth = []
    temporal_predictions = []
    harm_occurrence_timesteps = []
    
    model.eval()
    
    with torch.no_grad():
        for trajectory in test_data:
            # Get prediction
            pred = model(
                trajectory['attention_patterns'],
                trajectory['hidden_states'],
                trajectory['token_probs']
            )
            predictions.append(pred)
            
            # Ground truth
            label = 1 if trajectory['label'] == 'harmful' else 0
            ground_truth.append(label)
            
            # Temporal predictions
            temporal_preds = model.predict_at_each_timestep(
                trajectory['attention_patterns'],
                trajectory['hidden_states'],
                trajectory['token_probs']
            )
            temporal_predictions.append(temporal_preds)
            
            # Harm occurrence timestep (simplified)
            harm_timestep = len(trajectory['generated_tokens']) // 2
            harm_occurrence_timesteps.append(harm_timestep)
            
    # Convert to tensors
    ground_truth = torch.tensor(ground_truth)
    
    # Compute metrics
    results = {}
    
    # Basic prediction metrics
    results['prediction_metrics'] = compute_prediction_metrics(
        {'harm_probability': torch.stack([p['harm_probability'] for p in predictions])},
        ground_truth
    )
    
    # Early detection metrics
    results['early_detection_metrics'] = compute_early_detection_metrics(
        temporal_predictions, ground_truth, harm_occurrence_timesteps, config
    )
    
    # Computational metrics
    sequence_lengths = [len(traj['generated_tokens']) for traj in test_data]
    results['computational_metrics'] = compute_computational_overhead(
        model, sequence_lengths
    )
    
    print("Evaluation complete!")
    
    return results


class EvaluationLogger:
    """Logger for evaluation results"""
    
    def __init__(self, log_path: str = "evaluation/logs"):
        self.log_path = log_path
        Path(log_path).mkdir(parents=True, exist_ok=True)
        
    def log_evaluation(self, results: Dict[str, Any], experiment_name: str):
        """Log evaluation results"""
        
        log_entry = {
            'experiment_name': experiment_name,
            'timestamp': time.time(),
            'results': results
        }
        
        log_file = Path(self.log_path) / f"{experiment_name}_evaluation.json"
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
            
    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments"""
        
        comparison = {}
        
        for name in experiment_names:
            log_file = Path(self.log_path) / f"{name}_evaluation.json"
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    comparison[name] = data['results']
                    
        return comparison
