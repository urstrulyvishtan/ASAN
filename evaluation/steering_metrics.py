"""
Steering Effectiveness Metrics

Measure how well steering works compared to blocking.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from dataclasses import dataclass

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class SteeringMetricsConfig:
    """Configuration for steering metrics"""
    use_semantic_similarity: bool = True
    semantic_model_name: str = 'all-MiniLM-L6-v2'
    fluency_threshold: float = 0.7


class SteeringMetrics:
    """Measure how well steering works"""
    
    def __init__(self, config: SteeringMetricsConfig = None):
        """Initialize steering metrics"""
        self.config = config or SteeringMetricsConfig()
        
        # Load semantic similarity model if available
        if self.config.use_semantic_similarity and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer(self.config.semantic_model_name)
            except:
                self.semantic_model = None
                self.config.use_semantic_similarity = False
        else:
            self.semantic_model = None
            self.config.use_semantic_similarity = False
    
    def quality_preservation_score(self, 
                                   original_output: str,
                                   steered_output: str) -> Dict[str, float]:
        """
        How much semantic quality lost due to steering?
        
        Metrics:
        - Semantic similarity
        - Fluency
        - Relevance to prompt
        - Helpfulness rating
        """
        scores = {}
        
        # 1. Semantic similarity
        semantic_sim = self._compute_semantic_similarity(original_output, steered_output)
        scores['semantic_similarity'] = semantic_sim
        
        # 2. Fluency (simplified - would use language model)
        fluency_score = self._compute_fluency(steered_output)
        scores['fluency'] = fluency_score
        
        # 3. Relevance (simplified - would use context)
        relevance_score = 0.8  # Placeholder
        scores['relevance'] = relevance_score
        
        # 4. Overall quality preservation
        overall_quality = (
            semantic_sim * 0.4 +
            fluency_score * 0.3 +
            relevance_score * 0.3
        )
        scores['overall_quality'] = overall_quality
        
        return scores
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if self.config.use_semantic_similarity and self.semantic_model is not None:
            try:
                embeddings = self.semantic_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            except:
                pass
        
        # Fallback: simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        return jaccard_sim
    
    def _compute_fluency(self, text: str) -> float:
        """Compute fluency score (simplified)"""
        # Placeholder: would use language model perplexity
        # For now, use heuristics
        
        # Check for obvious issues
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Basic heuristics
        # - Length check (too short or too long might be poor)
        words = text.split()
        if len(words) < 3:
            return 0.5
        if len(words) > 500:
            return 0.7
        
        # - Repetition check (high repetition = poor fluency)
        unique_words = set(words)
        repetition_ratio = 1.0 - (len(unique_words) / len(words)) if words else 0.0
        if repetition_ratio > 0.5:
            return 0.5
        
        # Default: assume fluent
        return 0.8
    
    def steering_necessity_analysis(self, steering_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Were steering interventions necessary?
        
        Metrics:
        - True positive steering (actually prevented harm)
        - False positive steering (over-cautious)
        - Missed opportunities (should have steered but didn't)
        """
        analysis = {
            'total_interventions': len(steering_log),
            'true_positives': 0,
            'false_positives': 0,
            'missed_opportunities': 0
        }
        
        for log_entry in steering_log:
            harm_prob = log_entry.get('harm_probability', 0.0)
            actual_label = log_entry.get('actual_label', None)
            
            if actual_label == 'harmful' and harm_prob > 0.5:
                analysis['true_positives'] += 1
            elif actual_label == 'safe' and harm_prob > 0.5:
                analysis['false_positives'] += 1
        
        # Compute rates
        if analysis['total_interventions'] > 0:
            analysis['true_positive_rate'] = analysis['true_positives'] / analysis['total_interventions']
            analysis['false_positive_rate'] = analysis['false_positives'] / analysis['total_interventions']
        else:
            analysis['true_positive_rate'] = 0.0
            analysis['false_positive_rate'] = 0.0
        
        return analysis
    
    def steering_efficiency(self, 
                           steering_log: List[Dict[str, Any]]) -> Dict[str, float]:
        """Measure efficiency of steering (latency, overhead)"""
        if not steering_log:
            return {
                'avg_latency_ms': 0.0,
                'total_overhead_ms': 0.0,
                'steering_frequency': 0.0
            }
        
        latencies = [log.get('latency_ms', 0.0) for log in steering_log]
        
        return {
            'avg_latency_ms': np.mean(latencies) if latencies else 0.0,
            'total_overhead_ms': sum(latencies),
            'steering_frequency': len(steering_log) / 100.0  # Interventions per 100 tokens
        }
    
    def compare_steering_vs_blocking(self,
                                     steering_results: List[Dict[str, Any]],
                                     blocking_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compare steering vs blocking approaches"""
        # Output generation rate
        steering_output_rate = sum(
            1 for r in steering_results if r.get('output_generated', False)
        ) / len(steering_results) if steering_results else 0.0
        
        blocking_output_rate = sum(
            1 for r in blocking_results if r.get('output_generated', False)
        ) / len(blocking_results) if blocking_results else 0.0
        
        # Average quality
        steering_quality = np.mean([
            r.get('quality_score', 0.0) for r in steering_results
            if r.get('output_generated', False)
        ]) if steering_results else 0.0
        
        blocking_quality = np.mean([
            r.get('quality_score', 0.0) for r in blocking_results
            if r.get('output_generated', False)
        ]) if blocking_results else 0.0
        
        # Harm prevention rate
        steering_prevention = sum(
            1 for r in steering_results
            if r.get('true_label') == 'harmful' and r.get('steering_applied', False)
        ) / max(1, sum(1 for r in steering_results if r.get('true_label') == 'harmful'))
        
        blocking_prevention = sum(
            1 for r in blocking_results
            if r.get('true_label') == 'harmful' and r.get('blocking_applied', False)
        ) / max(1, sum(1 for r in blocking_results if r.get('true_label') == 'harmful'))
        
        return {
            'steering_output_rate': steering_output_rate,
            'blocking_output_rate': blocking_output_rate,
            'steering_quality': steering_quality,
            'blocking_quality': blocking_quality,
            'steering_prevention_rate': steering_prevention,
            'blocking_prevention_rate': blocking_prevention,
            'quality_improvement': steering_quality - blocking_quality,
            'output_rate_improvement': steering_output_rate - blocking_output_rate
        }
