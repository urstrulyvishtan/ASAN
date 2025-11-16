"""
Curriculum Learning for Safety Training

Progressive difficulty in safety scenarios to improve learning efficiency.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .multi_turn_environment import MultiTurnSafetyEnvironment, UserType


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    start_difficulty: DifficultyLevel = DifficultyLevel.EASY
    progression_threshold: float = 0.8  # Success rate needed to progress
    min_episodes_per_level: int = 100
    difficulty_schedule: Dict[DifficultyLevel, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.difficulty_schedule is None:
            self.difficulty_schedule = {
                DifficultyLevel.EASY: {
                    'normal_user_prob': 0.9,
                    'mild_adversarial_prob': 0.1,
                    'highly_adversarial_prob': 0.0,
                    'max_turns': 3
                },
                DifficultyLevel.MEDIUM: {
                    'normal_user_prob': 0.7,
                    'mild_adversarial_prob': 0.2,
                    'highly_adversarial_prob': 0.1,
                    'max_turns': 5
                },
                DifficultyLevel.HARD: {
                    'normal_user_prob': 0.5,
                    'mild_adversarial_prob': 0.3,
                    'highly_adversarial_prob': 0.2,
                    'max_turns': 7
                },
                DifficultyLevel.EXPERT: {
                    'normal_user_prob': 0.3,
                    'mild_adversarial_prob': 0.3,
                    'highly_adversarial_prob': 0.4,
                    'max_turns': 10
                }
            }


class CurriculumLearning:
    """
    Curriculum learning for progressive safety training
    
    Strategy:
    - Start with easy scenarios (mostly normal users)
    - Gradually increase difficulty as model improves
    - Adaptively adjust based on performance
    """
    
    def __init__(self, config: CurriculumConfig):
        """Initialize curriculum learning"""
        self.config = config
        self.current_difficulty = config.start_difficulty
        self.episode_count = 0
        self.success_history = []  # Track success rate
        self.performance_by_level = {
            level: [] for level in DifficultyLevel
        }
    
    def get_current_difficulty_config(self) -> Dict[str, float]:
        """Get configuration for current difficulty level"""
        return self.config.difficulty_schedule[self.current_difficulty].copy()
    
    def update_performance(self, success: bool, episode_reward: float):
        """
        Update performance tracking
        
        Args:
            success: Whether episode was successful (no harm detected)
            episode_reward: Reward for this episode
        """
        self.episode_count += 1
        self.success_history.append(1.0 if success else 0.0)
        self.performance_by_level[self.current_difficulty].append({
            'success': success,
            'reward': episode_reward
        })
        
        # Check if ready to progress
        if self._should_progress():
            self._progress_difficulty()
    
    def _should_progress(self) -> bool:
        """Check if ready to progress to next difficulty level"""
        # Need minimum episodes at current level
        current_level_performance = self.performance_by_level[self.current_difficulty]
        if len(current_level_performance) < self.config.min_episodes_per_level:
            return False
        
        # Compute success rate
        recent_performance = current_level_performance[-self.config.min_episodes_per_level:]
        success_rate = np.mean([p['success'] for p in recent_performance])
        
        # Progress if success rate exceeds threshold
        return success_rate >= self.config.progression_threshold
    
    def _progress_difficulty(self):
        """Move to next difficulty level"""
        difficulty_order = [
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
            DifficultyLevel.EXPERT
        ]
        
        current_idx = difficulty_order.index(self.current_difficulty)
        
        if current_idx < len(difficulty_order) - 1:
            self.current_difficulty = difficulty_order[current_idx + 1]
            print(f"Progressed to difficulty level: {self.current_difficulty.value}")
        else:
            # Already at highest level
            pass
    
    def adapt_difficulty(self, recent_performance: List[bool]) -> Optional[DifficultyLevel]:
        """
        Adaptively adjust difficulty based on recent performance
        
        Returns:
            New difficulty level if adjustment needed, None otherwise
        """
        if len(recent_performance) < 50:
            return None
        
        success_rate = np.mean(recent_performance)
        
        # If performance is too poor, decrease difficulty
        if success_rate < 0.5 and self.current_difficulty != DifficultyLevel.EASY:
            difficulty_order = [
                DifficultyLevel.EASY,
                DifficultyLevel.MEDIUM,
                DifficultyLevel.HARD,
                DifficultyLevel.EXPERT
            ]
            current_idx = difficulty_order.index(self.current_difficulty)
            if current_idx > 0:
                new_difficulty = difficulty_order[current_idx - 1]
                self.current_difficulty = new_difficulty
                print(f"Decreased difficulty to: {new_difficulty.value}")
                return new_difficulty
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum learning statistics"""
        stats = {
            'current_difficulty': self.current_difficulty.value,
            'episode_count': self.episode_count,
            'overall_success_rate': np.mean(self.success_history) if self.success_history else 0.0
        }
        
        # Per-level statistics
        for level in DifficultyLevel:
            level_perf = self.performance_by_level[level]
            if level_perf:
                stats[f'{level.value}_episodes'] = len(level_perf)
                stats[f'{level.value}_success_rate'] = np.mean([p['success'] for p in level_perf])
                stats[f'{level.value}_avg_reward'] = np.mean([p['reward'] for p in level_perf])
            else:
                stats[f'{level.value}_episodes'] = 0
                stats[f'{level.value}_success_rate'] = 0.0
                stats[f'{level.value}_avg_reward'] = 0.0
        
        return stats
    
    def should_sample_challenge(self) -> bool:
        """
        Decide whether to sample a challenging example
        
        Useful for maintaining diversity and preventing overfitting
        """
        # Sample challenging examples with some probability
        challenge_prob = 0.1  # 10% chance
        
        if np.random.random() < challenge_prob:
            return True
        
        return False
