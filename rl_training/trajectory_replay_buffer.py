"""
Trajectory Replay Buffer

Store multi-turn trajectories for learning from past interactions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass


@dataclass
class ReplayBufferConfig:
    """Configuration for replay buffer"""
    buffer_size: int = 10000
    min_buffer_size: int = 1000
    priority_sampling: bool = True
    priority_alpha: float = 0.6  # Priority exponent
    priority_beta: float = 0.4  # Importance sampling exponent


class TrajectoryReplayBuffer:
    """
    Replay buffer for storing multi-turn trajectories
    
    Features:
    - Stores complete trajectories with rewards
    - Supports priority sampling (sample important trajectories)
    - Maintains balance between safe and harmful trajectories
    """
    
    def __init__(self, config: ReplayBufferConfig):
        """Initialize replay buffer"""
        self.config = config
        self.buffer = deque(maxlen=config.buffer_size)
        self.priorities = deque(maxlen=config.buffer_size)
        self.buffer_index = 0
        
        # Statistics
        self.total_added = 0
        self.harmful_trajectories = 0
        self.safe_trajectories = 0
    
    def add(self, trajectory: Dict[str, Any], reward: float, harm_probability: float):
        """
        Add trajectory to buffer
        
        Args:
            trajectory: Complete trajectory data
            reward: Reward for this trajectory
            harm_probability: Harm probability from ASAN
        """
        # Store trajectory
        self.buffer.append({
            'trajectory': trajectory,
            'reward': reward,
            'harm_probability': harm_probability,
            'index': self.total_added
        })
        
        # Compute priority (higher priority for high-reward or high-harm trajectories)
        priority = self._compute_priority(reward, harm_probability)
        self.priorities.append(priority)
        
        # Update statistics
        self.total_added += 1
        if harm_probability > 0.5:
            self.harmful_trajectories += 1
        else:
            self.safe_trajectories += 1
    
    def _compute_priority(self, reward: float, harm_probability: float) -> float:
        """Compute priority for sampling"""
        if self.config.priority_sampling:
            # High priority for:
            # - High rewards (good examples)
            # - High harm probability (important negative examples)
            priority = abs(reward) + harm_probability * 0.5
            return priority ** self.config.priority_alpha
        else:
            return 1.0
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample batch of trajectories
        
        If priority sampling enabled, samples according to priorities
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if self.config.priority_sampling and len(self.priorities) > 0:
            # Priority sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / (priorities.sum() + 1e-8)
            
            indices = np.random.choice(
                len(self.buffer),
                size=batch_size,
                replace=False,
                p=probabilities
            )
            
            batch = [self.buffer[idx] for idx in indices]
            
            # Compute importance sampling weights
            for i, item in enumerate(batch):
                idx = indices[i]
                priority = priorities[idx]
                prob = probabilities[idx]
                weight = (len(self.buffer) * prob) ** (-self.config.priority_beta)
                item['importance_weight'] = weight
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            batch = [self.buffer[idx] for idx in indices]
            for item in batch:
                item['importance_weight'] = 1.0
        
        return batch
    
    def sample_balanced(self, batch_size: int, harmful_ratio: float = 0.5) -> List[Dict[str, Any]]:
        """
        Sample balanced batch with specified ratio of harmful/safe trajectories
        """
        # Split trajectories
        harmful = [item for item in self.buffer if item['harm_probability'] > 0.5]
        safe = [item for item in self.buffer if item['harm_probability'] <= 0.5]
        
        # Sample from each
        num_harmful = int(batch_size * harmful_ratio)
        num_safe = batch_size - num_harmful
        
        sampled_harmful = np.random.choice(
            len(harmful),
            size=min(num_harmful, len(harmful)),
            replace=False
        )
        sampled_safe = np.random.choice(
            len(safe),
            size=min(num_safe, len(safe)),
            replace=False
        )
        
        batch = (
            [harmful[i] for i in sampled_harmful] +
            [safe[i] for i in sampled_safe]
        )
        
        # Shuffle
        np.random.shuffle(batch)
        
        # Add importance weights
        for item in batch:
            item['importance_weight'] = 1.0
        
        return batch
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        """Update priorities for trajectories"""
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority ** self.config.priority_alpha
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'harmful_ratio': 0.0,
                'safe_ratio': 0.0,
                'avg_reward': 0.0,
                'avg_harm_prob': 0.0
            }
        
        rewards = [item['reward'] for item in self.buffer]
        harm_probs = [item['harm_probability'] for item in self.buffer]
        
        return {
            'size': len(self.buffer),
            'harmful_ratio': self.harmful_trajectories / len(self.buffer),
            'safe_ratio': self.safe_trajectories / len(self.buffer),
            'avg_reward': np.mean(rewards),
            'avg_harm_prob': np.mean(harm_probs),
            'total_added': self.total_added
        }
    
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= self.config.min_buffer_size
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.priorities.clear()
        self.total_added = 0
        self.harmful_trajectories = 0
        self.safe_trajectories = 0
