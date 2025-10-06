"""
Training Pipeline for ASAN

Training pipeline for ASAN predictor with contrastive learning and adversarial training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from dataclasses import dataclass

from ..models.asan_predictor import ASANPredictor, ASANConfig
from ..data.synthetic_llm_simulator import SyntheticLLMSimulator, SimulatorConfig


@dataclass
class TrainingConfig:
    """Configuration for ASAN training"""
    # Data
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Training phases
    pretrain_epochs: int = 50
    finetune_epochs: int = 30
    adversarial_epochs: int = 20
    
    # Loss weights
    classification_weight: float = 1.0
    category_weight: float = 0.5
    temporal_weight: float = 0.2
    contrastive_weight: float = 0.3
    confidence_weight: float = 0.1
    
    # Data augmentation
    noise_std: float = 0.01
    temporal_shift_prob: float = 0.3
    dropout_prob: float = 0.1
    
    # Validation
    validation_frequency: int = 5
    early_stopping_patience: int = 10
    
    # Paths
    save_path: str = "checkpoints"
    log_path: str = "logs"


class TrajectoryDataset(Dataset):
    """Dataset for trajectory training"""
    
    def __init__(self, trajectories: List[Dict], config: TrainingConfig):
        self.trajectories = trajectories
        self.config = config
        
    def __len__(self):
        return len(self.trajectories)
        
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        
        # Extract features
        attention_patterns = trajectory['attention_patterns']
        hidden_states = trajectory['hidden_states']
        token_probs = trajectory['token_probs']
        
        # Convert to tensors
        attention_tensors = {}
        for layer_idx, attn_list in attention_patterns.items():
            # Pad per-timestep attention matrices to the largest square in this trajectory
            # Each entry is [seq_len, seq_len] and seq_len can vary by timestep
            max_len = max(a.shape[0] for a in attn_list)
            padded = []
            for a in attn_list:
                if a.shape[0] == max_len:
                    padded.append(a)
                else:
                    pad_rows = max_len - a.shape[0]
                    # pad to [max_len, max_len]
                    pad_tensor = torch.zeros(max_len, max_len, dtype=a.dtype, device=a.device)
                    pad_tensor[:a.shape[0], :a.shape[1]] = a
                    padded.append(pad_tensor)
            attention_tensors[layer_idx] = torch.stack(padded)
            
        hidden_tensors = {}
        for layer_idx, hidden_list in hidden_states.items():
            # Pad per-timestep hidden states to largest seq_len in this trajectory for this layer
            # Each entry is [seq_len, hidden_dim]
            max_len = max(h.shape[0] for h in hidden_list)
            hidden_dim = hidden_list[0].shape[1]
            padded = []
            for h in hidden_list:
                if h.shape[0] == max_len:
                    padded.append(h)
                else:
                    pad_rows = max_len - h.shape[0]
                    pad_tensor = torch.zeros(max_len, hidden_dim, dtype=h.dtype, device=h.device)
                    pad_tensor[:h.shape[0], :h.shape[1]] = h
                    padded.append(pad_tensor)
            hidden_tensors[layer_idx] = torch.stack(padded)
            
        token_prob_tensors = torch.stack(token_probs)
        
        # Labels
        label = 1 if trajectory['label'] == 'harmful' else 0
        category = self._get_category_label(trajectory)
        
        return {
            'attention_patterns': attention_tensors,
            'hidden_states': hidden_tensors,
            'token_probs': token_prob_tensors,
            'label': torch.tensor(label, dtype=torch.float32),
            'category': torch.tensor(category, dtype=torch.long),
            'trajectory_id': trajectory.get('id', f'traj_{idx}')
        }
        
    def _get_category_label(self, trajectory: Dict) -> int:
        """Convert harm category to integer label"""
        category_mapping = {
            'jailbreak': 0,
            'hallucination': 1,
            'bias': 2,
            'harmful_instruction_following': 3,
            'privacy_violations': 4
        }
        
        harm_type = trajectory.get('harm_type', trajectory.get('category', 'jailbreak'))
        return category_mapping.get(harm_type, 0)


class ASANTrainingPipeline:
    """
    Training pipeline for ASAN predictor
    
    Training strategy:
    1. Pre-train on synthetic data
    2. Fine-tune on real LLM trajectories  
    3. Adversarial training with hard negatives
    4. Continual learning as new harm patterns emerge
    """
    
    def __init__(self, config: TrainingConfig, asan_config: ASANConfig):
        self.config = config
        self.asan_config = asan_config
        
        # Initialize model
        self.model = ASANPredictor(asan_config)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss functions
        self.classification_loss = nn.BCELoss()
        self.category_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = nn.TripletMarginLoss(margin=1.0)
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
    def pretrain_on_synthetic(self, synthetic_dataset: List[Dict], epochs: int = 50):
        """
        Pre-train on synthetic LLM trajectories
        
        Benefits:
        - Fast iteration
        - Controlled ground truth
        - No need for expensive real LLM inference
        
        Loss: Binary cross-entropy for harm classification
        """
        print("Starting synthetic pre-training...")
        
        # Create dataset
        dataset = TrajectoryDataset(synthetic_dataset, self.config)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Forward pass
                predictions = self.model(
                    batch['attention_patterns'],
                    batch['hidden_states'],
                    batch['token_probs']
                )
                
                # Compute losses
                classification_loss = self.classification_loss(
                    predictions['harm_probability'], 
                    batch['label']
                )
                
                category_loss = self.category_loss(
                    predictions['harm_category'],
                    batch['category']
                )
                
                # Total loss
                loss = (self.config.classification_weight * classification_loss + 
                       self.config.category_weight * category_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                
                # Accuracy
                predicted_labels = (predictions['harm_probability'] > 0.5).float()
                correct_predictions += (predicted_labels == batch['label']).sum().item()
                total_predictions += batch['label'].size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct_predictions/total_predictions:.4f}"
                })
                
            # Epoch statistics
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            
            # Update learning rate
            self.scheduler.step(avg_loss)
            
    def finetune_on_real_data(self, train_trajectories: List[Dict], 
                             val_trajectories: List[Dict], epochs: int = 30):
        """
        Fine-tune on real LLM trajectories
        
        Dataset split:
        - 8000 training
        - 1000 validation
        - 1000 held-out test
        
        Losses:
        - Classification loss (harm vs safe)
        - Category loss (type of harm)
        - Temporal consistency loss (predictions should be stable over time)
        - Confidence calibration loss (match predicted confidence to accuracy)
        """
        print("Starting fine-tuning on real data...")
        
        # Create datasets
        train_dataset = TrajectoryDataset(train_trajectories, self.config)
        val_dataset = TrajectoryDataset(val_trajectories, self.config)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
                
                # Save best model
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.epochs_without_improvement += 1
                
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Single training epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in dataloader:
            # Forward pass
            predictions = self.model(
                batch['attention_patterns'],
                batch['hidden_states'],
                batch['token_probs']
            )
            
            # Compute losses
            classification_loss = self.classification_loss(
                predictions['harm_probability'], 
                batch['label']
            )
            
            category_loss = self.category_loss(
                predictions['harm_category'],
                batch['category']
            )
            
            # Temporal consistency loss
            temporal_loss = self._compute_temporal_consistency_loss(
                batch['attention_patterns'],
                batch['hidden_states'],
                batch['token_probs']
            )
            
            # Confidence calibration loss
            confidence_loss = self._compute_confidence_calibration_loss(
                predictions, batch['label']
            )
            
            # Total loss
            loss = (self.config.classification_weight * classification_loss + 
                   self.config.category_weight * category_loss +
                   self.config.temporal_weight * temporal_loss +
                   self.config.confidence_weight * confidence_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Accuracy
            predicted_labels = (predictions['harm_probability'] > 0.5).float()
            correct_predictions += (predicted_labels == batch['label']).sum().item()
            total_predictions += batch['label'].size(0)
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
        
    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Single validation epoch"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Forward pass
                predictions = self.model(
                    batch['attention_patterns'],
                    batch['hidden_states'],
                    batch['token_probs']
                )
                
                # Compute losses
                classification_loss = self.classification_loss(
                    predictions['harm_probability'], 
                    batch['label']
                )
                
                category_loss = self.category_loss(
                    predictions['harm_category'],
                    batch['category']
                )
                
                # Total loss
                loss = (self.config.classification_weight * classification_loss + 
                       self.config.category_weight * category_loss)
                
                # Statistics
                total_loss += loss.item()
                
                # Accuracy
                predicted_labels = (predictions['harm_probability'] > 0.5).float()
                correct_predictions += (predicted_labels == batch['label']).sum().item()
                total_predictions += batch['label'].size(0)
                
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
        
    def contrastive_learning(self, trajectory_pairs: List[Tuple[Dict, Dict]]):
        """
        Learn to distinguish safe vs harmful patterns
        
        Training pairs:
        - Positive: similar harm trajectories should have similar embeddings
        - Negative: safe and harmful should be far apart
        
        Loss: Contrastive loss with margin
        """
        print("Starting contrastive learning...")
        
        self.model.train()
        
        for epoch in range(10):  # Fewer epochs for contrastive learning
            total_loss = 0
            
            for safe_traj, harmful_traj in trajectory_pairs:
                # Get embeddings
                safe_embedding = self.model.get_spectral_signature(safe_traj)
                harmful_embedding = self.model.get_spectral_signature(harmful_traj)
                
                # Create anchor, positive, negative
                anchor = safe_embedding
                positive = harmful_embedding  # Similar harmful pattern
                negative = safe_embedding + torch.randn_like(safe_embedding) * 0.1
                
                # Contrastive loss
                loss = self.contrastive_loss(anchor, positive, negative)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Contrastive Epoch {epoch+1}: Loss={total_loss/len(trajectory_pairs):.4f}")
            
    def adversarial_training(self, model: ASANPredictor, adversarial_examples: List[Dict]):
        """
        Train on adversarial examples that try to fool ASAN
        
        Generate adversarial trajectories:
        - Slightly perturb harmful trajectories to look safe
        - Create "edge case" trajectories near decision boundary
        
        Goal: Make ASAN robust to evasion attempts
        """
        print("Starting adversarial training...")
        
        self.model.train()
        
        for epoch in range(20):
            total_loss = 0
            
            for adv_example in adversarial_examples:
                # Forward pass
                predictions = self.model(
                    adv_example['attention_patterns'],
                    adv_example['hidden_states'],
                    adv_example['token_probs']
                )
                
                # Adversarial loss: encourage correct classification despite perturbations
                true_label = torch.tensor(1.0 if adv_example['label'] == 'harmful' else 0.0)
                
                loss = self.classification_loss(
                    predictions['harm_probability'],
                    true_label.unsqueeze(0)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Adversarial Epoch {epoch+1}: Loss={total_loss/len(adversarial_examples):.4f}")
            
    def _compute_temporal_consistency_loss(self, attention_patterns: Dict, 
                                          hidden_states: Dict, 
                                          token_probs: List[torch.Tensor]) -> torch.Tensor:
        """Compute temporal consistency loss"""
        
        # Get predictions at each timestep
        temporal_predictions = self.model.predict_at_each_timestep(
            attention_patterns, hidden_states, token_probs
        )
        
        if len(temporal_predictions) < 2:
            return torch.tensor(0.0)
            
        # Compute consistency between consecutive predictions
        consistency_loss = 0.0
        
        for i in range(len(temporal_predictions) - 1):
            pred1 = temporal_predictions[i]['harm_probability']
            pred2 = temporal_predictions[i + 1]['harm_probability']
            
            # Encourage smooth transitions
            consistency_loss += F.mse_loss(pred1, pred2)
            
        return consistency_loss / (len(temporal_predictions) - 1)
        
    def _compute_confidence_calibration_loss(self, predictions: Dict, 
                                           labels: torch.Tensor) -> torch.Tensor:
        """Compute confidence calibration loss"""
        
        predicted_probs = predictions['harm_probability']
        predicted_confidences = predictions['confidence']
        
        # Compute accuracy for each confidence bin
        confidence_bins = torch.linspace(0, 1, 10)
        calibration_loss = 0.0
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (predicted_confidences >= confidence_bins[i]) & \
                      (predicted_confidences < confidence_bins[i + 1])
            
            if bin_mask.sum() > 0:
                bin_probs = predicted_probs[bin_mask]
                bin_labels = labels[bin_mask]
                bin_confidences = predicted_confidences[bin_mask]
                
                # Accuracy should match confidence
                accuracy = (bin_probs > 0.5).float().mean()
                avg_confidence = bin_confidences.mean()
                
                calibration_loss += F.mse_loss(accuracy, avg_confidence)
                
        return calibration_loss
        
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        
        # Pad sequences to same length
        max_length = max(len(item['token_probs']) for item in batch)
        
        collated_batch = {
            'attention_patterns': {},
            'hidden_states': {},
            'token_probs': [],
            'labels': [],
            'categories': [],
            'trajectory_ids': []
        }
        
        # Get all layer indices
        all_attention_layers = set()
        all_hidden_layers = set()
        
        for item in batch:
            all_attention_layers.update(item['attention_patterns'].keys())
            all_hidden_layers.update(item['hidden_states'].keys())
            
        # Pad attention patterns (pad time and spatial dims per layer across batch)
        for layer_idx in all_attention_layers:
            layer_tensors = []
            # Determine target spatial size for this layer across batch
            target_spatial = 0
            for item in batch:
                if layer_idx in item['attention_patterns']:
                    attn_tensor = item['attention_patterns'][layer_idx]  # [T, S, S]
                    target_spatial = max(target_spatial, attn_tensor.shape[1])
            if target_spatial == 0:
                target_spatial = 10
            for item in batch:
                if layer_idx in item['attention_patterns']:
                    attn_tensor = item['attention_patterns'][layer_idx]
                    # Pad time dimension to max_length
                    if attn_tensor.size(0) < max_length:
                        pad_time = torch.zeros(max_length - attn_tensor.size(0), attn_tensor.size(1), attn_tensor.size(2))
                        attn_tensor = torch.cat([attn_tensor, pad_time], dim=0)
                    # Pad spatial dims to target_spatial
                    if attn_tensor.size(1) < target_spatial:
                        s = attn_tensor.size(1)
                        pad_spatial = torch.zeros(max_length, target_spatial, target_spatial)
                        pad_spatial[:, :s, :s] = attn_tensor
                        attn_tensor = pad_spatial
                    layer_tensors.append(attn_tensor)
                else:
                    # Create dummy tensor
                    dummy_tensor = torch.zeros(max_length, target_spatial, target_spatial)
                    layer_tensors.append(dummy_tensor)
            collated_batch['attention_patterns'][layer_idx] = torch.stack(layer_tensors)
            
        # Pad hidden states (pad time and spatial/hidden dims across batch)
        for layer_idx in all_hidden_layers:
            layer_tensors = []
            target_seq_len = 0
            target_hidden_dim = 0
            for item in batch:
                if layer_idx in item['hidden_states']:
                    hidden_tensor = item['hidden_states'][layer_idx]
                    # hidden_tensor may be [T, S, H] or [T, H]
                    if hidden_tensor.dim() == 3:
                        target_seq_len = max(target_seq_len, hidden_tensor.shape[1])
                        target_hidden_dim = max(target_hidden_dim, hidden_tensor.shape[2])
                    else:
                        target_hidden_dim = max(target_hidden_dim, hidden_tensor.shape[1])
            if target_hidden_dim == 0:
                target_hidden_dim = 768
            if target_seq_len == 0:
                target_seq_len = 10
            for item in batch:
                if layer_idx in item['hidden_states']:
                    hidden_tensor = item['hidden_states'][layer_idx]
                    if hidden_tensor.dim() == 2:
                        # [T, H] -> pad time then hidden
                        if hidden_tensor.size(0) < max_length:
                            pad_time = torch.zeros(max_length - hidden_tensor.size(0), hidden_tensor.size(1))
                            hidden_tensor = torch.cat([hidden_tensor, pad_time], dim=0)
                        if hidden_tensor.size(1) < target_hidden_dim:
                            h = hidden_tensor.size(1)
                            pad_hidden = torch.zeros(max_length, target_hidden_dim)
                            pad_hidden[:, :h] = hidden_tensor
                            hidden_tensor = pad_hidden
                    else:
                        # [T, S, H] -> pad time, seq_len, hidden_dim
                        T, S, H = hidden_tensor.shape
                        if T < max_length:
                            pad_time = torch.zeros(max_length - T, S, H)
                            hidden_tensor = torch.cat([hidden_tensor, pad_time], dim=0)
                        if S < target_seq_len or H < target_hidden_dim:
                            pad_full = torch.zeros(max_length, target_seq_len, target_hidden_dim)
                            pad_full[:hidden_tensor.shape[0], :S, :H] = hidden_tensor
                            hidden_tensor = pad_full
                    layer_tensors.append(hidden_tensor)
                else:
                    # Create dummy tensor [T, S, H]
                    dummy_tensor = torch.zeros(max_length, target_seq_len, target_hidden_dim)
                    layer_tensors.append(dummy_tensor)
            collated_batch['hidden_states'][layer_idx] = torch.stack(layer_tensors)
            
        # Pad token probabilities
        for item in batch:
            token_probs = item['token_probs']
            if token_probs.size(0) < max_length:
                padding = torch.zeros(max_length - token_probs.size(0), token_probs.size(1))
                token_probs = torch.cat([token_probs, padding], dim=0)
            collated_batch['token_probs'].append(token_probs)
            
        # Stack other tensors
        collated_batch['token_probs'] = torch.stack(collated_batch['token_probs'])
        collated_batch['labels'] = torch.stack([item['label'] for item in batch])
        collated_batch['categories'] = torch.stack([item['category'] for item in batch])
        collated_batch['trajectory_ids'] = [item['trajectory_id'] for item in batch]
        
        return collated_batch
        
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'training_history': self.training_history
        }
        
        # Save checkpoint
        checkpoint_path = Path(self.config.save_path) / f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.save_path) / "best_model.pt"
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        return checkpoint['epoch']
        
    def continual_learning_update(self, new_data: List[Dict]):
        """
        Update model as new harm patterns emerge
        
        Strategy:
        - Maintain memory of old patterns (prevent forgetting)
        - Quickly adapt to new patterns
        - Monitor for distribution shift
        """
        print("Starting continual learning update...")
        
        # Create dataset for new data
        new_dataset = TrajectoryDataset(new_data, self.config)
        new_loader = DataLoader(
            new_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Fine-tune on new data with lower learning rate
        old_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = old_lr * 0.1
        
        # Train for a few epochs
        for epoch in range(5):
            train_loss, train_acc = self._train_epoch(new_loader)
            print(f"Continual Learning Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            
        # Restore learning rate
        self.optimizer.param_groups[0]['lr'] = old_lr
