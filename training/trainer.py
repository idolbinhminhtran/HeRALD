"""
Training loop implementation for HGAT-LDA model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import os

from models.hgat_lda import HGAT_LDA
from data.graph_construction import get_positive_pairs, generate_negative_pairs


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)
        # Calculate focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


class HGATLDATrainer:
    """
    Trainer class for HGAT-LDA model with multi-GPU support.
    """
    
    def __init__(self, 
                 model: HGAT_LDA,
                 device: torch.device,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 batch_size: int = 256,
                 enable_progress: bool = True,
                 neg_ratio: int = 1,
                 cosine_tmax: Optional[int] = None,
                 use_amp: bool = True,
                 use_focal_loss: bool = True,
                 label_smoothing: float = 0.1,
                 use_multi_gpu: bool = True):
        """
        Initialize the trainer with optional multi-GPU support.
        
        Args:
            model: HGAT-LDA model
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            batch_size: Batch size for training
            enable_progress: If False, suppress tqdm bars/prints
            neg_ratio: Number of negatives per positive in a batch
            cosine_tmax: If provided, use CosineAnnealingLR with T_max
            use_amp: Whether to use automatic mixed precision (AMP) for training
            use_multi_gpu: Whether to use DataParallel for multi-GPU training
        """
        # Move model to device first
        self.model = model.to(device)
        self.device = device
        
        # Enable multi-GPU if available and requested
        self.use_multi_gpu = use_multi_gpu
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"🚀 Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            # Scale batch size for multi-GPU
            self.batch_size = batch_size * torch.cuda.device_count()
            print(f"   Scaled batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size
        self.enable_progress = enable_progress
        self.neg_ratio = max(1, neg_ratio)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.label_smoothing = label_smoothing
        
        # Optimizer with improved settings
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        
        # Loss function - use focal loss for imbalanced data
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        # Mixed precision scaler
        if torch.cuda.is_available() and self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Optional scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cosine_tmax) if cosine_tmax else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def prepare_data(self, 
                    lnc_disease_assoc: torch.Tensor,
                    num_lncRNAs: int,
                    num_diseases: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_pairs = get_positive_pairs(lnc_disease_assoc)
        neg_pairs_all = generate_negative_pairs(pos_pairs, num_lncRNAs, num_diseases)
        return pos_pairs, neg_pairs_all
    
    def _move_edges_to_device(self, edges: Dict) -> Dict:
        edges_device = {}
        for key, (src, dst, w) in edges.items():
            if src is not None:
                src = src.to(self.device)
                dst = dst.to(self.device)
                if w is not None:
                    w = w.to(self.device)
            edges_device[key] = (src, dst, w)
        return edges_device
    
    def train_epoch(self, pos_pairs: torch.Tensor, neg_pairs_all: torch.Tensor, edges: Dict) -> float:
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        edges_device = self._move_edges_to_device(edges)
        perm = torch.randperm(pos_pairs.size(0))
        pos_pairs = pos_pairs[perm]
        
        # Determine batch sizes
        pos_per_batch = max(1, self.batch_size // (1 + self.neg_ratio))
        batch_iterator = range(0, pos_pairs.size(0), pos_per_batch)
        batch_iterator = tqdm(batch_iterator, desc="Training batches", disable=not self.enable_progress)
        
        for start in batch_iterator:
            end = min(start + pos_per_batch, pos_pairs.size(0))
            batch_pos = pos_pairs[start:end]
            # Sample negatives according to ratio
            neg_indices = torch.randint(0, neg_pairs_all.size(0), (batch_pos.size(0) * self.neg_ratio,))
            batch_neg = neg_pairs_all[neg_indices]
            
            batch_all = torch.cat([batch_pos, batch_neg], dim=0)
            lnc_idx_batch = batch_all[:, 0].to(self.device)
            dis_idx_batch = batch_all[:, 1].to(self.device)
            
            # Create labels with label smoothing
            pos_labels = torch.ones(batch_pos.size(0)) * (1.0 - self.label_smoothing)
            neg_labels = torch.zeros(batch_neg.size(0)) + self.label_smoothing
            labels = torch.cat([pos_labels, neg_labels], dim=0).to(self.device)
            
            # Use the new autocast API if AMP is enabled
            if self.use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = self.model(lnc_idx_batch, dis_idx_batch, edges_device)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(lnc_idx_batch, dis_idx_batch, edges_device)
                loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / max(num_batches, 1)
    
    def train(self, 
             pos_pairs: torch.Tensor,
             neg_pairs_all: torch.Tensor,
             edges: Dict,
             num_epochs: int = 50,
             val_split: float = 0.1,
             early_stopping_patience: int = 10,
             save_path: Optional[str] = None) -> Dict[str, list]:
        if self.enable_progress:
            print("\nGraph Statistics:")
            total_edges = 0
            for (src_type, rel_name, dst_type), (src_idx, dst_idx, w) in edges.items():
                num_edges = len(src_idx) if src_idx is not None else 0
                print(f"  {src_type} -> {dst_type} ({rel_name}): {num_edges} edges")
                total_edges += num_edges
            print(f"  Total edges: {total_edges}")
            print(f"  Training positive pairs: {len(pos_pairs)}")
            print(f"  Available negative pairs: {len(neg_pairs_all)}")
        
        num_val = int(len(pos_pairs) * val_split)
        train_pos = pos_pairs[num_val:]
        val_pos = pos_pairs[:num_val]
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_pos, neg_pairs_all, edges)
            self.train_losses.append(train_loss)
            val_loss = self.validate(val_pos, neg_pairs_all, edges)
            self.val_losses.append(val_loss)
            
            if self.enable_progress and epoch % 5 == 0:
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if self.enable_progress:
                        print(f"Early stopping at epoch {epoch}")
                    break
        
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}
    
    def validate(self, val_pos: torch.Tensor, neg_pairs_all: torch.Tensor, edges: Dict) -> float:
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        edges_device = self._move_edges_to_device(edges)
        
        with torch.no_grad():
            pos_per_batch = max(1, self.batch_size // (1 + self.neg_ratio))
            for start in range(0, val_pos.size(0), pos_per_batch):
                end = min(start + pos_per_batch, val_pos.size(0))
                batch_pos = val_pos[start:end]
                neg_indices = torch.randint(0, neg_pairs_all.size(0), (batch_pos.size(0) * self.neg_ratio,))
                batch_neg = neg_pairs_all[neg_indices]
                batch_all = torch.cat([batch_pos, batch_neg], dim=0)
                lnc_idx_batch = batch_all[:, 0].to(self.device)
                dis_idx_batch = batch_all[:, 1].to(self.device)
                labels = torch.cat([
                    torch.ones(batch_pos.size(0)),
                    torch.zeros(batch_neg.size(0))
                ], dim=0).to(self.device)
                logits = self.model(lnc_idx_batch, dis_idx_batch, edges_device)
                loss = self.criterion(logits, labels)
                val_loss += loss.item()
                num_batches += 1
        return val_loss / max(num_batches, 1)
    
    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)) 