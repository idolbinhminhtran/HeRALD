"""
Training loop implementation for HGAT-LDA model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, Optional, List
import numpy as np
from tqdm import tqdm
import os

from models.hgat_lda import HGAT_LDA
from data.graph_construction import get_positive_pairs, generate_negative_pairs
from models.losses import get_loss_function, compute_hard_negatives, FocalLoss as FocalLossNew
from utils.scoring import canonical_affinity, calibrate_score_sign, require_sign


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
                 use_multi_gpu: bool = True,
                 full_ranking_eval: bool = True,
                 loss_type: str = 'bce',
                 pairwise_type: str = 'bpr',
                 use_hard_negatives: bool = True,
                 score_orientation: str = 'auto'):
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
            full_ranking_eval: Whether to use full ranking for validation (evaluates all negatives)
            loss_type: Type of loss ('bce', 'focal', 'pairwise')
            pairwise_type: Type of pairwise loss ('bpr' or 'auc') when loss_type='pairwise'
            use_hard_negatives: Whether to use hard negative mining for pairwise losses
            score_orientation: Score orientation ('affinity', 'distance', or 'auto')
        """
        # Move model to device first
        self.model = model.to(device)
        self.device = device
        
        # Enable multi-GPU if available and requested
        self.use_multi_gpu = use_multi_gpu
        self.batch_size = batch_size
        if use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"🚀 Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            # DataParallel automatically splits batch across GPUs
            print(f"   Batch size: {self.batch_size} total ({self.batch_size // torch.cuda.device_count()} per GPU)")
        self.enable_progress = enable_progress
        self.neg_ratio = max(1, neg_ratio)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.label_smoothing = label_smoothing
        self.full_ranking_eval = full_ranking_eval
        self.loss_type = loss_type.lower()
        self.pairwise_type = pairwise_type.lower()
        self.use_hard_negatives = use_hard_negatives
        self.score_orientation = score_orientation
        self.score_sign = None  # Will be calibrated on first batch if orientation='auto'
        self.calibrated = False
        
        # Optimizer with improved settings
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        
        # Loss function configuration based on loss_type
        if self.loss_type == 'pairwise':
            # Use pairwise ranking loss
            self.criterion = get_loss_function('pairwise', pairwise_type=self.pairwise_type)
            self.is_pairwise = True
            if enable_progress:
                print(f"   Using pairwise {self.pairwise_type.upper()} loss with {'hard' if use_hard_negatives else 'random'} negatives")
        elif self.loss_type == 'focal':
            self.criterion = get_loss_function('focal', alpha=0.25, gamma=2.0)
            self.is_pairwise = False
        elif self.loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
            self.is_pairwise = False
        else:
            # Backward compatibility: use_focal_loss parameter
            if use_focal_loss:
                self.criterion = get_loss_function('focal', alpha=0.25, gamma=2.0)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
            self.is_pairwise = False
            
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
    
    def calibrate_score_sign(self, pos_pairs: torch.Tensor, edges: Dict, batch_size: int = 256) -> None:
        """
        Calibrate score sign by sampling positives/negatives and checking raw score gap.
        
        Args:
            pos_pairs: All positive pairs for sampling
            edges: Graph edges
            batch_size: Sample size for calibration
        """
        if self.score_orientation != 'auto':
            # Fixed orientation, no calibration needed
            return
            
        if self.score_sign is not None:
            # Already calibrated
            return
        
        # Sample a small batch for calibration
        n_pos = min(batch_size, len(pos_pairs))
        sample_indices = torch.randperm(len(pos_pairs))[:n_pos]
        sample_pos = pos_pairs[sample_indices]
        
        pos_lnc = sample_pos[:, 0].to(self.device)
        pos_dis = sample_pos[:, 1].to(self.device)
        
        # Create a set of known positives for masking
        known_positives = set()
        for l, d in pos_pairs:
            known_positives.add((l.item(), d.item()))
        
        # For each positive, find a valid negative from same disease
        neg_lnc = []
        neg_dis = []
        
        for i, (l, d) in enumerate(zip(pos_lnc, pos_dis)):
            d_val = d.item()
            l_val = l.item()
            
            # Find candidate negative lncRNAs for this disease
            # Sample from all lncRNAs and filter out known positives
            max_lnc = max(pos_pairs[:, 0].max().item(), 500)  # Conservative estimate
            candidates = []
            attempts = 0
            
            while len(candidates) < 5 and attempts < 100:  # Try to find 5 candidates
                rand_lnc = torch.randint(0, max_lnc, (10,)).to(self.device)
                for rl in rand_lnc:
                    if (rl.item(), d_val) not in known_positives and rl.item() != l_val:
                        candidates.append(rl.item())
                        if len(candidates) >= 5:
                            break
                attempts += 1
            
            if candidates:
                # Pick first candidate as negative
                neg_lnc.append(candidates[0])
                neg_dis.append(d_val)
        
        if len(neg_lnc) == 0:
            # Fallback: use random negatives
            self.score_sign = 1.0
            print("[Calib] No valid negatives found, defaulting to score_sign=+1")
            return
        
        # Convert to tensors
        neg_lnc = torch.tensor(neg_lnc, device=self.device)
        neg_dis = torch.tensor(neg_dis, device=self.device)
        
        # Truncate positives to match negatives
        pos_lnc = pos_lnc[:len(neg_lnc)]
        pos_dis = pos_dis[:len(neg_dis)]
        
        # Move edges to device
        edges_device = self._move_edges_to_device(edges)
        
        # Compute raw scores (no canonicalization)
        self.model.eval()
        with torch.no_grad():
            raw_pos = self.model(pos_lnc, pos_dis, edges_device)
            raw_neg = self.model(neg_lnc, neg_dis, edges_device)
            
            # Compute mean gap
            gap = (raw_pos - raw_neg).mean().item()
            
            # Set sign based on gap
            self.score_sign = 1.0 if gap >= 0 else -1.0
            
            print(f"[Calib] score_orientation={self.score_orientation}, "
                  f"score_sign={'+1' if self.score_sign > 0 else '-1'}, "
                  f"mean_gap={gap:.4f}")
        
        self.model.train()
    
    def get_score_sign(self) -> Optional[float]:
        """Get the calibrated score sign."""
        return self.score_sign
    
    def train_epoch(self, pos_pairs: torch.Tensor, neg_pairs_all: torch.Tensor, edges: Dict, 
                   train_pos_pairs: torch.Tensor = None) -> Tuple[float, Optional[Dict]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        pos_scores_list = []
        neg_scores_list = []
        
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
            
            # Get positive indices
            pos_lnc = batch_pos[:, 0].to(self.device)
            pos_dis = batch_pos[:, 1].to(self.device)
            
            if self.is_pairwise:
                # For pairwise losses, handle differently
                
                # Sample or mine hard negatives (MUST be from same disease)
                if self.use_hard_negatives and len(neg_pairs_all) > 0:
                    # Mine hard negatives within same disease only
                    neg_lnc_pool = neg_pairs_all[:, 0].to(self.device)
                    neg_dis_pool = neg_pairs_all[:, 1].to(self.device)
                    
                    # Build set of known positives for masking
                    known_positives = set()
                    for l, d in pos_pairs:
                        known_positives.add((l.item(), d.item()))
                    
                    # Get unique diseases in this batch
                    unique_dis = torch.unique(pos_dis)
                    hard_neg_lnc = []
                    hard_neg_dis = []
                    
                    with torch.no_grad():
                        self.model.eval()  # Set to eval mode for hard negative mining
                        for d in unique_dis:
                            # Find positives for this disease in batch
                            mask = pos_dis == d
                            if mask.sum() == 0:
                                continue
                            
                            # Get negative lncRNAs for THIS SPECIFIC disease only
                            neg_mask = neg_dis_pool == d
                            if neg_mask.sum() > 0:
                                neg_lncs_for_d = neg_lnc_pool[neg_mask]
                                
                                # Filter out any known positives
                                valid_negs = []
                                for lnc in neg_lncs_for_d:
                                    if (lnc.item(), d.item()) not in known_positives:
                                        valid_negs.append(lnc)
                                
                                if len(valid_negs) == 0:
                                    continue
                                neg_lncs_for_d = torch.stack(valid_negs)
                                
                                # Score all negative lncRNAs in batches to avoid BN issues
                                if len(neg_lncs_for_d) > 1:
                                    d_repeated = d.repeat(len(neg_lncs_for_d))
                                    neg_scores = self.model(neg_lncs_for_d, d_repeated, edges_device)
                                    
                                    # Convert to canonical affinity for hard mining (highest affinity = hardest)
                                    require_sign(self.score_orientation, self.score_sign)
                                    neg_affinity = canonical_affinity(neg_scores, self.score_orientation, self.score_sign)
                                    
                                    # Select hardest negatives (highest affinity scores)
                                    k = min(mask.sum().item(), len(neg_affinity))
                                    _, hard_idx = torch.topk(neg_affinity, k)
                                    
                                    hard_neg_lnc.append(neg_lncs_for_d[hard_idx])
                                    hard_neg_dis.append(d_repeated[hard_idx])
                        self.model.train()  # Set back to train mode
                    
                    if hard_neg_lnc:
                        neg_lnc = torch.cat(hard_neg_lnc)
                        neg_dis = torch.cat(hard_neg_dis)
                        # Ensure same number of positives and negatives for pairwise loss
                        if len(neg_lnc) < len(pos_lnc):
                            # Need more negatives, sample randomly
                            num_needed = len(pos_lnc) - len(neg_lnc)
                            extra_indices = torch.randint(0, neg_pairs_all.size(0), (num_needed,))
                            extra_neg = neg_pairs_all[extra_indices]
                            neg_lnc = torch.cat([neg_lnc, extra_neg[:, 0].to(self.device)])
                            neg_dis = torch.cat([neg_dis, extra_neg[:, 1].to(self.device)])
                        elif len(neg_lnc) > len(pos_lnc):
                            # Too many negatives, truncate to match positives
                            neg_lnc = neg_lnc[:len(pos_lnc)]
                            neg_dis = neg_dis[:len(pos_dis)]
                    else:
                        # Fallback to random sampling
                        neg_indices = torch.randint(0, neg_pairs_all.size(0), (len(batch_pos),))
                        batch_neg = neg_pairs_all[neg_indices]
                        neg_lnc = batch_neg[:, 0].to(self.device)
                        neg_dis = batch_neg[:, 1].to(self.device)
                else:
                    # Random negative sampling
                    neg_indices = torch.randint(0, neg_pairs_all.size(0), (len(batch_pos),))
                    batch_neg = neg_pairs_all[neg_indices]
                    neg_lnc = batch_neg[:, 0].to(self.device)
                    neg_dis = batch_neg[:, 1].to(self.device)
                
                # Compute scores for pairwise loss
                if self.use_amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        pos_scores = self.model(pos_lnc, pos_dis, edges_device)
                        neg_scores = self.model(neg_lnc, neg_dis, edges_device)
                        
                        # Convert to canonical affinity scores
                        require_sign(self.score_orientation, self.score_sign)
                        aff_pos = canonical_affinity(pos_scores, self.score_orientation, self.score_sign)
                        aff_neg = canonical_affinity(neg_scores, self.score_orientation, self.score_sign)
                        
                        # Logistic AUC surrogate loss: -log σ(s_pos - s_neg)
                        if self.pairwise_type == 'auc':
                            loss = nn.functional.softplus(-(aff_pos - aff_neg)).mean()
                        else:
                            # BPR or other pairwise losses
                            loss = self.criterion(aff_pos, aff_neg)
                else:
                    pos_scores = self.model(pos_lnc, pos_dis, edges_device)
                    neg_scores = self.model(neg_lnc, neg_dis, edges_device)
                    
                    # Convert to canonical affinity scores
                    require_sign(self.score_orientation, self.score_sign)
                    aff_pos = canonical_affinity(pos_scores, self.score_orientation, self.score_sign)
                    aff_neg = canonical_affinity(neg_scores, self.score_orientation, self.score_sign)
                    
                    # Logistic AUC surrogate loss: -log σ(s_pos - s_neg)
                    if self.pairwise_type == 'auc':
                        loss = nn.functional.softplus(-(aff_pos - aff_neg)).mean()
                    else:
                        # BPR or other pairwise losses
                        loss = self.criterion(aff_pos, aff_neg)
                
                # Track scores for metrics
                pos_scores_list.append(pos_scores.detach().mean().item())
                neg_scores_list.append(neg_scores.detach().mean().item())
                
            else:
                # Standard BCE/Focal loss
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
                
                # Track scores for metrics
                with torch.no_grad():
                    pos_logits = logits[:batch_pos.size(0)]
                    neg_logits = logits[batch_pos.size(0):]
                    pos_scores_list.append(pos_logits.mean().item())
                    neg_scores_list.append(neg_logits.mean().item())
            
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
        
        # Compute metrics
        metrics = None
        if pos_scores_list and neg_scores_list:
            metrics = {
                'pos_score_mean': np.mean(pos_scores_list),
                'neg_score_mean': np.mean(neg_scores_list),
                'score_gap': np.mean(pos_scores_list) - np.mean(neg_scores_list)
            }
        
        return epoch_loss / max(num_batches, 1), metrics
    
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
        
        # Calibrate score sign before training starts
        self.calibrate_score_sign(pos_pairs, edges)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = self.train_epoch(train_pos, neg_pairs_all, edges, train_pos)
            self.train_losses.append(train_loss)
            val_loss = self.validate(val_pos, neg_pairs_all, edges)
            self.val_losses.append(val_loss)
            
            if self.enable_progress and epoch % 5 == 0:
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                if train_metrics and self.is_pairwise:
                    print(f"  Pos Score: {train_metrics['pos_score_mean']:.4f}, "
                          f"Neg Score: {train_metrics['neg_score_mean']:.4f}, "
                          f"Gap: {train_metrics['score_gap']:.4f}")
            
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