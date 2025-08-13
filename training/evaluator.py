"""
Evaluation utilities for HGAT-LDA model.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import time

from models.hgat_lda import HGAT_LDA
from training.trainer import HGATLDATrainer
from data.graph_construction import get_positive_pairs, generate_negative_pairs, rewrite_ld_for_isolated_diseases, construct_heterogeneous_graph
from utils.scoring import canonical_affinity, require_sign


class HGATLDAEvaluator:
    """
    Evaluator class for HGAT-LDA model.
    """
    
    def __init__(self, model: HGAT_LDA, device: torch.device, full_ranking: bool = True,
                 score_orientation: str = 'auto', score_sign: Optional[float] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: HGAT-LDA model
            device: Device to evaluate on
            full_ranking: Whether to use full ranking (all negatives) during evaluation
            score_orientation: Score orientation ('affinity', 'distance', or 'auto')
            score_sign: Sign multiplier from calibration (for auto mode)
        """
        self.model = model.to(device)
        self.device = device
        self.full_ranking = full_ranking
        self.score_orientation = score_orientation
        self.score_sign = score_sign
    
    def score_all_lnc(self,
                      model: torch.nn.Module,
                      d_idx: int,
                      edges: Dict,
                      num_lncRNAs: int,
                      batch_size: int = 100) -> torch.Tensor:
        """
        Score all lncRNAs for a given disease (vectorized).
        
        Args:
            model: Trained model
            d_idx: Disease index
            edges: Graph edges
            num_lncRNAs: Total number of lncRNAs
            batch_size: Batch size for scoring
            
        Returns:
            Logit scores for all lncRNAs (not sigmoid)
        """
        model.eval()
        all_scores = []
        
        with torch.no_grad():
            # Score in batches to avoid OOM
            for start in range(0, num_lncRNAs, batch_size):
                end = min(start + batch_size, num_lncRNAs)
                batch_size_actual = end - start
                
                # Create batch
                lnc_batch = torch.arange(start, end, device=self.device)
                dis_batch = torch.full((batch_size_actual,), d_idx, 
                                      dtype=torch.long, device=self.device)
                
                # Score batch (raw logits)
                scores = model(lnc_batch, dis_batch, edges)
                all_scores.append(scores)
        
        return torch.cat(all_scores)
    
    def rank_all_for_disease(self, 
                            disease_idx: int,
                            edges: Dict,
                            known_positives: set = None,
                            num_lncRNAs: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scores for all lncRNAs against a specific disease.
        
        Args:
            disease_idx: Index of the disease to rank lncRNAs for
            edges: Graph edges dictionary
            known_positives: Set of known positive lncRNA indices for this disease
            num_lncRNAs: Total number of lncRNAs (if None, inferred from model)
            
        Returns:
            Tuple of (scores, labels) arrays for all lncRNAs
        """
        self.model.eval()
        
        if num_lncRNAs is None:
            num_lncRNAs = self.model.lnc_embed.num_embeddings
        
        if known_positives is None:
            known_positives = set()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            # Batch processing to avoid OOM
            batch_size = 256
            for start_idx in range(0, num_lncRNAs, batch_size):
                end_idx = min(start_idx + batch_size, num_lncRNAs)
                batch_lncs = torch.arange(start_idx, end_idx, device=self.device)
                batch_dis = torch.full_like(batch_lncs, disease_idx)
                
                # Get scores for this batch
                batch_logits = self.model(batch_lncs, batch_dis, edges)
                batch_scores = torch.sigmoid(batch_logits).cpu().numpy()
                
                # Create labels
                for i, lnc_idx in enumerate(range(start_idx, end_idx)):
                    all_scores.append(batch_scores[i])
                    all_labels.append(1 if lnc_idx in known_positives else 0)
        
        return np.array(all_scores), np.array(all_labels)
    
    def rank_all_for_lncRNA(self,
                           lnc_idx: int,
                           edges: Dict,
                           known_positives: set = None,
                           num_diseases: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scores for all diseases against a specific lncRNA.
        
        Args:
            lnc_idx: Index of the lncRNA to rank diseases for
            edges: Graph edges dictionary
            known_positives: Set of known positive disease indices for this lncRNA
            num_diseases: Total number of diseases (if None, inferred from model)
            
        Returns:
            Tuple of (scores, labels) arrays for all diseases
        """
        self.model.eval()
        
        if num_diseases is None:
            num_diseases = self.model.disease_embed.num_embeddings
        
        if known_positives is None:
            known_positives = set()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            # Batch processing to avoid OOM
            batch_size = 256
            for start_idx in range(0, num_diseases, batch_size):
                end_idx = min(start_idx + batch_size, num_diseases)
                batch_dis = torch.arange(start_idx, end_idx, device=self.device)
                batch_lncs = torch.full_like(batch_dis, lnc_idx)
                
                # Get scores for this batch
                batch_logits = self.model(batch_lncs, batch_dis, edges)
                batch_scores = torch.sigmoid(batch_logits).cpu().numpy()
                
                # Create labels
                for i, dis_idx in enumerate(range(start_idx, end_idx)):
                    all_scores.append(batch_scores[i])
                    all_labels.append(1 if dis_idx in known_positives else 0)
        
        return np.array(all_scores), np.array(all_labels)
    
    def evaluate_auc(self, 
                    pos_pairs: torch.Tensor,
                    neg_pairs: torch.Tensor,
                    edges: Dict,
                    num_lncRNAs: int = None,
                    num_diseases: int = None) -> float:
        """
        Evaluate AUC score with optional full ranking.
        
        Args:
            pos_pairs: Positive pairs tensor
            neg_pairs: Negative pairs tensor (ignored if full_ranking=True)
            edges: Graph edges dictionary
            num_lncRNAs: Number of lncRNAs (required for full ranking)
            num_diseases: Number of diseases (required for full ranking)
            
        Returns:
            AUC score
        """
        self.model.eval()
        
        if self.full_ranking and num_lncRNAs is not None and num_diseases is not None:
            # Use full ranking evaluation
            print("  Using full ranking for evaluation (neg_sampling=False)")
            all_scores = []
            all_labels = []
            
            # Create mapping of positive pairs
            pos_set = set((int(l), int(d)) for l, d in pos_pairs)
            
            # Evaluate each disease with all lncRNAs
            for dis_idx in range(num_diseases):
                # Get positive lncRNAs for this disease
                pos_lncs = {l for l, d in pos_set if d == dis_idx}
                
                if len(pos_lncs) > 0:  # Only evaluate if disease has positives
                    scores, labels = self.rank_all_for_disease(dis_idx, edges, pos_lncs, num_lncRNAs)
                    all_scores.extend(scores)
                    all_labels.extend(labels)
            
            if len(all_scores) > 0 and sum(all_labels) > 0:
                auc = roc_auc_score(all_labels, all_scores)
            else:
                auc = 0.5  # Default if no valid pairs
        else:
            # Use traditional negative sampling evaluation
            with torch.no_grad():
                # Get predictions for positive pairs
                pos_lnc = pos_pairs[:, 0].to(self.device)
                pos_dis = pos_pairs[:, 1].to(self.device)
                pos_logits = self.model(pos_lnc, pos_dis, edges).cpu().numpy()
                
                # Get predictions for negative pairs
                neg_lnc = neg_pairs[:, 0].to(self.device)
                neg_dis = neg_pairs[:, 1].to(self.device)
                neg_logits = self.model(neg_lnc, neg_dis, edges).cpu().numpy()
                
                # Combine scores and labels
                scores = 1 / (1 + np.exp(-(np.concatenate([pos_logits, neg_logits]))))
                labels = np.concatenate([np.ones(len(pos_logits)), np.zeros(len(neg_logits))])
                
                # Calculate AUC
                auc = roc_auc_score(labels, scores)
            
        return auc
    
    def evaluate_aupr(self, 
                     pos_pairs: torch.Tensor,
                     neg_pairs: torch.Tensor,
                     edges: Dict,
                     num_lncRNAs: int = None,
                     num_diseases: int = None) -> float:
        """
        Evaluate AUPR (Average Precision) score with optional full ranking.
        
        Args:
            pos_pairs: Positive pairs tensor
            neg_pairs: Negative pairs tensor (ignored if full_ranking=True)
            edges: Graph edges dictionary
            num_lncRNAs: Number of lncRNAs (required for full ranking)
            num_diseases: Number of diseases (required for full ranking)
            
        Returns:
            AUPR score
        """
        self.model.eval()
        
        if self.full_ranking and num_lncRNAs is not None and num_diseases is not None:
            # Use full ranking evaluation
            all_scores = []
            all_labels = []
            
            # Create mapping of positive pairs
            pos_set = set((int(l), int(d)) for l, d in pos_pairs)
            
            # Evaluate each disease with all lncRNAs
            for dis_idx in range(num_diseases):
                # Get positive lncRNAs for this disease
                pos_lncs = {l for l, d in pos_set if d == dis_idx}
                
                if len(pos_lncs) > 0:  # Only evaluate if disease has positives
                    scores, labels = self.rank_all_for_disease(dis_idx, edges, pos_lncs, num_lncRNAs)
                    all_scores.extend(scores)
                    all_labels.extend(labels)
            
            if len(all_scores) > 0 and sum(all_labels) > 0:
                aupr = average_precision_score(all_labels, all_scores)
            else:
                aupr = 0.0  # Default if no valid pairs
        else:
            # Use traditional negative sampling evaluation
            with torch.no_grad():
                # Get predictions for positive pairs
                pos_lnc = pos_pairs[:, 0].to(self.device)
                pos_dis = pos_pairs[:, 1].to(self.device)
                pos_logits = self.model(pos_lnc, pos_dis, edges).cpu().numpy()
                
                # Get predictions for negative pairs
                neg_lnc = neg_pairs[:, 0].to(self.device)
                neg_dis = neg_pairs[:, 1].to(self.device)
                neg_logits = self.model(neg_lnc, neg_dis, edges).cpu().numpy()
                
                # Combine scores and labels
                scores = 1 / (1 + np.exp(-(np.concatenate([pos_logits, neg_logits]))))
                labels = np.concatenate([np.ones(len(pos_logits)), np.zeros(len(neg_logits))])
                
                # Calculate AUPR
                aupr = average_precision_score(labels, scores)
            
        return aupr
    
    def leave_one_out_cross_validation(self,
                                     pos_pairs: torch.Tensor,
                                     edges: Dict,
                                     num_lncRNAs: int,
                                     num_diseases: int,
                                     num_epochs: int = 30,
                                     batch_size: int = 256,
                                     lr: float = 1e-3,
                                     neg_ratio: int = 1,
                                     weight_decay: float = 1e-5,
                                     use_focal_loss: bool = True,
                                     label_smoothing: float = 0.1,
                                     cosine_tmax: Optional[int] = None) -> List[float]:
        """
        Perform Leave-One-Out Cross-Validation with multi-GPU optimization.
        
        Args:
            pos_pairs: Positive pairs tensor
            edges: Graph edges dictionary
            num_lncRNAs: Number of lncRNAs
            num_diseases: Number of diseases
            num_epochs: Number of training epochs per fold
            batch_size: Batch size for training
            lr: Learning rate
            neg_ratio: Negative sampling ratio
            
        Returns:
            List of AUC scores for each fold
        """
        import concurrent.futures
        from functools import partial
        
        auc_scores = []
        num_folds = len(pos_pairs)
        
        # Detect available GPUs for parallel processing
        n_gpus = torch.cuda.device_count()
        n_workers = min(n_gpus * 2, 8) if n_gpus > 1 else 1  # Use 2 workers per GPU
        
        print(f"Starting LOOCV with {num_folds} folds...")
        print(f"Training {num_epochs} epochs per fold")
        print(f"Using {n_workers} parallel workers for fold processing")
        
        for fold_idx in tqdm(range(num_folds), desc="LOOCV Progress", position=0, leave=True):
            # Get test pair
            test_lnc = int(pos_pairs[fold_idx, 0].item())
            test_dis = int(pos_pairs[fold_idx, 1].item())
            
            # Remove test pair from training data
            train_pos_pairs = torch.cat([pos_pairs[:fold_idx], pos_pairs[fold_idx+1:]], dim=0)
            
            # Create a fresh model for this fold using same architecture
            model_fold = HGAT_LDA(
                num_lncRNAs=num_lncRNAs,
                num_genes=self.model.gene_embed.num_embeddings,
                num_diseases=num_diseases,
                edges=edges,
                emb_dim=self.model.emb_dim,
                num_layers=self.model.num_layers,
                dropout=self.model.dropout,
                num_heads=self.model.num_heads if hasattr(self.model, 'num_heads') else 4,
                relation_dropout=self.model.relation_dropout if hasattr(self.model, 'relation_dropout') else 0.1,
                use_layernorm=self.model.use_layernorm if hasattr(self.model, 'use_layernorm') else True,
                use_residual=self.model.use_residual if hasattr(self.model, 'use_residual') else True,
                use_relation_norm=self.model.use_relation_norm if hasattr(self.model, 'use_relation_norm') else True
            ).to(self.device)
            
            # Create trainer using parameters from config
            trainer = HGATLDATrainer(
                model=model_fold,
                device=self.device,
                lr=float(lr),
                weight_decay=float(weight_decay),
                batch_size=int(batch_size),
                enable_progress=False,
                neg_ratio=int(neg_ratio),
                use_amp=torch.cuda.is_available(),  # Enable AMP for RTX 5090
                use_focal_loss=use_focal_loss,  # Use focal loss from config
                label_smoothing=label_smoothing,  # Use label smoothing from config
                cosine_tmax=cosine_tmax if cosine_tmax else num_epochs,  # Use cosine annealing from config
                use_multi_gpu=True  # Enable multi-GPU for each fold
            )
            
            # Generate negative pairs for this fold
            neg_pairs_all = generate_negative_pairs(train_pos_pairs, num_lncRNAs, num_diseases)
            
            # Train the model (suppress output for LOOCV)
            import sys
            import os
            # Suppress training output during LOOCV
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                trainer.train(
                    pos_pairs=train_pos_pairs,
                    neg_pairs_all=neg_pairs_all,
                    edges=edges,
                    num_epochs=num_epochs,
                    val_split=0.1,
                    early_stopping_patience=5,
                    save_path=None  # Don't save models during LOOCV
                )
            finally:
                sys.stdout = old_stdout
            
            # Evaluate on test pair
            model_fold.eval()
            with torch.no_grad():
                # Score for the positive pair
                pos_score = torch.sigmoid(model_fold(
                    torch.tensor([test_lnc], device=self.device),
                    torch.tensor([test_dis], device=self.device),
                    edges
                )).item()
                
                # Generate negative pairs for evaluation
                neg_scores = []
                neg_labels = []
                
                # All diseases not associated with test_lnc
                known = set((int(i), int(j)) for i, j in train_pos_pairs)
                for d in range(num_diseases):
                    if (test_lnc, d) not in known:
                        score = torch.sigmoid(model_fold(
                            torch.tensor([test_lnc], device=self.device),
                            torch.tensor([d], device=self.device),
                            edges
                        )).item()
                        neg_scores.append(score)
                        neg_labels.append(0)
                
                # All lncRNAs not associated with test_dis
                for l in range(num_lncRNAs):
                    if (l, test_dis) not in known:
                        score = torch.sigmoid(model_fold(
                            torch.tensor([l], device=self.device),
                            torch.tensor([test_dis], device=self.device),
                            edges
                        )).item()
                        neg_scores.append(score)
                        neg_labels.append(0)
                
                # Combine positive and negative scores
                y_true = [1] + neg_labels
                y_score = [pos_score] + neg_scores
                
                # Calculate AUC
                auc = roc_auc_score(y_true, y_score)
                auc_scores.append(auc)
                
                # Print progress every 10 folds
                if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
                    mean_auc_so_far = np.mean(auc_scores)
                    print(f"\nFold {fold_idx+1}/{num_folds}: Current AUC = {auc:.3f}, Mean AUC so far = {mean_auc_so_far:.3f}")
        
        return auc_scores
    
    def run_loocv_with_rewrite(self,
                              pos_pairs: torch.Tensor,
                              edges: Dict,
                              data_dict: Dict[str, torch.Tensor],
                              num_lncRNAs: int,
                              num_diseases: int,
                              num_epochs: int = 30,
                              batch_size: int = 256,
                              lr: float = 1e-3,
                              neg_ratio: int = 1,
                              weight_decay: float = 1e-5,
                              use_focal_loss: bool = True,
                              label_smoothing: float = 0.1,
                              cosine_tmax: Optional[int] = None,
                              rewrite_isolated: bool = True,
                              sim_topk: Optional[int] = None,
                              sim_row_normalize: bool = True,
                              sim_threshold: float = 0.0,
                              full_ranking: bool = True) -> Dict[str, Any]:
        """
        Perform Leave-One-Out Cross-Validation per positive (lncRNA, disease) pair
        with disease rewrite when isolated.
        
        This implements LOOCV where each positive pair is held out one at a time.
        When a disease becomes isolated (no remaining links after holdout), its
        connections are rewritten using disease similarity information.
        
        Args:
            pos_pairs: Positive pairs tensor (lncRNA, disease pairs)
            edges: Graph edges dictionary
            data_dict: Dictionary containing all data matrices including similarities
            num_lncRNAs: Number of lncRNAs
            num_diseases: Number of diseases
            num_epochs: Number of training epochs per fold
            batch_size: Batch size for training
            lr: Learning rate
            neg_ratio: Negative sampling ratio
            weight_decay: Weight decay for optimizer
            use_focal_loss: Whether to use focal loss
            label_smoothing: Label smoothing factor
            cosine_tmax: T_max for cosine annealing scheduler
            rewrite_isolated: Whether to rewrite isolated diseases
            sim_topk: Top-k for similarity pruning
            sim_row_normalize: Whether to normalize similarity rows
            sim_threshold: Threshold for similarity edges
            full_ranking: Whether to use full ranking (evaluate all lncRNAs) during evaluation
            
        Returns:
            Dictionary containing:
                - 'fold_scores': List of per-fold evaluation scores (ROC-AUC, AUPR, F1-max)
                - 'macro_averages': Macro-averaged scores across all folds
                - 'num_folds': Total number of folds
                - 'num_rewrites': Number of times disease rewrite was applied
                - 'rewritten_diseases': List of disease indices that were rewritten
        """
        import sys
        import os
        from sklearn.metrics import f1_score
        
        # Initialize tracking variables
        fold_results = []
        rewritten_diseases = []
        num_rewrites = 0
        num_folds = len(pos_pairs)
        
        print(f"\n{'='*70}")
        print(f"Starting LOOCV with rewrite for {num_folds} positive pairs")
        print(f"Evaluation settings: neg_sampling=False, full_ranking={full_ranking}")
        print(f"Scoring: orientation={self.score_orientation}, score_sign=will be calibrated per fold")
        print(f"Disease rewrite enabled: {rewrite_isolated}")
        print(f"Training {num_epochs} epochs per fold (neg_ratio={neg_ratio} for training only)")
        print(f"{'='*70}")
        print(f"Progress: Each fold shows current & running mean metrics")
        print(f"Legend: [R] = disease was rewritten due to isolation")
        print(f"{'='*70}\n")
        
        # Import build_fold_edges
        from data.graph_construction import build_fold_edges
        
        # Create config for build_fold_edges
        fold_config = {
            'eval': {'rewrite_isolated': rewrite_isolated},
            'data': {
                'sim_topk': sim_topk,
                'sim_row_normalize': sim_row_normalize,
                'threshold': sim_threshold,
                'sim_mutual': False,
                'sim_sym': True,
                'sim_row_norm': sim_row_normalize,
                'sim_tau': None,
                'use_bipartite_edge_weight': False
            }
        }
        
        for fold_idx in tqdm(range(num_folds), desc="LOOCV Progress", position=0, leave=True):
            # Get the held-out pair
            test_lnc = int(pos_pairs[fold_idx, 0].item())
            test_dis = int(pos_pairs[fold_idx, 1].item())
            
            # Build per-fold edges (removes test edge, handles isolation)
            fold_edges, removed_edge_present, disease_was_rewritten, ld_edge_count = build_fold_edges(
                data_dict, fold_config, test_lnc, test_dis
            )
            
            if disease_was_rewritten:
                num_rewrites += 1
                rewritten_diseases.append(test_dis)
            
            # Create training pairs (all except the held-out pair)
            train_pos_pairs = torch.cat([pos_pairs[:fold_idx], pos_pairs[fold_idx+1:]], dim=0)
            
            # Move edges to device
            fold_edges_device = {}
            for key, (src, dst, w) in fold_edges.items():
                if src is not None:
                    src = src.to(self.device)
                    dst = dst.to(self.device)
                    if w is not None:
                        w = w.to(self.device)
                fold_edges_device[key] = (src, dst, w)
            
            # Create a fresh model for this fold
            model_fold = HGAT_LDA(
                num_lncRNAs=num_lncRNAs,
                num_genes=self.model.gene_embed.num_embeddings,
                num_diseases=num_diseases,
                edges=fold_edges_device,
                emb_dim=self.model.emb_dim,
                num_layers=self.model.num_layers,
                dropout=self.model.dropout,
                num_heads=self.model.num_heads if hasattr(self.model, 'num_heads') else 4,
                relation_dropout=self.model.relation_dropout if hasattr(self.model, 'relation_dropout') else 0.1,
                use_layernorm=self.model.use_layernorm if hasattr(self.model, 'use_layernorm') else True,
                use_residual=self.model.use_residual if hasattr(self.model, 'use_residual') else True
            ).to(self.device)
            
            # Create trainer for this fold
            trainer = HGATLDATrainer(
                model=model_fold,
                device=self.device,
                lr=float(lr),
                weight_decay=float(weight_decay),
                batch_size=int(batch_size),
                enable_progress=False,
                neg_ratio=int(neg_ratio),
                use_amp=torch.cuda.is_available(),
                use_focal_loss=use_focal_loss,
                label_smoothing=label_smoothing,
                cosine_tmax=cosine_tmax if cosine_tmax else num_epochs,
                use_multi_gpu=True,
                score_orientation=self.score_orientation
            )
            
            # Generate negative pairs for training
            neg_pairs_all = generate_negative_pairs(train_pos_pairs, num_lncRNAs, num_diseases)
            
            # Train the model (suppress most output but keep calibration)
            trainer.train(
                pos_pairs=train_pos_pairs,
                neg_pairs_all=neg_pairs_all,
                edges=fold_edges_device,
                num_epochs=num_epochs,
                val_split=0.1,
                early_stopping_patience=5,
                save_path=None
            )
            
            # Get calibrated score_sign from trainer
            fold_score_sign = trainer.get_score_sign()
            if fold_score_sign is not None:
                self.score_sign = fold_score_sign
            
            # Full-ranking evaluation with proper masking
            model_fold.eval()
            
            # Score all lncRNAs for the test disease (vectorized)
            raw_scores = self.score_all_lnc(model_fold, test_dis, fold_edges_device, num_lncRNAs)  # [L]
            
            # Convert to canonical affinity scores (higher = better)
            require_sign(self.score_orientation, self.score_sign)
            aff_scores = canonical_affinity(raw_scores, self.score_orientation, self.score_sign)
            
            # Create mask for known positives (from original data, not just training)
            known_pos = data_dict['lnc_disease_assoc'][test_dis].bool()  # [L]
            
            # Create labels: only test_lnc is positive
            y = torch.zeros_like(aff_scores, dtype=torch.long)
            y[test_lnc] = 1
            
            # Create valid mask: exclude other known positives but include test_lnc
            valid = ~known_pos
            valid[test_lnc] = True
            
            # Extract valid scores and labels for evaluation
            s = aff_scores[valid].detach().cpu().numpy()
            lbl = y[valid].cpu().numpy()
            
            # Calculate metrics (NO per-fold flipping!)
            if len(s) > 1 and lbl.sum() > 0:  # Ensure we have both pos and neg
                # Calculate AUC with canonical affinity scores
                fold_auc = roc_auc_score(lbl, s)
                
                # Check for inversion as WARNING ONLY
                fold_auc_inv = roc_auc_score(lbl, -s)
                if fold_auc_inv > fold_auc + 1e-4:
                    print(f"  ⚠️ Fold {fold_idx+1}: inversion warning - auc_inv={fold_auc_inv:.3f} > auc={fold_auc:.3f}. Check score plumbing.")
                    # DO NOT flip scores!
                
                fold_aupr = average_precision_score(lbl, s)
                
                # Calculate F1-max
                precision, recall, thresholds = precision_recall_curve(lbl, s)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                fold_f1_max = np.max(f1_scores[np.isfinite(f1_scores)])
                
                # Calculate rank of true lncRNA using affinity scores (higher = better)
                true_score = aff_scores[test_lnc].item()
                rank = (aff_scores > true_score).sum().item() + 1
                
                fold_results.append({
                    'fold': fold_idx + 1,
                    'test_lnc': test_lnc,
                    'test_disease': test_dis,
                    'disease_was_rewritten': disease_was_rewritten,
                    'auc': fold_auc,
                    'aupr': fold_aupr,
                    'f1_max': fold_f1_max
                })
                
                # Print progress after EACH fold
                mean_auc = np.mean([r['auc'] for r in fold_results])
                mean_aupr = np.mean([r['aupr'] for r in fold_results])
                mean_f1 = np.mean([r['f1_max'] for r in fold_results])
                
                # Compact single-line output for each fold with diagnostics
                sign_str = '+1' if self.score_sign and self.score_sign > 0 else '-1' if self.score_sign else 'None'
                print(f"Fold {fold_idx+1:4d}/{num_folds}: "
                      f"AUC={fold_auc:.3f} (mean={mean_auc:.3f}), "
                      f"rank={rank:3d}, "
                      f"sign={sign_str}, "
                      f"edge={removed_edge_present}, "
                      f"ld_edges={ld_edge_count}/855 "
                      f"{'[R]' if disease_was_rewritten else ''}")
                
                # Print summary every 100 folds
                if (fold_idx + 1) % 100 == 0:
                    print(f"\n{'─'*60}")
                    print(f"Summary at fold {fold_idx+1}:")
                    print(f"  Mean AUC: {mean_auc:.4f}, AUPR: {mean_aupr:.4f}, F1: {mean_f1:.4f}")
                    print(f"  Rewrites so far: {num_rewrites}/{fold_idx+1} ({num_rewrites/(fold_idx+1)*100:.1f}%)")
                    print(f"{'─'*60}\n")
        
        # Calculate macro averages
        macro_averages = {
            'auc': np.mean([r['auc'] for r in fold_results]),
            'aupr': np.mean([r['aupr'] for r in fold_results]),
            'f1_max': np.mean([r['f1_max'] for r in fold_results]),
            'auc_std': np.std([r['auc'] for r in fold_results]),
            'aupr_std': np.std([r['aupr'] for r in fold_results]),
            'f1_max_std': np.std([r['f1_max'] for r in fold_results])
        }
        
        print(f"\n{'='*60}")
        print(f"LOOCV completed with {num_folds} folds")
        print(f"Disease rewrites applied: {num_rewrites}")
        print(f"Unique diseases rewritten: {len(set(rewritten_diseases))}")
        print(f"\nMacro-averaged results:")
        print(f"  ROC-AUC: {macro_averages['auc']:.4f} ± {macro_averages['auc_std']:.4f}")
        print(f"  AUPR:    {macro_averages['aupr']:.4f} ± {macro_averages['aupr_std']:.4f}")
        print(f"  F1-max:  {macro_averages['f1_max']:.4f} ± {macro_averages['f1_max_std']:.4f}")
        print(f"{'='*60}")
        
        return {
            'fold_scores': fold_results,
            'macro_averages': macro_averages,
            'num_folds': num_folds,
            'num_rewrites': num_rewrites,
            'rewritten_diseases': rewritten_diseases
        }
    
    def evaluate_all_metrics(self,
                           pos_pairs: torch.Tensor,
                           neg_pairs: torch.Tensor,
                           edges: Dict,
                           num_lncRNAs: int = None,
                           num_diseases: int = None) -> Dict[str, float]:
        """
        Evaluate all metrics for given pairs with optional full ranking.
        
        Args:
            pos_pairs: Positive pairs tensor
            neg_pairs: Negative pairs tensor (ignored if full_ranking=True)
            edges: Graph edges dictionary
            num_lncRNAs: Number of lncRNAs (required for full ranking)
            num_diseases: Number of diseases (required for full ranking)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        self.model.eval()
        
        if self.full_ranking and num_lncRNAs is not None and num_diseases is not None:
            # Use full ranking evaluation
            print("  Using full ranking for all metrics (neg_sampling=False)")
            all_scores = []
            all_labels = []
            
            # Create mapping of positive pairs
            pos_set = set((int(l), int(d)) for l, d in pos_pairs)
            
            # Evaluate each disease with all lncRNAs
            for dis_idx in range(num_diseases):
                # Get positive lncRNAs for this disease
                pos_lncs = {l for l, d in pos_set if d == dis_idx}
                
                if len(pos_lncs) > 0:  # Only evaluate if disease has positives
                    scores, labels = self.rank_all_for_disease(dis_idx, edges, pos_lncs, num_lncRNAs)
                    all_scores.extend(scores)
                    all_labels.extend(labels)
            
            scores = np.array(all_scores)
            labels = np.array(all_labels)
        else:
            # Use traditional negative sampling evaluation
            with torch.no_grad():
                # Get predictions for positive pairs
                pos_lnc = pos_pairs[:, 0].to(self.device)
                pos_dis = pos_pairs[:, 1].to(self.device)
                pos_logits = self.model(pos_lnc, pos_dis, edges).cpu().numpy()
                
                # Get predictions for negative pairs
                neg_lnc = neg_pairs[:, 0].to(self.device)
                neg_dis = neg_pairs[:, 1].to(self.device)
                neg_logits = self.model(neg_lnc, neg_dis, edges).cpu().numpy()
                
                # Combine scores and labels
                scores = 1 / (1 + np.exp(-(np.concatenate([pos_logits, neg_logits]))))
                labels = np.concatenate([np.ones(len(pos_logits)), np.zeros(len(neg_logits))])
        
        # Calculate metrics
        predictions = (scores > 0.5).astype(int)
        
        # Calculate metrics using the utility function
        from utils.metrics import calculate_metrics
        metrics = calculate_metrics(labels, predictions, scores)
        
        # Calculate additional metrics
        auc = roc_auc_score(labels, scores)
        aupr = average_precision_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_max = np.max(f1_scores[np.isfinite(f1_scores)]) if np.any(np.isfinite(f1_scores)) else 0.0
        
        return {'auc': auc, 'aupr': aupr, 'f1_max': f1_max, 'precision': precision, 'recall': recall} 