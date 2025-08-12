"""
Evaluation utilities for HGAT-LDA model.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

from models.hgat_lda import HGAT_LDA
from training.trainer import HGATLDATrainer
from data.graph_construction import get_positive_pairs, generate_negative_pairs


class HGATLDAEvaluator:
    """
    Evaluator class for HGAT-LDA model.
    """
    
    def __init__(self, model: HGAT_LDA, device: torch.device):
        """
        Initialize the evaluator.
        
        Args:
            model: HGAT-LDA model
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.device = device
    
    def evaluate_auc(self, 
                    pos_pairs: torch.Tensor,
                    neg_pairs: torch.Tensor,
                    edges: Dict) -> float:
        """
        Evaluate AUC score for given positive and negative pairs.
        
        Args:
            pos_pairs: Positive pairs tensor
            neg_pairs: Negative pairs tensor
            edges: Graph edges dictionary
            
        Returns:
            AUC score
        """
        self.model.eval()
        
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
                     edges: Dict) -> float:
        """
        Evaluate AUPR (Average Precision) score.
        
        Args:
            pos_pairs: Positive pairs tensor
            neg_pairs: Negative pairs tensor
            edges: Graph edges dictionary
            
        Returns:
            AUPR score
        """
        self.model.eval()
        
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
                use_residual=self.model.use_residual if hasattr(self.model, 'use_residual') else True
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
    
    def evaluate_all_metrics(self,
                           pos_pairs: torch.Tensor,
                           neg_pairs: torch.Tensor,
                           edges: Dict) -> Dict[str, float]:
        """
        Evaluate all metrics for given pairs.
        
        Args:
            pos_pairs: Positive pairs tensor
            neg_pairs: Negative pairs tensor
            edges: Graph edges dictionary
            
        Returns:
            Dictionary with all evaluation metrics
        """
        self.model.eval()
        
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
            
            # Convert scores to binary predictions for confusion matrix
            predictions = (scores > 0.5).astype(int)
            
            # Calculate metrics using the utility function
            from utils.metrics import calculate_metrics
            metrics = calculate_metrics(labels, predictions, scores)
            
            # Calculate additional metrics
            auc = roc_auc_score(labels, scores)
            aupr = average_precision_score(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_max = np.max(f1_scores)
            
        return {'auc': auc, 'aupr': aupr, 'f1_max': f1_max, 'precision': precision, 'recall': recall} 