"""
Custom loss functions for HGAT-LDA model including pairwise ranking losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits from model
            targets: Binary labels
            
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)
        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking loss for pairwise ranking.
    """
    
    def __init__(self):
        super(BPRLoss, self).__init__()
    
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute BPR loss: -log(sigmoid(pos_scores - neg_scores))
        
        Args:
            pos_scores: Scores for positive pairs
            neg_scores: Scores for negative pairs
            
        Returns:
            BPR loss value
        """
        # BPR loss
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        return loss


class AUCSurrogateLoss(nn.Module):
    """
    AUC surrogate loss for optimizing AUC directly.
    Approximates AUC optimization with a differentiable surrogate.
    """
    
    def __init__(self, margin: float = 1.0):
        super(AUCSurrogateLoss, self).__init__()
        self.margin = margin
    
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute AUC surrogate loss using hinge loss approximation.
        
        Args:
            pos_scores: Scores for positive pairs
            neg_scores: Scores for negative pairs
            
        Returns:
            AUC surrogate loss value
        """
        # Hinge loss approximation for AUC
        # Loss = max(0, margin - (pos_scores - neg_scores))
        diff = pos_scores - neg_scores
        loss = torch.clamp(self.margin - diff, min=0).mean()
        return loss


class PairwiseRankingLoss(nn.Module):
    """
    Unified pairwise ranking loss supporting BPR and AUC surrogate.
    Includes hard-negative mining capability.
    """
    
    def __init__(self, loss_type: str = 'bpr', margin: float = 1.0):
        """
        Initialize pairwise ranking loss.
        
        Args:
            loss_type: Type of pairwise loss ('bpr' or 'auc')
            margin: Margin for AUC surrogate loss
        """
        super(PairwiseRankingLoss, self).__init__()
        self.loss_type = loss_type.lower()
        
        if self.loss_type == 'bpr':
            self.loss_fn = BPRLoss()
        elif self.loss_type == 'auc':
            self.loss_fn = AUCSurrogateLoss(margin=margin)
        else:
            raise ValueError(f"Unknown pairwise loss type: {loss_type}")
    
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise ranking loss.
        
        Args:
            pos_scores: Scores for positive pairs (logits)
            neg_scores: Scores for negative pairs (logits)
            
        Returns:
            Loss value
        """
        return self.loss_fn(pos_scores, neg_scores)


def compute_hard_negatives(model, pos_lnc: torch.Tensor, pos_dis: torch.Tensor, 
                          neg_lnc_pool: torch.Tensor, neg_dis_pool: torch.Tensor,
                          edges: dict, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mine hard negatives by selecting negative samples with highest scores.
    
    Args:
        model: The HGAT-LDA model
        pos_lnc: Positive lncRNA indices
        pos_dis: Positive disease indices
        neg_lnc_pool: Pool of negative lncRNA indices
        neg_dis_pool: Pool of negative disease indices
        edges: Graph edges
        k: Number of hard negatives to select per positive
        
    Returns:
        Tuple of (hard_neg_lnc, hard_neg_dis)
    """
    model.eval()
    
    with torch.no_grad():
        hard_neg_lnc = []
        hard_neg_dis = []
        
        # For each positive pair
        for i in range(len(pos_lnc)):
            p_lnc = pos_lnc[i:i+1]
            p_dis = pos_dis[i:i+1]
            
            # Score all negative lncRNAs for this disease
            if len(neg_lnc_pool) > 0:
                # Create pairs with all negative lncRNAs
                dis_repeated = p_dis.repeat(len(neg_lnc_pool))
                neg_scores = model(neg_lnc_pool, dis_repeated, edges)
                
                # Select top-k hardest negatives (highest scores)
                _, hard_idx = torch.topk(neg_scores, min(k, len(neg_scores)))
                hard_neg_lnc.append(neg_lnc_pool[hard_idx])
                hard_neg_dis.append(dis_repeated[hard_idx])
        
        if hard_neg_lnc:
            hard_neg_lnc = torch.cat(hard_neg_lnc)
            hard_neg_dis = torch.cat(hard_neg_dis)
        else:
            # Fallback to random sampling if no hard negatives found
            hard_neg_lnc = neg_lnc_pool[:len(pos_lnc)]
            hard_neg_dis = neg_dis_pool[:len(pos_dis)]
    
    model.train()
    return hard_neg_lnc, hard_neg_dis


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get the appropriate loss function.
    
    Args:
        loss_type: Type of loss ('bce', 'focal', 'bpr', 'auc', 'pairwise')
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function module
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'bpr':
        return PairwiseRankingLoss('bpr')
    elif loss_type == 'auc':
        margin = kwargs.get('margin', 1.0)
        return PairwiseRankingLoss('auc', margin=margin)
    elif loss_type == 'pairwise':
        # Default pairwise is BPR
        pairwise_type = kwargs.get('pairwise_type', 'bpr')
        margin = kwargs.get('margin', 1.0)
        return PairwiseRankingLoss(pairwise_type, margin=margin)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
