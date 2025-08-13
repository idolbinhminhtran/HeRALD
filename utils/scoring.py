"""
Scoring utilities for canonical affinity computation.
Ensures consistent score direction (higher = better) across the entire pipeline.
"""

import torch
from typing import Optional, Union


def canonical_affinity(raw_scores: torch.Tensor, 
                       orientation: str, 
                       sign_override: Optional[float] = None) -> torch.Tensor:
    """
    Convert raw scores to canonical affinity scores where larger = better.
    
    Args:
        raw_scores: Raw model output scores (logits)
        orientation: Score orientation mode ('affinity', 'distance', or 'auto')
        sign_override: Sign multiplier (+1.0 or -1.0) when orientation='auto'
        
    Returns:
        Canonical affinity scores where higher values indicate better matches
        
    Notes:
        - 'affinity': Raw scores already represent affinity (higher = better)
        - 'distance': Raw scores represent distance (lower = better), so negate
        - 'auto': Use sign_override determined from calibration
    """
    if sign_override is not None:
        # When sign_override is provided, use it directly
        return sign_override * raw_scores
    
    if orientation == "affinity":
        # Raw scores already represent affinity (higher = better)
        return raw_scores
    
    if orientation == "distance":
        # Raw scores represent distance (lower = better), so negate
        return -raw_scores
    
    if orientation == "auto":
        # In auto mode, sign_override should always be provided after calibration
        # If not provided, return raw scores with a warning
        print("⚠️ Warning: auto mode without sign_override, returning raw scores")
        return raw_scores
    
    raise ValueError(f"model.score_orientation must be affinity|distance|auto, got {orientation}")


def calibrate_score_sign(pos_scores: torch.Tensor, 
                         neg_scores: torch.Tensor,
                         orientation: str) -> Optional[float]:
    """
    Calibrate the score sign based on a sample of positive and negative scores.
    
    Args:
        pos_scores: Raw scores for positive pairs
        neg_scores: Raw scores for negative pairs
        orientation: Score orientation mode
        
    Returns:
        Sign multiplier (+1.0 or -1.0) for auto mode, None for fixed orientations
    """
    if orientation in ["affinity", "distance"]:
        # Fixed orientation, no calibration needed
        return None
    
    if orientation == "auto":
        # Compute mean gap between positive and negative scores
        with torch.no_grad():
            gap = (pos_scores - neg_scores).mean().item()
        
        # If positives score higher than negatives, use positive sign
        # Otherwise, negate scores
        score_sign = 1.0 if gap >= 0 else -1.0
        
        print(f"[Score Calibration] orientation=auto, mean_gap={gap:.4f}, "
              f"score_sign={'+1' if score_sign > 0 else '-1'}")
        
        return score_sign
    
    raise ValueError(f"Invalid orientation: {orientation}")
