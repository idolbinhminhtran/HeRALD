#!/usr/bin/env python3
"""
Main training script for HGAT-LDA model.
"""

import os
import sys
import torch
import numpy as np
import random
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.data_loader import load_dataset, get_dataset_info
from data.graph_construction import construct_heterogeneous_graph, generate_negative_pairs, get_positive_pairs
from models.hgat_lda import HGAT_LDA
from training.trainer import HGATLDATrainer
from training.evaluator import HGATLDAEvaluator
from utils.config import Config, parse_args, create_config_file
from utils.metrics import save_results, print_metrics_summary, plot_training_history


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)


def main():
    args = parse_args()
    config = Config.from_yaml(args.config) if args.config else Config.from_args(args)
    set_seed(config.system.seed)
    device = get_device(config.system.device)
    print(f"Using device: {device}")
    results_dir = Path(args.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    data_dict = load_dataset(config.data.data_dir)
    dataset_info = get_dataset_info()

    print("Constructing heterogeneous graph...")
    edges = construct_heterogeneous_graph(
        data_dict,
        sim_topk=getattr(config.data, 'sim_topk', None),
        sim_row_normalize=getattr(config.data, 'sim_row_normalize', True),
        sim_threshold=getattr(config.data, 'threshold', 0.0)
    )

    pos_pairs = get_positive_pairs(data_dict['lnc_disease_assoc'])
    print(f"Number of positive pairs: {len(pos_pairs)}")

    print("Initializing model...")
    model = HGAT_LDA(
        num_lncRNAs=dataset_info['num_lncRNAs'],
        num_genes=dataset_info['num_genes'],
        num_diseases=dataset_info['num_diseases'],
        edges=edges,
        emb_dim=config.model.emb_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        num_heads=getattr(config.model, 'num_heads', 4),
        relation_dropout=getattr(config.model, 'relation_dropout', 0.0),
        use_layernorm=getattr(config.model, 'use_layernorm', True),
        use_residual=getattr(config.model, 'use_residual', True)
    )

    trainer = HGATLDATrainer(
        model=model,
        device=device,
        lr=float(getattr(config.training, 'lr', 1e-3)),
        weight_decay=float(getattr(config.training, 'weight_decay', 1e-5)),
        batch_size=int(getattr(config.training, 'batch_size', 256)),
        enable_progress=True,
        neg_ratio=int(getattr(config.training, 'neg_ratio', 1)),
        cosine_tmax=(int(config.training.cosine_tmax) if getattr(config.training, 'cosine_tmax', None) is not None else None),
        use_amp=getattr(config.system, 'use_amp', True),
        use_focal_loss=getattr(config.training, 'use_focal_loss', True),
        label_smoothing=float(getattr(config.training, 'label_smoothing', 0.1))
    )

    pos_pairs, neg_pairs_all = trainer.prepare_data(
        data_dict['lnc_disease_assoc'],
        dataset_info['num_lncRNAs'],
        dataset_info['num_diseases']
    )

    if args.mode == 'train':
        print("Starting training...")
        model_save_path = args.model_path or results_dir / "best_model.pth"
        history = trainer.train(
            pos_pairs=pos_pairs,
            neg_pairs_all=neg_pairs_all,
            edges=edges,
            num_epochs=config.training.num_epochs,
            val_split=config.training.val_split,
            early_stopping_patience=config.training.early_stopping_patience,
            save_path=str(model_save_path)
        )
        plot_training_history(
            history['train_losses'],
            history['val_losses'],
            save_path=str(results_dir / "training_history.png")
        )
        print(f"Training completed. Best model saved to {model_save_path}")

    if args.mode in ['evaluate', 'train']:
        print("Evaluating model...")
        evaluator = HGATLDAEvaluator(model, device)
        eval_neg_pairs = generate_negative_pairs(
            pos_pairs,
            dataset_info['num_lncRNAs'],
            dataset_info['num_diseases'],
            num_negatives=len(pos_pairs)
        )
        metrics = evaluator.evaluate_all_metrics(pos_pairs, eval_neg_pairs, edges)
        print_metrics_summary(metrics)
        save_results(metrics, str(results_dir / "evaluation_results"))
        from utils.metrics import plot_roc_curve, plot_pr_curve
        plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'], save_path=str(results_dir / "roc_curve.png"))
        plot_pr_curve(metrics['precision'], metrics['recall'], metrics['aupr'], save_path=str(results_dir / "pr_curve.png"))

    if args.mode == 'loocv':
        print("Starting LOOCV evaluation...")
        evaluator = HGATLDAEvaluator(model, device)
        auc_scores = evaluator.leave_one_out_cross_validation(
            pos_pairs=pos_pairs,
            edges=edges,
            num_lncRNAs=dataset_info['num_lncRNAs'],
            num_diseases=dataset_info['num_diseases'],
            num_epochs=int(getattr(config.evaluation, 'loocv_epochs', 30)),
            batch_size=int(getattr(config.evaluation, 'loocv_batch_size', 256)),
            lr=float(getattr(config.evaluation, 'loocv_lr', 1e-3)),
            neg_ratio=int(getattr(config.training, 'neg_ratio', 1))
        )
        from utils.metrics import calculate_loocv_statistics, plot_loocv_results
        stats = calculate_loocv_statistics(auc_scores)
        print("\n" + "="*50)
        print("LOOCV RESULTS")
        print("="*50)
        print(f"Mean AUC: {stats['mean_auc']:.4f} ± {stats['std_auc']:.4f}")
        print(f"Min AUC:  {stats['min_auc']:.4f}")
        print(f"Max AUC:  {stats['max_auc']:.4f}")
        print(f"Median AUC: {stats['median_auc']:.4f}")
        print(f"Number of folds: {stats['num_folds']}")
        print("="*50)
        import json
        loocv_results = {'auc_scores': auc_scores, 'statistics': stats}
        with open(results_dir / "loocv_results.json", 'w') as f:
            json.dump(loocv_results, f, indent=2)
        plot_loocv_results(auc_scores, save_path=str(results_dir / "loocv_results.png"))

    print(f"All results saved to {results_dir}")


if __name__ == "__main__":
    main() 