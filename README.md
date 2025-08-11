## HeRALD: Heterogeneous Relational Attention for lncRNA–Disease Association Prediction

HeRALD is a heterogeneous graph attention model for predicting lncRNA–disease associations by jointly leveraging lncRNA–lncRNA, gene–gene, disease–disease similarities and lncRNA–gene / gene–disease / lncRNA–disease associations. This repository provides a clean, reproducible PyTorch implementation with first-class support for Leave-One-Out Cross-Validation (LOOCV).

### Highlights
- Multi-relation, multi-head attention over heterogeneous graph relations
- End-to-end PyTorch with clean trainer/evaluator APIs
- Built-in LOOCV with summary statistics and plots
- Configurable via YAML or CLI; sensible defaults included
- CPU and GPU support (GPU recommended for speed)

### Project structure
```
HGAT-LDA/
  main.py                    # Entry point (train/evaluate/loocv)
  configs/default.yaml       # Default configuration
  data/                      # Data loading + heterogeneous graph construction
  models/                    # HeRALD model and layers
  training/                  # Trainer and LOOCV evaluator
  utils/                     # Config parser, metrics, plotting
  Dataset/                   # Input matrices (included)
```

### Requirements
- Python 3.8–3.11
- CUDA-capable GPU optional (CPU supported; GPU recommended)

Install dependencies
```bash
pip install -r requirements.txt
```

The provided `requirements.txt` includes only the necessary packages:
- torch>=2.0.0, numpy, scikit-learn, pandas, matplotlib, seaborn, tqdm, pyyaml

### Dataset
Place (or keep) the following files under `Dataset/` (already included):
- `LncFunGauSim.txt` — lncRNA–lncRNA similarity
- `GeneLlsGauSim.txt` — gene–gene similarity
- `DisSemGauSim.txt` — disease–disease similarity
- `GeneLncMat.txt` — gene–lncRNA associations
- `GeneDisMat.txt` — gene–disease associations
- `DisLncMat.txt` — disease–lncRNA associations

Optional: `lncRNA_name.txt`, `gene_name.txt`, `disease_name.txt` for interpretability.

### Quick start (LOOCV)
Run LOOCV with default configs and save outputs to `results/`:
```bash
python3 main.py --mode loocv --config configs/default.yaml --results_path results
```

Artifacts produced:
- `results/loocv_results.json` — per-fold AUCs and summary statistics
- `results/loocv_results.png` — histogram + box plot of LOOCV AUCs

Key LOOCV parameters live under `evaluation` in `configs/default.yaml`:
- `loocv_epochs` — epochs per fold
- `loocv_batch_size` — batch size during LOOCV training
- `loocv_lr` — optimizer learning rate used in LOOCV
- `loocv_neg_ratio` — negative sampling ratio used per positive

### Train once and evaluate (optional)
```bash
python3 main.py --mode train --config configs/default.yaml --results_path results
```
Saves:
- `results/best_model.pth` — best checkpoint by validation loss
- `results/training_history.png` — train/val loss curves
- `results/evaluation_results.{json,csv}` — metrics summary (AUC, AUPR, F1, etc.)

### Configuration
All hyperparameters are configured in `configs/default.yaml`. Example fields:
```yaml
model:
  emb_dim: 128
  num_layers: 1
  dropout: 0.3
  num_heads: 4
  relation_dropout: 0.2
  use_layernorm: true
  use_residual: true

training:
  lr: 0.0013
  weight_decay: 1.0e-6
  batch_size: 256
  num_epochs: 100
  val_split: 0.1
  early_stopping_patience: 15
  neg_ratio: 5
  cosine_tmax: 100
  use_focal_loss: false
  label_smoothing: 0.15

evaluation:
  loocv_epochs: 10
  loocv_batch_size: 64
  loocv_lr: 0.0027
  loocv_neg_ratio: 2

data:
  data_dir: Dataset
  sim_row_normalize: true
  sim_topk: 5
  threshold: 0.0

system:
  device: auto   # auto | cuda | cpu
  seed: 42
```

You can override many of these via CLI. See `python3 main.py --help` for all options.

### Reproducibility
- We set seeds for Python, NumPy, and PyTorch, and enable deterministic/cuDNN settings when feasible.
- For strict reproducibility across hardware/drivers, keep `system.seed` fixed and run on the same device type. Small numerical differences can still occur due to low-level kernels.

### Performance tips
- Use a GPU when available (`system.device: auto` will select CUDA when present).
- `training.batch_size` and `training.neg_ratio` affect throughput and memory.
- `data.sim_topk` sparsifies similarity graphs; increasing it may improve accuracy at the cost of speed.
- AMP is enabled in standard training when CUDA is available; LOOCV disables AMP for stability.

### Model naming
HeRALD is the project name. The core model class in code is `HGAT_LDA` under `models/hgat_lda.py`.

### Citation
If you use HeRALD in your research, please cite this repository. A suggested BibTeX entry template:
```
@software{herald_2025,
  title   = {HeRALD: Heterogeneous Relational Attention for lncRNA–Disease Association Prediction},
  author  = {Your Name},
  year    = {2025},
  url     = {https://github.com/your-org/your-repo}
}
```



