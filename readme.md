# Evaluating Graph Neural Networks for Causal Structure Learning in Bayesian Networks

This repository investigates whether Graph Neural Networks (GNNs) can evaluate or rank causal structures (DAGs) based on downstream predictive performance.

## Motivation

Neural networks excel as universal function approximators across many domains. However, in healthcare and high-stakes applications, predictive success alone is insufficient—correlation does not imply causation. This project tests whether GNNs with relational inductive bias can be used for causal discovery using only observational data.

**Central Question:** Can downstream predictive performance reliably signal causal correctness?

## Hypotheses

- **H₀ (Null):** Downstream prediction performance does not vary meaningfully with DAG structure interventions
- **Hₐ (Alternative):** Downstream prediction performance varies with structural intervention degree

## Methodology

### DAG Perturbation
- Start with known Bayesian Networks from [bnlearn](https://www.bnlearn.com/bnrepository/)
- Systematically perturb DAG structure by randomly deleting/adding edges
- Preserve edge count and acyclicity
- Noise levels: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`

### Data Generation
- Forward-sample 10,000 samples per BN
- Split: 6,000 train / 2,000 validation / 2,000 test
- Target node selected via topological ordering
- Discrete variables embedded into 16-dimensional feature space

### Model Architecture
- **GNN:** 2-layer Graph Convolutional Network (GCN)
- **Classifier:** 2-layer MLP with LeakyReLU
- **Regularization:** Dropout (0.5) between GCN and MLP
- **Optimizer:** Adam (lr=0.01 with decay)
- **Early stopping:** patience=10 epochs

## Datasets

| Size | Networks |
|------|----------|
| Small | Asia, Cancer, Sachs |
| Medium | Alarm, Child, Water |
| Large | Hailfinder, Hepar2, Diabetes |

**Total experiments:** 9 networks × 6 noise levels = 54 trained models

## Repository Structure

```
├── src/
│   ├── config.yaml          # Model & training hyperparameters
│   ├── constants.py          # Dataset constants and seeds
│   ├── data.py               # BNDataset class and data loading
│   ├── train.py              # Training, evaluation, inference loops
│   ├── run_experiment.py     # Main experiment runner
│   ├── utils.py              # DAG perturbation, early stopping, utilities
│   └── models/
│       ├── BNNet.py          # Main model (embeddings + GNN + MLP)
│       └── GNN.py            # GCN and GAT implementations
├── bn_notebook.ipynb         # Bayesian network exploration
├── plot_notebook.ipynb       # Results visualization
└── pyproject.toml            # Code formatting config
```

## Installation

```bash
pip install torch torch-geometric pgmpy pandas numpy scikit-learn matplotlib tensorboard pyyaml
```

## Usage

```bash
# Run experiments
python -m src.run_experiment \
    --fpath_config src/config.yaml \
    --dirpath_results ./results

# Train single BN
python -m src.train \
    --bn_name alarm \
    --fpath_config src/config.yaml \
    --dirpath_results ./results
```

## Evaluation Metrics

- **Downstream Classification Accuracy**
- **Bayesian Information Criterion (BIC)** — standard structure learning metric

## Key Findings

- BIC scores decrease with noise for small/medium BNs but become unreliable for larger networks
- Downstream accuracy shows weak/no correlation with noise for most BNs
- Only Asia, Water, and Diabetes show statistically significant trends

**Conclusion:** Results support H₀ — downstream predictive performance is **not** a reliable metric for evaluating causal DAGs.

## Citation

```bibtex
@article{karwande2023geometric,
  title={Geometric Deep Learning for Healthcare Applications},
  author={Karwande, Gaurang},
  year={2023}
}
```
