# GNN Message Passing Survey

---

## Overview

This repository contains the code, notebooks, and experiments accompanying a survey on **message passing in Graph Neural Networks (GNNs)**, with a particular focus on **Anomaly detection In graph structure** can be leveraged to enhance information propagation across graph structures.

The project investigates which model best fits for which type of graph structure

---

---

## Repository Structure

```
GNN-Message-Passing-Survey/
│
├── GCN/                    # Standard Graph Convolutional Network baseline
├── GAT/                    # Graph Attention Network experiments
├── GraphSage/              # GraphSAGE inductive learning experiments
├── HeterophilicGNN/        # GNN experiments on heterophilic graphs
├── CNN/                    # CNN baselines for comparison
├── Anamoly/                # Anomaly detection using GNN message passing
├── DataExperimentation/    # Dataset exploration and preprocessing notebooks
├── data_ablation/          # Ablation studies on graph data properties
│
├── main.py                 # Entry point / runner script
├── neural.ipynb            # Core neural architecture notebook
└── LICENSE                 # MIT License
```

---

## Methods Studied

| Model | Description |
|---|---|
| **GCN** | Kipf & Welling (2017) baseline with standard spectral message passing |
| **GAT** | Attention-weighted message passing (Veličković et al., 2018) |
| **GraphSAGE** | Inductive neighborhood sampling and aggregation |
| **HeterophilicGNN** | Message passing tailored for disassortative graphs |
| **Anomaly Detection** | GNN-based graph anomaly detection via message distortion analysis |

-
---

## Getting Started

### Prerequisites

```bash
pip install torch torch-geometric numpy pandas matplotlib scikit-learn networkx jupyter
```

### Running the Notebooks

Each subdirectory contains self-contained Jupyter notebooks. To explore the GCN baseline:

```bash
cd GCN
jupyter notebook
```

To run the main script:

```bash
python main.py
```

### Recommended Order

1. `DataExperimentation/` — understand the datasets used
2. `GCN/` — establish the baseline
3. `GAT/` and `GraphSage/` — explore attention and inductive variants
4. `HeterophilicGNN/` — test on challenging heterophilic settings
5. `Anamoly/` — apply to anomaly detection
6. `data_ablation/` — understand the effect of graph structure on results

---

## Datasets

Experiments are conducted on standard graph benchmark datasets including (but not limited to):

- **Cora**, **Citeseer**, **PubMed** — citation network node classification
- **Cornell**, **Texas**, **Wisconsin** — heterophilic WebKB graphs
- Custom synthetic graphs for controlled amplitude amplification studies

---

## Results (Summary)

- **Heterophilic graphs** where uniform neighbor aggregation degrades performance
- **Sparse graphs** where amplification of relevant signals compensates for limited connectivity
- **Anomaly detection** where selective suppression of "normal" messages helps surface anomalous nodes

Full experimental results and ablation tables are available in the respective notebooks.

---

## Theoretical Background

This work draws from:

- **Spectral Graph Theory** — graph Laplacian and signal propagation
- **GCN** (Kipf & Welling, 2017) — semi-supervised node classification
- **GAT** (Veličković et al., 2018) — attention in graph networks
- **Heterophily in GNNs** — Zhu et al. (2020), Pei et al. (2020)

---


## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Ashutosh Kumar** [GitHub](https://github.com/AshutoshKumar1007)
**Gowrav Sharma** [GitHub](https://github.com/Gowravsharma)
