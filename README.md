# A Principled Framework for Multi-View Contrastive Learning

[![Paper](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)](https://github.com/pakoromilas/Multi-View-CL)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10+-red.svg)](https://pytorch.org/)

**[Panagiotis Koromilas](https://scholar.google.com/citations?user=DMI5W9wAAAAJ&hl=el), Efthymios Georgiou, Giorgos Bouritsas, Theodoros Giannakopoulos, Mihalis A. Nicolaou, Yannis Panagakis**

This repository contains the official PyTorch implementation of our paper **"A Principled Framework for Multi-View Contrastive Learning"**.


## 🎯 TL;DR

We introduce **MV-InfoNCE** and **MV-DHEL**, two theoretically grounded objectives for multi-view contrastive learning that:
- ✅ Properly incorporate **interactions** across all views (not just pairs!)
- ✅ Scale to **arbitrary number of views** with consistent improvements
- ✅ **Mitigate dimensionality collapse** when using 5+ views
- ✅ Extends Multimodal Contrastive Learning beyond 2 modalities

## 📊 Key Results

Our methods consistently outperform existing approaches across all benchmarks:

<table>
<tr>
<td>

| Dataset | Method | 2 Views | 4 Views |
|---------|--------|---------|---------|
| CIFAR10 | Baseline Best | 86.0% | 88.7% |
| | **MV-DHEL** | **87.4%** | **89.5%** |
| CIFAR100 | Baseline Best | 58.1% | 61.1% |
| | **MV-DHEL** | **59.4%** | **62.7%** |

</td>
<td>

| Dataset | Method | 2 Views | 4 Views |
|---------|--------|---------|---------|
| ImageNet-100 | Baseline Best | 72.2% | 75.0% |
| | **MV-DHEL** | **73.3%** | **77.2%** |
| ImageNet-1K | Baseline Best | 60.0% | 62.4% |
| | **MV-DHEL** | **61.2%** | **62.6%** |

</td>
</tr>
</table>

## 🔬 What's Wrong with Current Multi-View Methods?

Current approaches simply aggregate pairwise losses, leading to:

1. **Conflicting Objectives**: Each view must satisfy multiple competing loss terms
2. **Missing Interactions**: Critical view relationships are ignored
3. **Coupled Optimization**: Alignment and uniformity interfere with each other
4. **Poor Scaling**: Benefits diminish with more views

Our framework addresses **all** these limitations with a principled mathematical foundation.

## 🎨 Our Approach

### Three Fundamental Principles

We identify three principles that any proper multi-view contrastive loss must satisfy:

| Principle | Description | PWE | PVC | MV-InfoNCE | MV-DHEL |
|-----------|-------------|:---:|:---:|:----------:|:-------:|
| **P1**: Simultaneous Alignment | All views aligned in one term | ❌ | ❌ | ✅ | ✅ |
| **P2**: Accurate Energy | Complete pairwise interactions | ❌ | ❌ | ✅ | ✅ |
| **P3**: One Term per Instance | Single optimization objective | ❌ | ❌ | ✅ | ✅ |
| **Bonus**: Decoupled Optimization | Alignment ⊥ Uniformity | ❌ | ❌ | ❌ | ✅ |

### Our Methods

**MV-InfoNCE** - Natural extension of InfoNCE to multiple views:
```math
\mathcal{L}_{\text{MV-InfoNCE}} = \frac{1}{M} \sum_{i=1}^M \log \left(\frac{\sum_{l\in[N], l' \in [N] \setminus l} e^{\mathbf{U}_{i,l}^{\top} \mathbf{U}_{i,l'}/\tau}}{\sum_{l \in [N], m \in [N]\setminus l, j \in [M]} e^{\mathbf{U}_{i,l}^{\top} \mathbf{U}_{j,m}/\tau}}\right)
```

**MV-DHEL** - Decoupled optimization with superior efficiency:
```math
\mathcal{L}_{\text{MV-DHEL}} = \frac{1}{M} \sum_{i=1}^M \log \left(\frac{\sum_{l\in[N], l' \in [N] \setminus l} e^{\mathbf{U}_{i,l}^{\top} \mathbf{U}_{i,l'}/\tau}}{\prod_{l \in [N]} \sum_{j \in [M]} e^{\mathbf{U}_{i,l}^{\top} \mathbf{U}_{j,l}/\tau}}\right)
```

Where:
- `i, j ∈ [M]`: instances in the batch
- `l, l', m ∈ [N]`: different views of the data
- `U_{i,l}`: representation of instance `i` in view `l`
- `τ`: temperature parameter

## 🚀 Quick Start

### Training

Train with our state-of-the-art MV-DHEL objective:

```bash
# MV-DHEL (Recommended)
python3 train.py \
    --config=configs/cifar_train_epochs200_bs256_05.yaml \
    --multiplier=4 \
    --loss=MVDHEL

# MV-InfoNCE
python3 train.py \
    --config=configs/cifar_train_epochs200_bs256_05.yaml \
    --multiplier=4 \
    --loss=MVINFONCE
```

<details>
<summary><b>Baseline Methods</b></summary>

```bash
# PVC (Poly-View Contrastive)
python3 train.py --config=configs/cifar_train_epochs200_bs256_05.yaml --multiplier=4 --loss=PVCLoss

# PWE (Pairwise Aggregation)
python3 train.py --config=configs/cifar_train_epochs200_bs256_05.yaml --multiplier=4 --loss=NTXent --loss_pwe=True

# AVG (Average Views)
python3 train.py --config=configs/cifar_train_epochs200_bs256_05.yaml --multiplier=4 --loss=NTXent --loss_avg=True
```
</details>

### Evaluation

```bash
# Linear evaluation
python3 train.py \
    --config=configs/cifar_eval.yaml \
    --checkpoint=path/to/checkpoint.pth
```

## 🔍 Key Insights

1. **Theoretical Foundation**: We prove both methods optimize for the same asymptotic behavior as InfoNCE
2. **Computational Efficiency**: MV-DHEL has O(M²N) complexity vs O(M²N²) for other methods
3. **Dimensionality Preservation**: With 5+ views, MV-DHEL fully utilizes the embedding space
4. **Multimodal Ready**: Extends naturally to 3+ modalities (validated on sentiment analysis)

## 📚 Citation

If you find our work useful, please cite:

```bibtex
@article{koromilas2025principled,
  title={A Principled Framework for Multi-View Contrastive Learning},
  author={Koromilas, Panagiotis and Georgiou, Efthymios and Bouritsas, Giorgos and 
          Giannakopoulos, Theodoros and Nicolaou, Mihalis A. and Panagakis, Yannis},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## 🤝 Acknowledgments

This codebase builds upon [SimCLR-PyTorch](https://github.com/AndrewAtanov/simclr-pytorch). We thank the authors for their excellent implementation.

## 📧 Contact

For questions and discussions:
- Open an [issue](https://github.com/pakoromilas/Multi-View-CL/issues) for bug reports or for general questions
- Email: [pakoromilas@di.uoa.gr](mailto:pakoromilas@di.uoa.gr)
