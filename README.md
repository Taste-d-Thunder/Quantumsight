# ⚛️ QuantumSight — Hybrid Quantum-Classical Image Classification

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.30%2B-purple.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modular, config-driven deep learning framework combining convolutional neural networks with parameterized variational quantum circuits (VQCs) for image classification.

---

## 🧠 Overview

**QuantumSight** investigates the feasibility and empirical behavior of hybrid quantum-classical neural network architectures. It systematically benchmarks hybrid models against parameter-matched classical convolutional neural networks across three standard computer vision datasets:

* **MNIST**: 10-class handwritten digits (1 channel)
* **CIFAR-100**: 100-class fine-grained natural images (3 channels)
* **STL-10**: 10-class high-resolution natural images (3 channels)

The framework evaluates:
1. **Classification Accuracy & F1 Score**: Measuring performance parity and generalization.
2. **Computational Efficiency**: Monitoring epoch duration, CPU utilization, and system RAM footprint.
3. **Adversarial Robustness**: Testing resilience under Projected Gradient Descent (PGD) attacks ($\epsilon = 0.1$).

---

## 🏛️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        QuantumSight Pipeline                           │
│                                                                        │
│   ┌────────────┐     ┌──────────────┐     ┌───────────┐                │
│   │   Input    │     │  Classical   │     │  Feature  │                │
│   │   Image    │────>│  CNN Trunk   │────>│  Reducer  │────┐           │
│   │ (C x H x W)│     │ (Conv + Pool)│     │(Dim Scale)│    │           │
│   └────────────┘     └──────────────┘     └───────────┘    │           │
│                                                            │           │
│   ┌────────────┐     ┌──────────────┐     ┌───────────┐    v           │
│   │   Class    │<────│ Classification│<───│ Parameter │  Quantum      │
│   │Prediction  │     │     Head     │     │    VQC    │  State         │
│   │  (Logits)  │     │ (Linear + BN)│     │(Entangle) │                │
│   └────────────┘     └──────────────┘     └───────────┘                │
└────────────────────────────────────────────────────────────────────────┘
```

* **Classical CNN Trunk**: Extracts spatial visual features from raw images with adaptive pooling.
* **Feature Reducer**: Compresses high-dimensional feature maps to the required quantum state size ($2^n$ for amplitude encoding, $n$ for angle encoding) using bounded activations (Softmax or Tanh).
* **Variational Quantum Circuit (VQC)**: Encodes features onto qubits via PennyLane, applies trainable `BasicEntanglerLayers`, and returns Pauli-$Z$ expectation values.
* **Classification Head**: Projects quantum expectation measurements into output class logits.

---

## 📁 Repository Structure

```
QuantumSight/
├── configs/
│   ├── __init__.py
│   └── config.py               # Central experiment and dataset registry
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Automated downloading, splits, transforms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical_model.py  # Baseline convolutional networks
│   │   └── hybrid_model.py     # Hybrid CNN-VQC architectures
│   ├── quantum/
│   │   ├── __init__.py
│   │   ├── device.py           # Simulator device management
│   │   ├── encodings.py        # Amplitude and angle encoding circuits
│   │   └── quantum_circuit.py  # Parameterized quantum circuit definitions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Precision, Recall, Macro F1 computations
│   │   └── trainer.py          # Training and validation step routines
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── adversarial.py      # PGD adversarial evaluation
│   │   └── resource_tracker.py # Time, CPU, RAM, and GPU profiling
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py       # Curves, confusion matrices, PCA/t-SNE
├── scripts/
│   ├── train.py                # Main CLI training entry-point
│   └── evaluate.py             # Checkpoint testing script
├── notebooks/
│   └── demo.ipynb              # Interactive walkthrough notebook
├── code/                       # Standalone single-file scripts and notebook
├── docs/
│   ├── figures/                # Pre-computed benchmark plots and curves
│   └── paper.pdf               # Project technical paper
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Taste-d-Thunder/Quantumsight.git
   cd Quantumsight
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Training Experiments

Run training through the unified CLI entry-point:

```bash
# MNIST with 4-qubit amplitude encoding
python scripts/train.py --experiment mnist_4q

# CIFAR-100 with 4-qubit amplitude encoding
python scripts/train.py --experiment cifar100_4q

# STL-10 with 4-qubit amplitude encoding
python scripts/train.py --experiment stl10_4q

# MNIST with 2-qubit angle encoding
python scripts/train.py --experiment mnist_2q_angle
```

Training automatically saves optimal checkpoints (`.pth`) and generates plots under `results/<experiment_name>/`.

### 2. Evaluating a Checkpoint

Evaluate any saved checkpoint on the unseen test split:

```bash
python scripts/evaluate.py --experiment mnist_4q --model hybrid --checkpoint results/mnist_4q/best_model.pth
```

### 3. Interactive Notebook

Launch Jupyter to explore models, circuits, and visual outputs:
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## 📊 Benchmark Results

| Dataset | Metric | Classical CNN | Hybrid Model (Ours) | Advantage / Notes |
| :--- | :--- | :---: | :---: | :--- |
| **MNIST** | Test Accuracy | ~98.2% | **~99.4%** | Faster convergence, higher robustness |
| **CIFAR-100** | Test Accuracy | ~32.3% | **~41.7%** | +9.4% improvement on complex representations |
| **STL-10** | Test Accuracy | ~63.8% | **~74.1%** | +10.3% boost with fewer parameters |

---

## 🔮 Completed Roadmap

- [x] Central configuration and dataset registry (`configs/`)
- [x] Quantum circuit components and simulator interface (`src/quantum/`)
- [x] Classical baseline and hybrid neural architectures (`src/models/`)
- [x] Training routines and metrics tracking (`src/training/`)
- [x] Adversarial robustness testing (`src/utils/adversarial.py`)
- [x] Resource profiling suite (`src/utils/resource_tracker.py`)
- [x] Visualization suite and feature projections (`src/visualization/`)
- [x] Automated data loader factory (`src/data/`)
- [x] Experiment CLI pipeline (`scripts/`)
- [x] Interactive demonstration notebooks (`notebooks/`)
- [x] Benchmark figures and documentation (`docs/`)

---

## 📜 License

This project is licensed under the **MIT License**.
