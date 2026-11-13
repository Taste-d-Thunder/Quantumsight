# ⚛️ QuantumSight — Hybrid Quantum-Classical Image Classification

A hybrid quantum-classical deep learning framework for image classification, combining classical convolutional networks with parameterized quantum circuits.

This project investigates whether hybrid quantum-classical architectures can provide advantages in **classification accuracy, computational efficiency, model size, and adversarial robustness** compared with purely classical CNNs.

---

## 🚀 Getting Started

### Requirements

Python **3.8+** is required.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Core Dependencies

| Package       | Purpose              |
|---------------|----------------------|
| PyTorch       | Deep learning        |
| PennyLane     | Quantum ML circuits  |
| NumPy         | Numerical computing  |
| scikit-learn  | ML utilities         |
| Matplotlib    | Visualization        |
| torchattacks  | Adversarial testing  |

---

## 📁 Project Structure

```
QuantumSight/
├── .gitignore
├── requirements.txt
├── README.md
└── configs/
    ├── __init__.py
    └── config.py          # Central configuration
```

---

## ⚙️ Configuration

All experiment settings are centralized in `configs/config.py`.

### Supported Datasets

| Dataset     | Classes | Channels | Description                      |
|-------------|---------|----------|----------------------------------|
| MNIST       | 10      | 1        | Handwritten digit classification |
| CIFAR-100   | 100     | 3        | Natural image classification     |
| STL-10      | 10      | 3        | Larger natural images            |

Datasets are loaded via `torchvision.datasets` and downloaded automatically to `./data/` at runtime.

### Experiment Registry

Four experiment configurations are available:

| Experiment       | Dataset  | Qubits | Encoding  | Epochs |
|------------------|----------|--------|-----------|--------|
| `mnist_4q`       | MNIST    | 4      | Amplitude | 50     |
| `cifar100_4q`    | CIFAR-100| 4      | Amplitude | 50     |
| `stl10_4q`       | STL-10   | 4      | Amplitude | 50     |
| `mnist_2q_angle` | MNIST    | 2      | Angle     | 10     |

---

## 🔮 Roadmap

- [ ] Quantum circuit components (device, encodings)
- [ ] Classical and hybrid model architectures
- [ ] Training and evaluation pipeline
- [ ] Adversarial robustness testing
- [ ] Visualization and analysis tools
- [ ] Experiment scripts and notebooks
- [ ] Research paper and figures

---

## 🧰 Technology Stack

| Component         | Technology  |
|-------------------|-------------|
| Deep Learning     | PyTorch     |
| Quantum ML        | PennyLane   |
| Numerical         | NumPy       |
| Visualization     | Matplotlib  |
| ML Utilities      | scikit-learn|

---

## 📜 License

This project is released under the **MIT License**.
