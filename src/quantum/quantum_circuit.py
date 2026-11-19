"""Quantum circuit definition and device helpers."""
import torch
import pennylane as qml
from configs.config import EXPERIMENTS, DEFAULT_EXPERIMENT

_default_cfg = EXPERIMENTS.get(DEFAULT_EXPERIMENT, {})
num_qubits = _default_cfg.get("num_qubits", 4)
num_layers = _default_cfg.get("num_layers", 2)


def get_quantum_device(wires=None):
    """Return a PennyLane simulator device."""
    n_wires = wires if wires is not None else num_qubits
    try:
        dev = qml.device("default.qubit", wires=n_wires)
        print(f"Using default qubit simulator with {n_wires} qubits")
        return dev
    except Exception as e:
        print(f"Could not initialize quantum device: {str(e)}")
        raise


def quantum_circuit(inputs, weights):
    """Quantum circuit with amplitude encoding for 4 qubits (16 amplitudes)."""
    inputs = torch.abs(inputs)
    inputs = inputs / torch.norm(inputs)

    qml.AmplitudeEmbedding(
        inputs,
        wires=range(num_qubits),
        normalize=True,
        pad_with=0.0,
    )
    qml.BasicEntanglerLayers(weights, wires=range(num_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
