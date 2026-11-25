"""Evaluate a trained model checkpoint on test dataset."""
import os
import sys
import argparse
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import DEVICE, EXPERIMENTS, DATASETS
from src.data.data_loader import get_data_loaders
from src.models.hybrid_model import EnhancedHybridModel
from src.models.classical_model import EnhancedClassicalCNN
from src.training.trainer import test_and_log
from src.visualization.visualizer import MLVisualizer


def evaluate_checkpoint(exp_name: str, model_type: str, checkpoint_path: str):
    """Load model checkpoint and compute performance metrics against test split."""
    cfg = EXPERIMENTS[exp_name]
    dataset_name = cfg['dataset']
    out_dir = os.path.join(ROOT, 'results', exp_name)

    _, _, test_loader = get_data_loaders(dataset_name, cfg['batch_size'])

    if model_type == 'hybrid':
        model = EnhancedHybridModel(
            dataset_name,
            cfg['num_qubits'],
            cfg['num_layers'],
            cfg['encoding']
        ).to(DEVICE)
        metric_key = 'Hybrid_Test'
        display_name = 'Hybrid'
    else:
        model = EnhancedClassicalCNN(dataset_name).to(DEVICE)
        metric_key = 'Classical_Test'
        display_name = 'Classical'

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    visualizer = MLVisualizer(out_dir=out_dir, fmt=cfg['plot_format'])

    test_and_log(model, DEVICE, test_loader, 0, metric_key, 0, dataset_name, visualizer, cfg['epochs'])
    visualizer.plot_confusion_matrix(model, test_loader, DEVICE, DATASETS[dataset_name]['class_names'], display_name)
    visualizer._plot_predictions(model, test_loader, DEVICE, display_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved QuantumSight model checkpoint")
    parser.add_argument('--experiment', required=True, choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--model', default='hybrid', choices=['hybrid', 'classical'])
    parser.add_argument('--checkpoint', required=True, help="Path to .pth checkpoint file")
    args = parser.parse_args()

    evaluate_checkpoint(args.experiment, args.model, args.checkpoint)
