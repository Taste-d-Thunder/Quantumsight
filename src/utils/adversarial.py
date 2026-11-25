import torch

try:
    from torchattacks import PGD
except ImportError:
    PGD = None


def evaluate_adversarial_robustness(model, data_loader, device, epsilon: float = 0.1):
    """Evaluate model accuracy under Projected Gradient Descent (PGD) adversarial perturbation.

    Args:
        model (torch.nn.Module): Target evaluation model.
        data_loader (DataLoader): DataLoader providing evaluation batches.
        device (torch.device): Compute device (CPU or CUDA).
        epsilon (float): Maximum perturbation magnitude for the attack.

    Returns:
        float: Top-1 adversarial classification accuracy percentage.
    """
    if PGD is None:
        raise ImportError(
            "The 'torchattacks' package is required for adversarial robustness evaluation. "
            "Install dependencies with 'pip install -r requirements.txt' or 'pip install torchattacks'."
        )

    model.eval()
    attack = PGD(model, eps=epsilon, alpha=0.01, steps=10)
    correct = 0
    total = 0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        perturbed_data = attack(data, target)
        predictions = model(perturbed_data).argmax(dim=1)
        correct += predictions.eq(target).sum().item()
        total += target.size(0)

    robustness = 100.0 * correct / max(total, 1)
    print(f"Adversarial Robustness (eps={epsilon}): {robustness:.2f}%")
    return robustness
