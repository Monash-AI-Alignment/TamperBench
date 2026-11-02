"""Probes that check where harmful and harmless activations are separable.

Within a moment-guarded band, the difference-in-means probe and the linear probe should
sit at chance (equal class means blind every linear classifier; Belrose et al. 2023)
while a nonlinear probe still separates the classes. Each returns held-out accuracy on
balanced classes, so chance is 0.5.
"""

import torch
import torch.nn.functional as F


def train_test_split(n: int, test_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffled train and test indices into `n` samples."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = int(n * (1 - test_fraction))
    return order[:cut], order[cut:]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(((logits > 0).long() == labels).float().mean())


def dim_probe_accuracy(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor
) -> float:
    """Classify by projection onto the training difference in means, threshold at the midpoint."""
    mean_1, mean_0 = x_train[y_train == 1].mean(0), x_train[y_train == 0].mean(0)
    direction = mean_1 - mean_0
    threshold = (mean_1 + mean_0) @ direction / 2
    return _accuracy(x_test @ direction - threshold, y_test)


def _standardise(x_train: torch.Tensor, x_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def linear_probe_accuracy(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    l2: float = 1e-2,
    steps: int = 200,
) -> float:
    """L2-regularised logistic regression (L-BFGS) on standardised features."""
    x_train, x_test = _standardise(x_train, x_test)
    weight = torch.zeros(x_train.shape[1], device=x_train.device, requires_grad=True)
    bias = torch.zeros((), device=x_train.device, requires_grad=True)
    optimiser = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")
    targets = y_train.float()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x_train @ weight + bias, targets) + l2 * weight.square().sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        return _accuracy(x_test @ weight + bias, y_test)


def mlp_probe_accuracy(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    hidden: int = 256,
    epochs: int = 300,
    seed: int = 0,
) -> float:
    """One-hidden-layer ReLU network, full-batch Adam with weight decay, on standardised features."""
    x_train, x_test = _standardise(x_train, x_test)
    torch.manual_seed(seed)
    network = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
    ).to(x_train.device)
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-3)
    targets = y_train.float()
    for _ in range(epochs):
        optimiser.zero_grad()
        F.binary_cross_entropy_with_logits(network(x_train).squeeze(-1), targets).backward()
        optimiser.step()
    with torch.no_grad():
        return _accuracy(network(x_test).squeeze(-1), y_test)
