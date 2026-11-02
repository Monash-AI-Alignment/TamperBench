"""Moment-matching penalty: MMD under homogeneous polynomial kernels.

Under k_j(x, y) = (x^T y)^j, MMD^2 equals the squared Frobenius distance between the
j-th uncentered moment tensors E[x^{(x)j}] of the two distributions, so summing degrees
1..m-1 is zero exactly when the first m-1 moments match. Every term is a Gram matrix of
dot products; the moment tensors are never formed.

The estimator is the unbiased one (diagonal of the within-class Gram matrices
excluded). The plug-in estimator, which keeps the diagonal, is biased upwards by the
within-class spread (for j = 1 by tr(Sigma_x)/n + tr(Sigma_y)/m), so minimising it also
collapses each class towards a point instead of only matching the classes.
"""

import torch


def polynomial_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int,
    x_extra: torch.Tensor | None = None,
    y_extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unbiased MMD^2 between samples `x` and `y` under k(a, b) = (a^T b)^degree.

    Args:
        x: Samples of the first distribution, shape (n, d).
        y: Samples of the second distribution, shape (m, d).
        degree: Kernel degree j >= 1; the estimate targets ||E[x^{(x)j}] - E[y^{(x)j}]||_F^2.
        x_extra: Earlier samples of the first distribution, pooled with `x` but detached.
        y_extra: Earlier samples of the second distribution, pooled with `y` but detached.

    Returns:
        Scalar estimate (may be slightly negative, as unbiased estimates can be).
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    if x_extra is not None:
        x = torch.cat([x, x_extra.detach()])
    if y_extra is not None:
        y = torch.cat([y, y_extra.detach()])
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        raise ValueError(f"need at least 2 samples per distribution, got {n} and {m}")
    k_xx = (x @ x.T) ** degree
    k_yy = (y @ y.T) ** degree
    k_xy = (x @ y.T) ** degree
    xx = (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
    yy = (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
    return xx + yy - 2 * k_xy.mean()


def moment_penalty(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    m: int,
    bases: torch.Tensor | None = None,
    harmful_extra: torch.Tensor | None = None,
    harmless_extra: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """D_m summed over layers, averaged over token positions.

    D_m = sum_{j=1}^{m-1} MMD^2_{k_j}. Degree 1 (the class means, which difference-in-means
    abliteration reads) uses the full space; degrees >= 2 use the per-layer `bases`
    subspace when given, where the moment estimates are not dominated by noise.

    Args:
        harmful: Activations, shape (layers, positions, n, d), already scaled.
        harmless: Activations, shape (layers, positions, n', d), already scaled.
        m: Lowest moment allowed to differ; degrees 1..m-1 are penalised (m >= 2).
        bases: Orthonormal columns per layer for degrees >= 2, shape (layers, d, k).
        harmful_extra: Detached earlier harmful activations, shape (layers, positions, q, d).
        harmless_extra: Detached earlier harmless activations, shape (layers, positions, q', d).

    Returns:
        The penalty and its per-degree parts (each summed over layers, averaged over positions).
    """
    if m < 2:
        raise ValueError(f"m must be >= 2 (m = 1 penalises nothing), got {m}")
    layers, positions = harmful.shape[:2]
    parts: dict[int, torch.Tensor] = {}
    for degree in range(1, m):
        total = harmful.new_zeros(())
        for layer in range(layers):
            basis = None if degree == 1 or bases is None else bases[layer]
            for position in range(positions):
                total = total + polynomial_mmd2(
                    _select(harmful, layer, position, basis),
                    _select(harmless, layer, position, basis),
                    degree,
                    None if harmful_extra is None else _select(harmful_extra, layer, position, basis),
                    None if harmless_extra is None else _select(harmless_extra, layer, position, basis),
                )
        parts[degree] = total / positions
    return sum(parts.values(), harmful.new_zeros(())), parts


def _select(activations: torch.Tensor, layer: int, position: int, basis: torch.Tensor | None) -> torch.Tensor:
    sample = activations[layer, position]
    return sample if basis is None else sample @ basis


class MomentBank:
    """Ring buffer of detached activations per class, extending the moment estimates past one batch."""

    def __init__(self, size: int) -> None:
        """Keep up to `size` samples per (layer, position) and class."""
        self.size: int = size
        self.harmful: torch.Tensor | None = None
        self.harmless: torch.Tensor | None = None

    def push(self, harmful: torch.Tensor, harmless: torch.Tensor) -> None:
        """Append a batch (shape (layers, positions, n, d)); the oldest samples drop out."""
        self.harmful = self._append(self.harmful, harmful)
        self.harmless = self._append(self.harmless, harmless)

    def _append(self, bank: torch.Tensor | None, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.detach()
        merged = batch if bank is None else torch.cat([bank, batch], dim=2)
        return merged[:, :, -self.size :]
