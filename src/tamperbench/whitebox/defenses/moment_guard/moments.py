"""Moment-matching penalty: MMD under homogeneous polynomial kernels.

Under k_j(x, y) = (x^T y)^j, MMD^2 equals the squared Frobenius distance between the
j-th uncentered moment tensors E[x^{(x)j}] of the two distributions, so summing degrees
1..m-1 is zero exactly when the first m-1 moments match. Every term is a Gram matrix of
dot products; the moment tensors are never formed.

Two estimators, for two jobs:

- Unbiased (diagonal of the within-class Gram matrices excluded): for measurement. Its
  expectation is the true MMD^2, but it is not bounded below, so as a training loss a
  model can drive it to minus infinity by spreading each class (x_1 = -x_2 with large
  norm makes the within-class term arbitrarily negative). This happened in the first
  Qwen smoke runs at lambda = 10.
- Plug-in (diagonal kept): for training. It is exactly the squared distance between the
  empirical moment tensors, so it is never negative. It is biased upwards by the
  within-class spread (for j = 1 by tr(Sigma_x)/n + tr(Sigma_y)/m), which gives a weak
  pull towards shrinking each class; the activation bank keeps n large to make it small.
"""

import torch


def polynomial_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int,
    x_extra: torch.Tensor | None = None,
    y_extra: torch.Tensor | None = None,
    *,
    unbiased: bool,
) -> torch.Tensor:
    """MMD^2 between samples `x` and `y` under k(a, b) = (a^T b)^degree.

    Args:
        x: Samples of the first distribution, shape (n, d).
        y: Samples of the second distribution, shape (m, d).
        degree: Kernel degree j >= 1; the estimate targets ||E[x^{(x)j}] - E[y^{(x)j}]||_F^2.
        x_extra: Earlier samples of the first distribution, pooled with `x` but detached.
        y_extra: Earlier samples of the second distribution, pooled with `y` but detached.
        unbiased: Exclude within-class self-pairs (measurement) or keep them (training loss).

    Returns:
        Scalar estimate. The unbiased one may be negative; the plug-in one never is.
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
    if not unbiased:
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    xx = (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
    yy = (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
    return xx + yy - 2 * k_xy.mean()


def whitened_mean_gap(
    x: torch.Tensor,
    y: torch.Tensor,
    shrinkage: float,
    x_extra: torch.Tensor | None = None,
    y_extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """Squared mean difference under the shrunk inverse within-class covariance, times its mean variance.

    tau * dmu^T Sigma_reg^{-1} dmu with Sigma_reg = (1 - shrinkage) Sigma_within + shrinkage tau I
    and tau = tr(Sigma_within) / d. For isotropic Sigma it equals the Euclidean ||dmu||^2,
    but a mean gap along a direction the classes barely vary in costs up to 1/shrinkage
    times more: that is the gap a linear probe exploits even when ||dmu|| is tiny. The
    covariance is detached; the gradient flows through the means. The inverse uses the
    Woodbury identity, so no d x d matrix is formed.

    Args:
        x: Samples of the first class, shape (n, d).
        y: Samples of the second class, shape (m, d).
        shrinkage: Weight of the isotropic part, in (0, 1].
        x_extra: Earlier detached samples of the first class.
        y_extra: Earlier detached samples of the second class.
    """
    if not 0 < shrinkage <= 1:
        raise ValueError(f"shrinkage must be in (0, 1], got {shrinkage}")
    if x_extra is not None:
        x = torch.cat([x, x_extra.detach()])
    if y_extra is not None:
        y = torch.cat([y, y_extra.detach()])
    gap = x.mean(0) - y.mean(0)
    centred = torch.cat([x - x.mean(0), y - y.mean(0)]).detach()
    n, d = centred.shape
    tau = centred.square().sum() / (n * d)
    c = shrinkage * tau
    b = centred * ((1 - shrinkage) / n) ** 0.5
    inner = c * torch.eye(n, device=b.device, dtype=b.dtype) + b @ b.T
    solved = (gap - b.T @ torch.linalg.solve(inner, b @ gap)) / c
    return tau * (gap @ solved)


def moment_penalty(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    m: int,
    bases: torch.Tensor | None = None,
    harmful_extra: torch.Tensor | None = None,
    harmless_extra: torch.Tensor | None = None,
    whiten_shrinkage: float = 0.0,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """D_m for training: plug-in estimates summed over layers, averaged over token positions.

    D_m = sum_{j=1}^{m-1} MMD^2_{k_j}. Degree 1 (the class means, which difference-in-means
    abliteration reads) uses the full space; degrees >= 2 use the per-layer `bases`
    subspace when given, where the moment estimates are not dominated by noise. The
    plug-in estimator keeps the loss bounded below (see the module docstring).

    Args:
        harmful: Activations, shape (layers, positions, n, d), already scaled.
        harmless: Activations, shape (layers, positions, n', d), already scaled.
        m: Lowest moment allowed to differ; degrees 1..m-1 are penalised (m >= 2).
        bases: Orthonormal columns per layer for degrees >= 2, shape (layers, d, k).
        harmful_extra: Detached earlier harmful activations, shape (layers, positions, q, d).
        harmless_extra: Detached earlier harmless activations, shape (layers, positions, q', d).
        whiten_shrinkage: If > 0, degree 1 uses `whitened_mean_gap` with this shrinkage
            instead of the Euclidean mean gap.

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
                harmful_now = _select(harmful, layer, position, basis)
                harmless_now = _select(harmless, layer, position, basis)
                harmful_old = None if harmful_extra is None else _select(harmful_extra, layer, position, basis)
                harmless_old = None if harmless_extra is None else _select(harmless_extra, layer, position, basis)
                if degree == 1 and whiten_shrinkage > 0:
                    term = whitened_mean_gap(harmful_now, harmless_now, whiten_shrinkage, harmful_old, harmless_old)
                else:
                    term = polynomial_mmd2(harmful_now, harmless_now, degree, harmful_old, harmless_old, unbiased=False)
                total = total + term
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
