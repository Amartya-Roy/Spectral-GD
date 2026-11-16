"""
Spectral-GD and Muon optimizer implementations.

- SpectralGD: Deterministic, momentum-free polar-factor update via exact SVD:
      W_{k+1} = W_k - lr * scale * Polar(G_k),   where Polar(G_k) = U V^T from SVD(G_k) = U S V^T
  For 1D non-bias parameters (e.g., LayerNorm weights), applies the normalized direction G_k / ||G_k||_2.
- Muon: Momentum-based polar update using 5-step quintic Newton-Schulz iteration.
"""
import torch


def exact_polar_factor(G: torch.Tensor) -> torch.Tensor:
    """Compute the exact polar factor U @ Vh of a 2D matrix G via thin SVD."""
    orig_dtype = G.dtype
    U, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return (U @ Vh).to(orig_dtype)


def newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """5-step quintic Newton-Schulz iteration for approximate orthogonalization."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class SpectralGD(torch.optim.Optimizer):
    """
    Momentum-free Spectral Gradient Descent (exact polar-factor update).
    Updates all passed parameters (2D matrices via exact SVD polar factor;
    1D non-bias parameters via l2-normalized gradient). Biases are excluded by the caller.
    """
    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0, rms_scale: bool = True):
        defaults = dict(lr=lr, weight_decay=weight_decay, rms_scale=rms_scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            rms_scale = group["rms_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                if g.ndim == 2:
                    if g.size(0) == 3 * g.size(1):
                        # Fused QKV projection: orthogonalize Q, K, V blocks separately
                        u = torch.cat([exact_polar_factor(g_i) for g_i in g.split(g.size(1), dim=0)], dim=0)
                        scale = (g.size(1) ** 0.5) if rms_scale else 1.0
                    else:
                        u = exact_polar_factor(g)
                        scale = (max(g.size(0), g.size(1)) ** 0.5) if rms_scale else 1.0
                    p.add_(u, alpha=-lr * scale)
                elif g.ndim > 2:
                    g_2d = g.reshape(g.size(0), -1)
                    u = exact_polar_factor(g_2d).view_as(g)
                    scale = (max(g_2d.size(0), g_2d.size(1)) ** 0.5) if rms_scale else 1.0
                    p.add_(u, alpha=-lr * scale)
                else:
                    # 1D non-bias parameters (e.g., LayerNorm weights)
                    u = g / (g.norm() + 1e-8)
                    scale = (g.numel() ** 0.5) if rms_scale else 1.0
                    p.add_(u, alpha=-lr * scale)

        return loss


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer: Nesterov SGD-momentum orthogonalized via 5-step Newton-Schulz iteration.
    """
    def __init__(
        self,
        params,
        lr: float = 0.005,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if nesterov else buf

                if g_eff.ndim == 2:
                    if g_eff.size(0) == 3 * g_eff.size(1):
                        u = torch.cat(
                            [newton_schulz_5(g_i, steps=ns_steps) for g_i in g_eff.split(g_eff.size(1), dim=0)],
                            dim=0,
                        )
                        scale = g_eff.size(1) ** 0.5
                    else:
                        u = newton_schulz_5(g_eff, steps=ns_steps)
                        scale = max(g_eff.size(0), g_eff.size(1)) ** 0.5
                    p.add_(u, alpha=-lr * scale)
                else:
                    u = g_eff / (g_eff.norm() + 1e-8)
                    scale = g_eff.numel() ** 0.5
                    p.add_(u, alpha=-lr * scale)

        return loss
