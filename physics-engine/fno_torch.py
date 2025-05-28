from __future__ import annotations

import time
from typing import Tuple, Dict

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


Tensor = torch.Tensor

# -----------------------------------------------------------------------------
#  Device ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# -----------------------------------------------------------------------------
#  Fourier layer (2-D, real) = baseline FNO
# -----------------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    """2-D spectral convolution with a limited number of Fourier modes.

    Implements Eq.(4) of the FNO paper for **real-valued** inputs.
    The first `modes` x `modes` Fourier coefficients are learned; others are
    left untouched (i.e. multiplied by zero).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # learn up to this many spectral modes per dim

        # Complex weights for the learned modes
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
            * scale
        )

    def compl_mul2d(self, x_ft: Tensor) -> Tensor:
        """Multiply in Fourier domain (batched complex matmul).

        x_ft: (batch, C_in, H, W_half+1)
        returns (batch, C_out, H, W_half+1)
        """
        b, c_in, h, w_half = x_ft.shape
        m = min(self.modes, h, w_half)  # clip to grid Nyquist

        weight = self.weight[:, :, :m, :m]  # (C_in, C_out, m, m)

        # allocate output & perform batched matmul on the low-freq block
        out_ft = torch.zeros(
            b,
            self.out_channels,
            h,
            w_half,
            dtype=torch.cfloat,
            device=x_ft.device,
        )
        out_ft[:, :, :m, :m] = torch.einsum(
            "bcij,coij->boij",
            x_ft[:, :, :m, :m],
            weight,
        )
        return out_ft

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) real
        x_ft = fft.rfft2(x, norm="ortho")  # (B, C, H, W//2+1)
        out_ft = self.compl_mul2d(x_ft)
        x = fft.irfft2(out_ft, s=x.shape[-2:], norm="ortho")  # back to real
        return x


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        depth: int = 4,
        modes: int = 12,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.proj = nn.Conv2d(width, out_channels, 1)

        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(depth)]
        )
        self.w_local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.lift(x)
        for spec, pw in zip(self.spectral_layers, self.w_local):
            v = spec(x)
            x = F.gelu(v + pw(x))
        return self.proj(x)


# -----------------------------------------------------------------------------
#  Synthetic data - Gaussian random field + toy operator
# -----------------------------------------------------------------------------
def make_complex_dataset(
    seed: int, n_samples: int, n: int = 64
) -> Tuple[Tensor, Tensor]:
    """Return (a, u) with shapes:
    a - (N, 1, n, n)
    u - (N, 1, n, n)
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")

    def generate_grf_2d(alpha: float = 2.0, tau: float = 3.0) -> np.ndarray:
        """Spectral-domain GRF with Hermitian symmetry → real field."""
        kx = np.fft.fftfreq(n, d=1 / n).reshape(-1, 1)
        ky = np.fft.rfftfreq(n, d=1 / n).reshape(1, -1)
        k_norm = np.sqrt(kx**2 + ky**2)
        k_norm[0, 0] = 1.0

        power = (1 + (k_norm**2) / tau**2) ** (-alpha / 2)
        power[0, 0] = 0.0  # zero-mean field

        noise_real = rng.normal(0, 1, (n, n // 2 + 1))
        noise_imag = rng.normal(0, 1, (n, n // 2 + 1))
        noise_imag[:, 0] = 0.0  # enforce Hermitian symmetry on borders
        if n % 2 == 0:
            noise_imag[n // 2] = 0.0

        coeffs = (noise_real + 1j * noise_imag) * np.sqrt(power)
        field = np.fft.irfft2(coeffs, s=(n, n))
        field -= field.mean()
        field /= field.std()
        return field[..., None]

    def apply_operator(field: np.ndarray) -> np.ndarray:
        mean_val = field.mean()
        std_val = field.std()
        u1 = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
        u2 = np.sin(4 * np.pi * xx) * np.sin(4 * np.pi * yy)
        u3 = np.cos(6 * np.pi * (xx + yy))
        field_fft = np.fft.fft2(field[..., 0])
        low_freq_power = np.abs(field_fft[:5, :5]).mean()
        u = (
            0.5 * u1
            + 0.3 * u2 * mean_val
            + 0.2 * u3 * std_val
            + 0.1 * low_freq_power
            + 0.1 * field[..., 0]
        )
        return u[..., None]

    a = np.stack([generate_grf_2d() for _ in range(n_samples)], axis=0)
    u = np.stack([apply_operator(a[i]) for i in range(n_samples)], axis=0)

    # → torch
    a_t = torch.from_numpy(a.astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)
    u_t = torch.from_numpy(u.astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)
    return a_t, u_t


# -----------------------------------------------------------------------------
#  Naïve MLP baseline
# -----------------------------------------------------------------------------
class FlattenMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], output_dim: int):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.GELU()]
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = x.view(b, -1)
        x = self.net(x)
        x = x.view(b, 1, h, w)
        return x


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------
def train(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    a: Tensor,
    u: Tensor,
    steps: int,
    batch: int,
):
    model.train()
    n = a.shape[0]
    for step in range(steps):
        idx = torch.arange(batch) + (step * batch) % (n - batch)
        batch_a, batch_u = a[idx], u[idx]
        optimiser.zero_grad()
        pred = model(batch_a)
        loss = F.mse_loss(pred, batch_u)
        loss.backward()
        optimiser.step()
        if step % 100 == 0:
            print(f"Step {step:4d} | loss = {loss.item():.4e}")


@torch.no_grad()
def evaluate(model: nn.Module, a_test: Tensor, u_test: Tensor) -> Dict[str, float]:
    model.eval()
    pred = model(a_test)
    mse = F.mse_loss(pred, u_test).item()
    err = torch.abs(pred - u_test)
    rel = (err / (u_test.abs() + 1e-8)).mean().item()
    mx = err.max().item()
    return {"mse": mse, "relative_error": rel, "max_error": mx}


# -----------------------------------------------------------------------------
#  Visualisation
# -----------------------------------------------------------------------------
def visualize(a_s: Tensor, u_true: Tensor, mlp_pred: Tensor, fno_pred: Tensor):
    a_s_np = a_s.detach().cpu().numpy()
    u_true_np = u_true.detach().cpu().numpy()
    mlp_pred_np = mlp_pred.detach().cpu().numpy()
    fno_pred_np = fno_pred.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row
    axes[0, 0].imshow(a_s_np[0, 0], cmap="viridis")
    axes[0, 0].set_title("Input Field")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mlp_pred_np[0, 0], cmap="RdBu_r")
    axes[0, 1].set_title("MLP Prediction")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(fno_pred_np[0, 0], cmap="RdBu_r")
    axes[0, 2].set_title("FNO Prediction")
    axes[0, 2].axis("off")

    # Bottom row
    axes[1, 0].imshow(u_true_np[0, 0], cmap="RdBu_r")
    axes[1, 0].set_title("Ground Truth")
    axes[1, 0].axis("off")

    mlp_err = np.abs(mlp_pred_np - u_true_np)[0, 0]
    axes[1, 1].imshow(mlp_err, cmap="hot")
    axes[1, 1].set_title(f"MLP Error (max={mlp_err.max():.3f})")
    axes[1, 1].axis("off")

    fno_err = np.abs(fno_pred_np - u_true_np)[0, 0]
    axes[1, 2].imshow(fno_err, cmap="hot")
    axes[1, 2].set_title(f"FNO Error (max={fno_err.max():.3f})")
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
#  Main comparison
# -----------------------------------------------------------------------------
def main():
    n = 64
    train_samples = 25600
    test_samples = 64
    steps = 1000
    fno_steps = 5000
    batch = 32

    print("=" * 72)
    print("Baseline FNO vs MLP")
    print("=" * 72)

    a_train, u_train = make_complex_dataset(42, train_samples, n)
    a_test, u_test = make_complex_dataset(123, test_samples, n)

    # ---------------- MLP baseline ----------------
    mlp = FlattenMLP(n * n, [256, 256, 256], n * n).to(DEVICE)
    print(f"MLP parameters: {count_params(mlp):,}")

    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    t0 = time.time()
    train(mlp, opt_mlp, a_train, u_train, steps=steps, batch=batch)
    mlp_time = time.time() - t0
    mlp_res = evaluate(mlp, a_test, u_test)

    # ---------------- FNO ------------------------
    fno = FNO2d(in_channels=1, out_channels=1, width=32, depth=4, modes=16).to(DEVICE)
    print(f"FNO parameters: {count_params(fno):,}")

    opt_fno = torch.optim.Adam(fno.parameters(), lr=3e-4)
    t0 = time.time()
    print(f"Training FNO for {fno_steps} steps...")
    train(fno, opt_fno, a_train, u_train, steps=fno_steps, batch=batch)
    fno_time = time.time() - t0
    fno_res = evaluate(fno, a_test, u_test)

    # ---------------- Summary --------------------
    print("\nResults (test set):")
    print(
        f"MLP = MSE: {mlp_res['mse']:.4e} | RelErr: {mlp_res['relative_error']:.2%} | MaxErr: {mlp_res['max_error']:.3f}"
    )
    print(
        f"FNO = MSE: {fno_res['mse']:.4e} | RelErr: {fno_res['relative_error']:.2%} | MaxErr: {fno_res['max_error']:.3f}"
    )
    print(f"Training times = MLP: {mlp_time:.1f}s | FNO: {fno_time:.1f}s")

    # ---------------- Visualisation --------------
    with torch.no_grad():
        mlp_pred = mlp(a_test[:1])
        fno_pred = fno(a_test[:1])

    fig = visualize(a_test[:1], u_test[:1], mlp_pred, fno_pred)
    fig.savefig("fno_vs_mlp_predictions_fixed.png", dpi=150, bbox_inches="tight")
    print("Saved figure to 'fno_vs_mlp_predictions_fixed.png'")

    return fno


# -----------------------------------------------------------------------------
#  Multi-scale test - no more down/upsampling
# -----------------------------------------------------------------------------
@torch.no_grad()
def test_fno_multiscale(fno: nn.Module):
    print("\n" + "=" * 72)
    print("Testing FNO Multi-scale Capabilities (native grids)")
    print("=" * 72)

    test_resolutions = [32, 64, 128]
    results = {}

    fig, axes = plt.subplots(4, len(test_resolutions), figsize=(15, 16))

    for i, n_test in enumerate(test_resolutions):
        print(f"\nTesting on {n_test}x{n_test} grid ...")
        a_test, u_test = make_complex_dataset(999 + n_test, 16, n_test)

        pred = fno(a_test)

        # Metrics
        res = evaluate(fno, a_test, u_test)
        results[n_test] = res

        # Visualise first sample
        a_np = a_test[0, 0].cpu().numpy()
        u_np = u_test[0, 0].cpu().numpy()
        pred_np = pred[0, 0].cpu().numpy()
        err_np = np.abs(pred_np - u_np)

        for row, img, title in zip(
            range(4),
            [a_np, u_np, pred_np, err_np],
            [
                f"Input ({n_test}x{n_test})",
                "Ground Truth",
                "FNO Prediction",
                f"Error (max={err_np.max():.3f})",
            ],
        ):
            axes[row, i].imshow(
                img,
                cmap="viridis" if row == 0 else "RdBu_r" if row in [1, 2] else "hot",
            )
            axes[row, i].set_title(title)
            axes[row, i].axis("off")

    plt.tight_layout()
    plt.savefig("fno_multiscale_test_fixed.png", dpi=150, bbox_inches="tight")
    print("\nSaved multi-scale figure to 'fno_multiscale_test_fixed.png'")

    # Results table
    print("\nMulti-scale Results Summary:")
    print("-" * 50)
    for res_size, metrics in results.items():
        print(
            f"{res_size}²: MSE={metrics['mse']:.4e}, "
            f"RelErr={metrics['relative_error']:.2%}, "
            f"MaxErr={metrics['max_error']:.3f}"
        )


if __name__ == "__main__":
    fno_model = main()
    test_fno_multiscale(fno_model)
