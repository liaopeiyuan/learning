# Fourier Neural Operator (baseline) with PyTorch
# =================================================
# This script is a simplified rewrite of the original JAX U-NO showcase. It illustrates the *baseline*
# Fourier Neural Operator (FNO) described by Li et al., *ICLR 2021* = see Figure 2(b) and Eq. (4) in the
# paper. The two requested changes are fulfilled:
#   1.  **Switched the framework from JAX to PyTorch**
#   2.  **Replaced the U-NO architecture with the plain FNO (no U-Net skip-connections)**
#
# The rest of the script (dataset generation, naive MLP baseline, training loop, visualisation) is kept
# very close to the original so comparisons remain meaningful.
# ------------------------------------------------------------

from __future__ import annotations
import time
from typing import Tuple, Dict
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Helper aliases & utilities
# -----------------------------------------------------------------------------
Tensor = torch.Tensor

# Check for available device (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
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
    """2-D spectral convolution layer with a *limited* number of Fourier modes.

    Implements Eq. (4) of the paper: hat u(k) = R(k) · hat v(k).
    Only the lowest *modes* are learned, higher frequencies are zeroed out.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of Fourier modes along each dimension

        # Complex weights for learned modes = initialise with normal distribution
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
            * scale
        )

    def compl_mul2d(self, input: Tensor, weight: Tensor) -> Tensor:
        # (b, c_in, h, w), weight (c_in, c_out, m, m)
        b, c_in, h, w = input.shape
        m = self.modes
        out_ft = torch.zeros(
            b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=input.device
        )

        # top-left corner              (k_x < m, k_y < m)
        out_ft[:, :, :m, :m] = torch.einsum(
            "bcih,coih->boih",
            input[:, :, :m, :m],
            weight,  # type: ignore[misc]
        )
        return out_ft

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        # Fourier transform
        x_ft = fft.rfft2(x, norm="ortho")
        # Apply learned complex weights on low-frequency modes
        out_ft = self.compl_mul2d(x_ft, self.weight)
        # Inverse FFT to spatial domain
        x = fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return x


class FNO2d(nn.Module):
    """Baseline 2-D FNO = stack of spectral conv + pointwise linear layers."""

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

        self.spectral_layers = nn.ModuleList()
        self.w_local = nn.ModuleList()
        for _ in range(depth):
            self.spectral_layers.append(SpectralConv2d(width, width, modes))
            self.w_local.append(nn.Conv2d(width, width, 1))
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        # x: (b, c_in, h, w)
        x = self.lift(x)
        for k in range(self.depth):
            v = self.spectral_layers[k](x)
            x = v + self.w_local[k](x)
            x = F.gelu(x)
        x = self.proj(x)
        return x


# -----------------------------------------------------------------------------
#  Complex synthetic dataset (unchanged, now outputs torch tensors)
# -----------------------------------------------------------------------------


def make_complex_dataset(
    seed: int, n_samples: int, n: int = 64
) -> Tuple[Tensor, Tensor]:
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")

    def generate_structured_input() -> np.ndarray:
        low_freq = rng.normal(size=(8, 8, 1))
        low_freq = np.stack(
            [np.kron(low_freq[:, :, 0], np.ones((n // 8, n // 8)))], axis=-1
        )  # crude resize without scipy
        high_freq = 0.3 * rng.normal(size=(n, n, 1))
        return low_freq + high_freq

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

    a = np.stack([generate_structured_input() for _ in range(n_samples)], axis=0)
    u = np.stack([apply_operator(a[i]) for i in range(n_samples)], axis=0)

    # to torch
    a_t = torch.from_numpy(a.astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)
    u_t = torch.from_numpy(u.astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)
    return a_t, u_t


# -----------------------------------------------------------------------------
#  Naive MLP baseline = kept for comparison (now Torch)
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
#  Training & evaluation helpers
# -----------------------------------------------------------------------------


def train(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    a: Tensor,
    u: Tensor,
    steps: int = 300,
    batch: int = 8,
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
            print(f"Step {step:3d} | loss = {loss.item():.4e}")


def evaluate(model: nn.Module, a_test: Tensor, u_test: Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(a_test)
        mse = F.mse_loss(pred, u_test).item()
        err = torch.abs(pred - u_test)
        rel = (err / (u_test.abs() + 1e-8)).mean().item()
        mx = err.max().item()
    return {"mse": mse, "relative_error": rel, "max_error": mx}


# -----------------------------------------------------------------------------
#  Visualisation (identical to original, now expects torch tensors)
# -----------------------------------------------------------------------------


def visualize(a_s: Tensor, u_true: Tensor, mlp_pred: Tensor, fno_pred: Tensor):
    a_s_np = a_s.detach().cpu().numpy()
    u_true_np = u_true.detach().cpu().numpy()
    mlp_pred_np = mlp_pred.detach().cpu().numpy()
    fno_pred_np = fno_pred.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].imshow(a_s_np[0, 0], cmap="viridis")
    axes[0, 0].set_title("Input Field")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(u_true_np[0, 0], cmap="RdBu_r")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(mlp_pred_np[0, 0], cmap="RdBu_r")
    axes[0, 2].set_title("MLP Prediction")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(fno_pred_np[0, 0], cmap="RdBu_r")
    axes[1, 0].set_title("FNO Prediction")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    mlp_err = np.abs(mlp_pred_np[0, 0] - u_true_np[0, 0])
    im4 = axes[1, 1].imshow(mlp_err, cmap="hot")
    axes[1, 1].set_title(f"MLP Error (max={mlp_err.max():.3f})")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    fno_err = np.abs(fno_pred_np[0, 0] - u_true_np[0, 0])
    im5 = axes[1, 2].imshow(fno_err, cmap="hot")
    axes[1, 2].set_title(f"FNO Error (max={fno_err.max():.3f})")
    axes[1, 2].axis("off")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
#  Main comparison
# -----------------------------------------------------------------------------


def main():
    n = 64
    train_samples = 256
    test_samples = 64
    steps = 1000  # Increased from 300
    batch = 8

    print("=" * 70)
    print("Baseline FNO vs MLP = PyTorch edition")
    print("=" * 70)

    a_train, u_train = make_complex_dataset(42, train_samples, n)
    a_test, u_test = make_complex_dataset(123, test_samples, n)

    # ---------------- MLP baseline ----------------
    # Reduced MLP size to match FNO parameter count (~1M params)
    mlp = FlattenMLP(n * n, [256, 512, 256], n * n).to(DEVICE)
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
    train(fno, opt_fno, a_train, u_train, steps=steps, batch=batch)
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
    fig.savefig("fno_vs_mlp_predictions.png", dpi=150, bbox_inches="tight")
    print("Saved figure to 'fno_vs_mlp_predictions.png'")


def test_fno_multiscale():
    """Test FNO's ability to handle different scales/resolutions."""
    print("\n" + "=" * 70)
    print("Testing FNO Multi-scale Capabilities")
    print("=" * 70)

    # Train on one resolution
    n_train = 64
    n_samples = 256
    a_train, u_train = make_complex_dataset(42, n_samples, n_train)

    # Create FNO model
    fno = FNO2d(in_channels=1, out_channels=1, width=32, depth=4, modes=16).to(DEVICE)
    print(f"Training FNO on {n_train}x{n_train} resolution...")

    # Train
    optimizer = torch.optim.Adam(fno.parameters(), lr=3e-4)
    train(fno, optimizer, a_train, u_train, steps=500, batch=8)

    # Test on multiple resolutions
    test_resolutions = [32, 64, 128]
    results = {}

    fig, axes = plt.subplots(4, len(test_resolutions), figsize=(15, 16))

    for i, n_test in enumerate(test_resolutions):
        print(f"\nTesting on {n_test}x{n_test} resolution...")

        # Generate test data at different resolution
        a_test, u_test = make_complex_dataset(123, 16, n_test)

        # For resolutions different from training, we need to interpolate
        if n_test != n_train:
            # Interpolate input to training resolution
            a_interp = F.interpolate(
                a_test, size=(n_train, n_train), mode="bilinear", align_corners=False
            )

            # Get prediction at training resolution
            with torch.no_grad():
                pred_train_res = fno(a_interp)

            # Interpolate prediction back to test resolution
            pred = F.interpolate(
                pred_train_res,
                size=(n_test, n_test),
                mode="bilinear",
                align_corners=False,
            )
        else:
            with torch.no_grad():
                pred = fno(a_test)

        # Evaluate
        res = (
            evaluate(fno, a_test, u_test)
            if n_test == n_train
            else {
                "mse": F.mse_loss(pred, u_test).item(),
                "relative_error": (torch.abs(pred - u_test) / (u_test.abs() + 1e-8))
                .mean()
                .item(),
                "max_error": torch.abs(pred - u_test).max().item(),
            }
        )
        results[n_test] = res

        # Visualize first sample
        a_np = a_test[0, 0].cpu().numpy()
        u_np = u_test[0, 0].cpu().numpy()
        pred_np = pred[0, 0].detach().cpu().numpy()
        error_np = np.abs(pred_np - u_np)

        # Input
        im0 = axes[0, i].imshow(a_np, cmap="viridis")
        axes[0, i].set_title(f"Input ({n_test}x{n_test})")
        axes[0, i].axis("off")
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046)

        # Ground Truth
        im1 = axes[1, i].imshow(u_np, cmap="RdBu_r")
        axes[1, i].set_title(f"Ground Truth")
        axes[1, i].axis("off")
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046)

        # Prediction
        im2 = axes[2, i].imshow(pred_np, cmap="RdBu_r")
        axes[2, i].set_title(f"FNO Prediction")
        axes[2, i].axis("off")
        plt.colorbar(im2, ax=axes[2, i], fraction=0.046)

        # Error
        im3 = axes[3, i].imshow(error_np, cmap="hot")
        axes[3, i].set_title(f"Error (max={error_np.max():.3f})")
        axes[3, i].axis("off")
        plt.colorbar(im3, ax=axes[3, i], fraction=0.046)

    plt.tight_layout()
    plt.savefig("fno_multiscale_test.png", dpi=150, bbox_inches="tight")
    print("\nSaved multi-scale test figure to 'fno_multiscale_test.png'")

    # Print results summary
    print("\nMulti-scale Results Summary:")
    print("-" * 50)
    for res_size, metrics in results.items():
        print(
            f"{res_size}x{res_size}: MSE={metrics['mse']:.4e}, RelErr={metrics['relative_error']:.2%}, MaxErr={metrics['max_error']:.3f}"
        )


if __name__ == "__main__":
    main()
    test_fno_multiscale()
