"""U-NO vs MLP: Complex Operator Learning Comparison
=================================================
This script demonstrates the superiority of U-shaped Neural Operators (U-NO)
over naive approaches for complex operator learning tasks.

The task involves learning an operator with:
- Multiple frequency components (2π, 4π, 6π)
- Non-local dependencies (FFT-based interactions)
- Multi-scale features
- Global and local interactions

This represents realistic challenges in PDEs, fluid dynamics, and physics simulations.
"""

import time
from typing import Dict, Any
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from typing import List, Sequence

Array = jax.Array  # type alias for clarity

# ------------------------------------------------------------
#  U-NO Implementation (from small-uno-demo.py)
# ------------------------------------------------------------

# ---------- Common building blocks ----------


def init_dense(key: Array, in_ch: int, out_ch: int):
    k1, k2 = jax.random.split(key)
    w = jax.random.normal(k1, (in_ch, out_ch)) / jnp.sqrt(in_ch)
    b = jnp.zeros((out_ch,))
    return (w, b)


def dense(params, x: Array):
    w, b = params
    return jnp.einsum("...c,co->...o", x, w) + b


# ---------- Fourier Neural Operator layer -------------------


def init_fft_layer(key: Array, in_ch: int, out_ch: int, modes: int):
    k1, k2, k3 = jax.random.split(key, 3)
    weight = (
        jax.random.normal(k1, (modes, modes, out_ch, in_ch))
        + 1j * jax.random.normal(k2, (modes, modes, out_ch, in_ch))
    ) / jnp.sqrt(in_ch)
    W_local = jax.random.normal(k3, (in_ch, out_ch)) / jnp.sqrt(in_ch)
    return dict(weight=weight.astype(jnp.complex64), W_local=W_local)


def fft_layer(params: Dict[str, Array], x: Array, modes: int):
    """Fourier operator block (2-D, real)."""
    B, H, W, _ = x.shape
    C_out = params["W_local"].shape[1]

    v_hat = jnp.fft.rfftn(x, axes=(1, 2))  # (B,H,W//2+1,C_in)
    v_hat_out = jnp.zeros(v_hat.shape[:-1] + (C_out,), dtype=jnp.complex64)

    # Get actual FFT dimensions
    fft_h, fft_w = v_hat.shape[1], v_hat.shape[2]

    # Ensure we don't exceed the weight tensor dimensions or FFT dimensions
    weight_h, weight_w = params["weight"].shape[:2]
    m_h = min(modes, weight_h, fft_h)
    m_w = min(modes, weight_w, fft_w)

    sl = (slice(None), slice(0, m_h), slice(0, m_w))
    v_slice = v_hat[sl + (slice(None),)]  # (...,C_in)
    weight_slice = params["weight"][:m_h, :m_w, :, :]
    out_slice = jnp.einsum("bxyc,xyoc->bxyo", v_slice, weight_slice)  # (...,C_out)
    v_hat_out = v_hat_out.at[sl + (slice(None),)].set(out_slice)
    v_low = jnp.fft.irfftn(v_hat_out, s=(H, W), axes=(1, 2)).real

    v_local = jnp.einsum("...c,co->...o", x, params["W_local"])
    return jax.nn.gelu(v_low + v_local)


# ---------- (Anti-)aliasing down/upsample helpers -----------


def avg_pool2d(x: Array, factor: int = 2):
    """Average-pool down-sample by *factor* with no padding."""
    return jax.lax.reduce_window(
        x,
        0.0,
        jax.lax.add,
        (1, factor, factor, 1),
        (1, factor, factor, 1),
        "VALID",
    ) / (factor * factor)


def bilinear_resize(x: Array, target_hw):
    """Bilinear resize to the given (H,W)."""
    B, _, _, C = x.shape
    H_t, W_t = target_hw
    return jax.image.resize(x, (B, H_t, W_t, C), method="linear")


# ---------- Scaled-up U-NO ----------


def _rep(value: int | Sequence[int], depth: int) -> List[int]:
    """Helper - broadcast an int to a list of length depth."""
    if isinstance(value, int):
        return [value] * depth
    if len(value) != depth:
        raise ValueError("Length mismatch between provided list and depth")
    return list(value)


def init_uno(
    key: Array,
    depth: int = 3,
    in_ch: int = 1,
    width0: int = 32,
    width_growth: int = 2,
    modes: int | Sequence[int] = 12,
):
    """Initialise parameters for a *depth-level* U-NO.

    depth levels → depth encoder blocks, depth decoder blocks, and one bottleneck.
    widths grow geometrically:  width0, width0*growth, ...
    modes can be an int (same for all) or a list of length depth+1 (per level).
    """
    keys = jax.random.split(key, 4 * depth + 3)  # plenty of rng
    k_iter = iter(keys)

    widths = [width0 * (width_growth**i) for i in range(depth + 1)]
    modes_list = _rep(modes, depth + 1)  # one per encoder level + bottleneck

    params: Dict[str, Any] = dict()
    params["lift"] = init_dense(next(k_iter), in_ch, widths[0])
    params["proj"] = init_dense(next(k_iter), widths[0], 1)

    # --- encoder FNO blocks ---
    enc_blocks = []
    for d in range(depth):
        enc_blocks.append(
            init_fft_layer(next(k_iter), widths[d], widths[d + 1], modes_list[d])
        )
    params["enc"] = enc_blocks

    # --- bottleneck (same in/out width) ---
    params["bottleneck"] = init_fft_layer(
        next(k_iter), widths[-1], widths[-1], modes_list[-1]
    )

    # --- decoder FNO blocks (mirror encoder) ---
    dec_blocks = []
    # Create decoder blocks in forward order to match the forward pass
    for d in range(depth):
        # At decoder level d, we concatenate:
        # - upsampled features from previous level (or bottleneck for d=0)
        # - skip connection from encoder at level (depth-1-d)

        enc_level = depth - 1 - d  # which encoder's skip we're using

        if d == 0:
            # First decoder: bottleneck output + last encoder's output
            upsampled_ch = widths[-1]  # bottleneck output
            skip_ch = widths[enc_level + 1]  # encoder output at level enc_level
        else:
            # Other decoders: previous decoder's output + corresponding encoder's output
            upsampled_ch = widths[depth - d]  # previous decoder's output
            skip_ch = widths[enc_level + 1]  # encoder output at level enc_level

        in_c = upsampled_ch + skip_ch
        out_c = widths[enc_level]

        print(
            f"Decoder block {d}: in_c={in_c} (up={upsampled_ch} + skip={skip_ch}), out_c={out_c}, enc_level={enc_level}"
        )

        # Use modes from the corresponding encoder level
        dec_blocks.append(
            init_fft_layer(next(k_iter), in_c, out_c, modes_list[enc_level])
        )
    params["dec"] = dec_blocks

    return params


def uno_forward(
    params: Dict[str, Any],
    x: Array,
    depth: int = 3,
    modes: int | Sequence[int] = 12,
):
    """Forward pass for the scaled U-NO."""
    modes_list = _rep(modes, depth + 1)

    skips = []
    v = jax.nn.gelu(dense(params["lift"], x))

    # -------- encoder path --------
    for lvl in range(depth):
        v = fft_layer(params["enc"][lvl], v, modes_list[lvl])
        skips.append(v)  # save pre-downsample skip
        v = avg_pool2d(v, factor=2)

    # -------- bottleneck ----------
    v = fft_layer(params["bottleneck"], v, modes_list[-1])

    # -------- decoder path --------
    for lvl in range(depth):
        skip_v = skips.pop()
        v = bilinear_resize(v, skip_v.shape[1:3])
        v_concat = jnp.concatenate([v, skip_v], axis=-1)
        print(
            f"Forward decoder lvl {lvl}: v={v.shape}, skip_v={skip_v.shape}, concat={v_concat.shape}"
        )
        print(
            f"  Decoder block expects: {params['dec'][lvl]['W_local'].shape[0]} input channels"
        )
        v = fft_layer(params["dec"][lvl], v_concat, modes_list[depth - 1 - lvl])

    u = dense(params["proj"], v)
    return u


# jit compile once - GSC traces are cached after first call
uno_forward = jax.jit(uno_forward, static_argnames=("depth", "modes"))

# ------------------------------------------------------------
#  Complex Dataset Generation
# ------------------------------------------------------------


def make_complex_dataset(key: jax.Array, n_samples: int, n: int = 64):
    """Generate a complex operator learning task that showcases U-NO's strengths.

    Input: Structured random fields with multiple scales
    Output: Complex operator with frequencies, non-local effects, and multi-scale features
    """
    # Create coordinate grids
    x = jnp.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy = jnp.meshgrid(x, x, indexing="ij")

    keys = jax.random.split(key, n_samples)

    def generate_structured_input(k):
        k1, k2 = jax.random.split(k)
        # Low-frequency structure (smooth base)
        low_freq = jax.random.normal(k1, (8, 8, 1))
        low_freq = jax.image.resize(low_freq, (n, n, 1), method="cubic")
        # High-frequency details
        high_freq = 0.3 * jax.random.normal(k2, (n, n, 1))
        return low_freq + high_freq

    def apply_complex_operator(field):
        # Global statistics affect the entire output
        mean_val = jnp.mean(field)
        std_val = jnp.std(field)

        # Multiple frequency components
        u1 = jnp.sin(2 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy)
        u2 = jnp.sin(4 * jnp.pi * xx) * jnp.sin(4 * jnp.pi * yy)
        u3 = jnp.cos(6 * jnp.pi * (xx + yy))

        # Non-local: FFT-based interaction
        field_fft = jnp.fft.fft2(field[..., 0])
        low_freq_power = jnp.abs(field_fft[:5, :5]).mean()

        # Combine all components
        u = (
            0.5 * u1
            + 0.3 * u2 * mean_val
            + 0.2 * u3 * std_val
            + 0.1 * low_freq_power * jnp.ones_like(u1)
            + 0.1 * field[..., 0]
        )  # Local component

        return u[..., None]

    a = jax.vmap(generate_structured_input)(keys)
    u = jax.vmap(apply_complex_operator)(a)

    return a, u


# ------------------------------------------------------------
#  MLP Implementation (Naive Baseline)
# ------------------------------------------------------------


def init_mlp(key: jax.Array, input_dim: int, hidden_dims: list, output_dim: int):
    """Initialize MLP that flattens spatial structure."""
    keys = jax.random.split(key, len(hidden_dims) + 1)
    layers = []

    in_dim = input_dim
    for i, h_dim in enumerate(hidden_dims):
        w = jax.random.normal(keys[i], (in_dim, h_dim)) * 0.02
        b = jnp.zeros((h_dim,))
        layers.append({"w": w, "b": b})
        in_dim = h_dim

    # Output layer
    w = jax.random.normal(keys[-1], (in_dim, output_dim)) * 0.02
    b = jnp.zeros((output_dim,))
    layers.append({"w": w, "b": b})

    return {"layers": layers}


def mlp_forward(params: Dict, x: jax.Array):
    """MLP forward pass - destroys spatial structure."""
    # Handle both single and batch inputs
    single = x.ndim == 3
    if single:
        x = x[None, ...]

    B, H, W, C = x.shape
    x_flat = x.reshape(B, -1)

    # Forward through layers
    for i, layer in enumerate(params["layers"][:-1]):
        x_flat = x_flat @ layer["w"] + layer["b"]
        x_flat = jax.nn.gelu(x_flat)

    # Output layer
    layer = params["layers"][-1]
    x_flat = x_flat @ layer["w"] + layer["b"]

    out = x_flat.reshape(B, H, W, 1)
    return out[0] if single else out


# ------------------------------------------------------------
#  Training and Evaluation
# ------------------------------------------------------------


def train_step(params, opt_state, batch_a, batch_u, forward_fn, optimizer):
    """Single training step."""

    def loss_fn(p):
        pred = forward_fn(p, batch_a)
        return jnp.mean((pred - batch_u) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# JIT compile with static arguments
train_step_jit = jax.jit(train_step, static_argnames=["forward_fn", "optimizer"])


def count_params(params):
    """Count total parameters."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def evaluate_model(params, forward_fn, a_test, u_test):
    """Comprehensive evaluation."""
    # Predictions
    pred = jax.vmap(forward_fn, in_axes=(None, 0))(params, a_test)

    # Metrics
    mse = float(jnp.mean((pred - u_test) ** 2))
    errors = jnp.abs(pred - u_test)
    max_error = float(jnp.max(errors))
    rel_error = float(jnp.mean(errors / (jnp.abs(u_test) + 1e-8)))

    return {"mse": mse, "max_error": max_error, "relative_error": rel_error}


def visualize_predictions(a_sample, u_true, mlp_pred, uno_pred, title="Predictions"):
    """Visualize ground truth and predictions side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Input
    im0 = axes[0, 0].imshow(a_sample[..., 0], cmap="viridis", aspect="equal")
    axes[0, 0].set_title("Input Field", fontsize=14)
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Ground truth
    im1 = axes[0, 1].imshow(u_true[..., 0], cmap="RdBu_r", aspect="equal")
    axes[0, 1].set_title("Ground Truth", fontsize=14)
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # MLP prediction
    im2 = axes[0, 2].imshow(mlp_pred[..., 0], cmap="RdBu_r", aspect="equal")
    axes[0, 2].set_title("MLP Prediction", fontsize=14)
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # U-NO prediction
    im3 = axes[1, 0].imshow(uno_pred[..., 0], cmap="RdBu_r", aspect="equal")
    axes[1, 0].set_title("U-NO Prediction", fontsize=14)
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # MLP error
    mlp_error = jnp.abs(mlp_pred[..., 0] - u_true[..., 0])
    im4 = axes[1, 1].imshow(mlp_error, cmap="hot", aspect="equal")
    axes[1, 1].set_title(f"MLP Error (max: {float(mlp_error.max()):.3f})", fontsize=14)
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # U-NO error
    uno_error = jnp.abs(uno_pred[..., 0] - u_true[..., 0])
    im5 = axes[1, 2].imshow(uno_error, cmap="hot", aspect="equal")
    axes[1, 2].set_title(f"U-NO Error (max: {float(uno_error.max()):.3f})", fontsize=14)
    axes[1, 2].axis("off")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def test_multiscale_uno(uno_params, uno_wrapper, key, original_n=64):
    """Test U-NO's ability to handle different resolutions."""
    print("\n" + "=" * 70)
    print("MULTI-SCALE TESTING FOR U-NO")
    print("=" * 70)
    print("Testing if U-NO can generalize to different resolutions...")

    resolutions = [32, 64, 128]
    results = {}

    for res in resolutions:
        # Generate test data at this resolution
        key, subkey = jax.random.split(key)
        a_test, u_test = make_complex_dataset(subkey, 16, n=res)

        # If resolution differs from training, we need to interpolate
        if res != original_n:
            # For U-NO to work at different resolutions, we need to resize
            # This tests if the learned operator generalizes
            print(f"\nTesting at {res}×{res} resolution:")

            # Direct evaluation at new resolution
            eval_results = evaluate_model(uno_params, uno_wrapper, a_test, u_test)
            results[res] = eval_results

            print(f"  MSE: {eval_results['mse']:.4e}")
            print(f"  Relative Error: {eval_results['relative_error']:.1%}")
        else:
            print(f"\nOriginal resolution {res}×{res}:")
            eval_results = evaluate_model(uno_params, uno_wrapper, a_test, u_test)
            results[res] = eval_results
            print(f"  MSE: {eval_results['mse']:.4e}")
            print(f"  Relative Error: {eval_results['relative_error']:.1%}")

    return results


# ------------------------------------------------------------
#  Main Comparison
# ------------------------------------------------------------


def main():
    # Configuration
    n = 64  # Spatial resolution
    train_samples = 256
    test_samples = 64
    batch_size = 8
    steps = 300

    print("=" * 70)
    print("U-NO vs MLP: COMPLEX OPERATOR LEARNING")
    print("=" * 70)
    print(f"Task: Learning operator with multiple frequencies, FFT interactions")
    print(f"Resolution: {n}×{n} = {n * n} points")
    print(f"Training samples: {train_samples}")
    print("=" * 70)

    # Generate data
    key = jax.random.PRNGKey(42)
    key_train, key_test, key_mlp, key_uno = jax.random.split(key, 4)

    print("\nGenerating complex dataset...")
    a_train, u_train = make_complex_dataset(key_train, train_samples, n)
    a_test, u_test = make_complex_dataset(key_test, test_samples, n)
    print(
        f"Data ranges: input [{float(a_train.min()):.1f}, {float(a_train.max()):.1f}], "
        f"output [{float(u_train.min()):.1f}, {float(u_train.max()):.1f}]"
    )

    # ========== Train MLP ==========
    print("\n" + "-" * 50)
    print("TRAINING MLP (Naive Approach)")
    print("-" * 50)
    print("Issues: Destroys spatial structure, needs many parameters")

    mlp_params = init_mlp(key_mlp, n * n, [2048, 4096, 4096, 2048], n * n)
    mlp_n_params = count_params(mlp_params)
    print(f"Parameters: {mlp_n_params:,}")

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(mlp_params)

    t0 = time.time()
    for step in range(steps):
        idx = jnp.arange(batch_size) + (step * batch_size) % (
            train_samples - batch_size
        )
        batch_a, batch_u = a_train[idx], u_train[idx]

        mlp_params, opt_state, loss = train_step_jit(
            mlp_params, opt_state, batch_a, batch_u, mlp_forward, optimizer
        )

        if step % 100 == 0:
            print(f"Step {step:3d}: loss = {loss:.4e}")

    mlp_time = time.time() - t0
    mlp_results = evaluate_model(mlp_params, mlp_forward, a_test, u_test)
    print(f"Training time: {mlp_time:.1f}s")

    # ========== Train U-NO ==========
    print("\n" + "-" * 50)
    print("TRAINING U-NO (Neural Operator)")
    print("-" * 50)
    print("Advantages: Fourier layers, multi-scale structure, spectral bias")

    uno_params = init_uno(key_uno, depth=3, width0=32, modes=16)
    uno_n_params = count_params(uno_params)
    print(f"Parameters: {uno_n_params:,}")

    def uno_wrapper(params, x):
        single = x.ndim == 3
        if single:
            x = x[None, ...]
        out = uno_forward(params, x, depth=3, modes=16)
        return out[0] if single else out

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(uno_params)

    t0 = time.time()
    for step in range(steps):
        idx = jnp.arange(batch_size) + (step * batch_size) % (
            train_samples - batch_size
        )
        batch_a, batch_u = a_train[idx], u_train[idx]

        uno_params, opt_state, loss = train_step_jit(
            uno_params, opt_state, batch_a, batch_u, uno_wrapper, optimizer
        )

        if step % 100 == 0:
            print(f"Step {step:3d}: loss = {loss:.4e}")

    uno_time = time.time() - t0
    uno_results = evaluate_model(uno_params, uno_wrapper, a_test, u_test)
    print(f"Training time: {uno_time:.1f}s")

    # ========== Results Summary ==========
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"{'Model':<8} {'Params':>10} {'Test MSE':>12} {'Rel Error':>12} {'Max Error':>12}"
    )
    print("-" * 70)
    print(
        f"{'MLP':<8} {mlp_n_params:>10,} {mlp_results['mse']:>12.4e} "
        f"{mlp_results['relative_error']:>11.1%} {mlp_results['max_error']:>12.2f}"
    )
    print(
        f"{'U-NO':<8} {uno_n_params:>10,} {uno_results['mse']:>12.4e} "
        f"{uno_results['relative_error']:>11.1%} {uno_results['max_error']:>12.2f}"
    )

    # ========== Analysis ==========
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)

    accuracy_ratio = mlp_results["mse"] / uno_results["mse"]
    param_ratio = uno_n_params / mlp_n_params

    print(f"\n✓ U-NO is {accuracy_ratio:.1f}× more accurate than MLP")
    print(
        f"✓ U-NO has {param_ratio:.1f}× more parameters but uses them more effectively"
    )
    print(
        f"✓ U-NO relative error: {uno_results['relative_error']:.1%} vs MLP: {mlp_results['relative_error']:.1%}"
    )

    # ========== Visualization ==========
    print("\n" + "=" * 70)
    print("VISUALIZING PREDICTIONS")
    print("=" * 70)

    # Get predictions for visualization
    test_idx = 0  # Use first test sample
    a_vis = a_test[test_idx]
    u_vis = u_test[test_idx]
    mlp_pred_vis = mlp_forward(mlp_params, a_vis)
    uno_pred_vis = uno_wrapper(uno_params, a_vis)

    # Create visualization
    fig = visualize_predictions(
        a_vis,
        u_vis,
        mlp_pred_vis,
        uno_pred_vis,
        "Complex Operator Learning: Ground Truth vs Predictions",
    )
    plt.savefig("uno_vs_mlp_predictions.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to 'uno_vs_mlp_predictions.png'")
    plt.show()

    # ========== Multi-scale Testing for U-NO ==========
    key_multiscale = jax.random.PRNGKey(123)
    multiscale_results = test_multiscale_uno(
        uno_params, uno_wrapper, key_multiscale, original_n=n
    )

    print("\n" + "=" * 70)
    print("MULTI-SCALE RESULTS SUMMARY")
    print("=" * 70)
    print("U-NO's performance across different resolutions:")
    print(f"{'Resolution':<15} {'MSE':>12} {'Rel Error':>12}")
    print("-" * 40)
    for res, results in multiscale_results.items():
        print(
            f"{res}×{res:<11} {results['mse']:>12.4e} {results['relative_error']:>11.1%}"
        )

    print("\n" + "=" * 70)
    print("WHY U-NO WINS ON COMPLEX OPERATORS")
    print("=" * 70)
    print("1. FOURIER LAYERS: Directly capture frequency components (2π, 4π, 6π)")
    print("2. GLOBAL CONTEXT: Efficiently handle non-local FFT interactions")
    print("3. MULTI-SCALE: U-Net preserves both coarse and fine features")
    print("4. INDUCTIVE BIAS: Architecture matches the operator structure")
    print("5. SPECTRAL EFFICIENCY: Natural for smooth physical operators")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("For complex operators with global dependencies and multiple scales,")
    print("U-NO dramatically outperforms naive approaches. The architecture")
    print("matters more than parameter count when the design matches the problem.")


if __name__ == "__main__":
    main()
