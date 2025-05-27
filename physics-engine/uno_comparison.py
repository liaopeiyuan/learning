"""Comparison of U-NO with naive approaches
==========================================
This script compares the U-shaped Neural Operator (U-NO) with simpler baseline models:
1. A naive MLP (Multi-Layer Perceptron) that flattens the input
2. A simple CNN (Convolutional Neural Network) without Fourier layers
3. The U-NO architecture

The comparison demonstrates U-NO's advantages in:
- Parameter efficiency
- Ability to capture global dependencies
- Performance on operator learning tasks
"""

import time
from functools import partial
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

Array = jax.Array

# Import the U-NO implementation and utilities from the demo
from small_uno_demo import (
    grid,
    make_dataset,
    mse,
    init_uno,
    uno_forward,
    init_dense,
    dense,
    init_fft_layer,
    fft_layer,
    avg_pool2d,
    bilinear_resize,
)

# ------------------------------------------------------------
#  Naive Approach 1: Simple MLP
# ------------------------------------------------------------


def init_mlp(key: Array, input_dim: int, hidden_dims: list, output_dim: int):
    """Initialize a simple MLP that flattens spatial dimensions."""
    keys = jax.random.split(key, len(hidden_dims) + 1)
    params = {"layers": []}

    # Input layer
    in_dim = input_dim
    for i, h_dim in enumerate(hidden_dims):
        w = jax.random.normal(keys[i], (in_dim, h_dim)) / jnp.sqrt(in_dim)
        b = jnp.zeros((h_dim,))
        params["layers"].append((w, b))
        in_dim = h_dim

    # Output layer
    w = jax.random.normal(keys[-1], (in_dim, output_dim)) / jnp.sqrt(in_dim)
    b = jnp.zeros((output_dim,))
    params["layers"].append((w, b))

    return params


def mlp_forward(params: Dict[str, Any], x: Array):
    """Forward pass for MLP - flattens input and reshapes output."""
    B, H, W, C = x.shape
    # Flatten spatial dimensions
    x_flat = x.reshape(B, -1)  # (B, H*W*C)

    # Forward through layers
    for i, (w, b) in enumerate(params["layers"][:-1]):
        x_flat = jnp.dot(x_flat, w) + b
        x_flat = jax.nn.gelu(x_flat)

    # Output layer (no activation)
    w, b = params["layers"][-1]
    x_flat = jnp.dot(x_flat, w) + b

    # Reshape back to spatial dimensions
    return x_flat.reshape(B, H, W, 1)


# ------------------------------------------------------------
#  Naive Approach 2: Simple CNN (no Fourier layers)
# ------------------------------------------------------------


def init_conv2d(key: Array, in_ch: int, out_ch: int, kernel_size: int = 3):
    """Initialize a 2D convolution layer."""
    fan_in = in_ch * kernel_size * kernel_size
    w = jax.random.normal(key, (kernel_size, kernel_size, in_ch, out_ch)) / jnp.sqrt(
        fan_in
    )
    b = jnp.zeros((out_ch,))
    return (w, b)


def conv2d(params, x: Array, padding: str = "SAME"):
    """Apply 2D convolution."""
    w, b = params
    # Use lax.conv_general_dilated for the convolution
    x_conv = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return x_conv + b


def init_simple_cnn(key: Array, depth: int = 3, width: int = 32):
    """Initialize a simple CNN with fixed width (no U-shape)."""
    keys = jax.random.split(key, 2 * depth + 3)
    params = {}

    # Lift to higher dimension
    params["lift"] = init_dense(keys[0], 1, width)

    # Convolutional layers (no downsampling)
    params["conv_layers"] = []
    for i in range(depth):
        params["conv_layers"].append(init_conv2d(keys[i + 1], width, width))

    # Project back to output
    params["proj"] = init_dense(keys[-1], width, 1)

    return params


def simple_cnn_forward(params: Dict[str, Any], x: Array):
    """Forward pass for simple CNN."""
    # Lift
    v = jax.nn.gelu(dense(params["lift"], x))

    # Conv layers
    for conv_params in params["conv_layers"]:
        v = jax.nn.gelu(conv2d(conv_params, v))

    # Project
    return dense(params["proj"], v)


# ------------------------------------------------------------
#  Training and evaluation utilities
# ------------------------------------------------------------


@jax.jit
def train_step(params, opt_state, batch_a, batch_u, forward_fn, optimizer):
    """Generic training step."""

    def loss_fn(p):
        pred = forward_fn(p, batch_a)
        return mse(pred, batch_u)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def count_params(params):
    """Count total number of parameters."""
    if isinstance(params, (list, tuple)):
        return sum(p.size for p in jax.tree_util.tree_leaves(params))
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def train_model(
    key: Array,
    init_fn,
    forward_fn,
    model_name: str,
    epochs: int = 500,
    n: int = 32,
    train_size: int = 512,
    batch_size: int = 16,
    lr: float = 3e-4,
    **init_kwargs,
):
    """Train a model and return results."""
    # Data
    key_train, key_test, key_model = jax.random.split(key, 3)
    a_train, u_train = make_dataset(key_train, train_size, n)
    a_test, u_test = make_dataset(key_test, 64, n)

    # Model
    params = init_fn(key_model, **init_kwargs)
    n_params = count_params(params)
    print(f"\n{model_name}: {n_params:,} parameters")

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Training
    losses = []
    test_losses = []
    times = []
    t0 = time.time()

    for step in range(epochs):
        # Get batch
        idx = jnp.arange(batch_size) + (step * batch_size) % (train_size - batch_size)
        batch_a, batch_u = a_train[idx], u_train[idx]

        # Update
        params, opt_state, loss = train_step(
            params, opt_state, batch_a, batch_u, forward_fn, optimizer
        )

        # Log
        if step % 50 == 0 or step == epochs - 1:
            test_pred = jax.vmap(forward_fn, in_axes=(None, 0))(params, a_test)
            test_loss = mse(test_pred, u_test)
            losses.append(float(loss))
            test_losses.append(float(test_loss))
            times.append(time.time() - t0)
            print(
                f"  Step {step:4d} | train MSE {loss:8.4e} | test MSE {test_loss:8.4e}"
            )

    total_time = time.time() - t0
    print(f"  Training time: {total_time:.1f}s")

    # Final evaluation
    final_test_pred = jax.vmap(forward_fn, in_axes=(None, 0))(params, a_test)
    final_test_loss = float(mse(final_test_pred, u_test))

    return {
        "name": model_name,
        "params": params,
        "n_params": n_params,
        "losses": losses,
        "test_losses": test_losses,
        "times": times,
        "final_test_loss": final_test_loss,
        "total_time": total_time,
        "predictions": final_test_pred,
    }


# ------------------------------------------------------------
#  Visualization utilities
# ------------------------------------------------------------


def plot_comparison(results_dict: Dict[str, Any], a_test: Array, u_test: Array):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Training curves
    ax = axes[0, 0]
    for name, res in results_dict.items():
        steps = jnp.arange(len(res["test_losses"])) * 50
        ax.semilogy(
            steps, res["test_losses"], label=f"{name} ({res['n_params']:,} params)"
        )
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Test MSE (log scale)")
    ax.set_title("Test Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Parameter efficiency
    ax = axes[0, 1]
    names = list(results_dict.keys())
    params = [results_dict[n]["n_params"] for n in names]
    losses = [results_dict[n]["final_test_loss"] for n in names]
    colors = plt.cm.tab10(range(len(names)))
    bars = ax.bar(names, params, color=colors)
    ax.set_ylabel("Number of Parameters")
    ax.set_title("Model Size Comparison")
    # Add loss values on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"MSE: {loss:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 3. Time efficiency
    ax = axes[0, 2]
    times = [results_dict[n]["total_time"] for n in names]
    bars = ax.bar(names, times, color=colors)
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison")

    # 4-6. Visual comparison of predictions
    test_idx = 0  # Show first test sample
    vmin = min(
        u_test[test_idx].min(),
        min(res["predictions"][test_idx].min() for res in results_dict.values()),
    )
    vmax = max(
        u_test[test_idx].max(),
        max(res["predictions"][test_idx].max() for res in results_dict.values()),
    )

    # Ground truth
    ax = axes[1, 0]
    im = ax.imshow(u_test[test_idx, :, :, 0], cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title("Ground Truth")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Model predictions
    for idx, (name, res) in enumerate(results_dict.items()):
        if idx < 2:  # Only show first two models
            ax = axes[1, idx + 1]
            pred = res["predictions"][test_idx, :, :, 0]
            im = ax.imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
            error = jnp.abs(pred - u_test[test_idx, :, :, 0])
            ax.set_title(f"{name}\nMax Error: {error.max():.3e}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------
#  Main comparison
# ------------------------------------------------------------


def main():
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Common settings
    n = 32  # spatial resolution
    epochs = 500

    print("=" * 60)
    print("Comparing U-NO with Naive Approaches")
    print("=" * 60)
    print(f"Task: Learning operator a(x,y) -> u(x,y)")
    print(f"Spatial resolution: {n}x{n}")
    print(f"Training epochs: {epochs}")

    # Generate test data for visualization
    a_test, u_test = make_dataset(keys[0], 8, n)

    results = {}

    # 1. Train naive MLP
    print("\n1. Training Naive MLP (flattens spatial structure)")
    results["MLP"] = train_model(
        keys[1],
        init_mlp,
        mlp_forward,
        "Naive MLP",
        epochs=epochs,
        n=n,
        input_dim=n * n * 1,
        hidden_dims=[256, 512, 256],
        output_dim=n * n * 1,
    )

    # 2. Train simple CNN
    print("\n2. Training Simple CNN (local convolutions only)")
    results["CNN"] = train_model(
        keys[2],
        init_simple_cnn,
        simple_cnn_forward,
        "Simple CNN",
        epochs=epochs,
        n=n,
        depth=4,
        width=64,
    )

    # 3. Train U-NO
    print("\n3. Training U-NO (Fourier layers + U-Net structure)")

    # Create a wrapper for U-NO forward that matches our interface
    def uno_forward_wrapper(params, x):
        return uno_forward(params, x, depth=3, modes=12)

    results["U-NO"] = train_model(
        keys[3],
        init_uno,
        uno_forward_wrapper,
        "U-NO",
        epochs=epochs,
        n=n,
        depth=3,
        width0=32,
        modes=12,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Parameters':<15} {'Final Test MSE':<15} {'Time (s)':<10}")
    print("-" * 60)
    for name, res in results.items():
        print(
            f"{name:<15} {res['n_params']:<15,} {res['final_test_loss']:<15.2e} {res['total_time']:<10.1f}"
        )

    # Analysis
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    # Compare performance
    uno_loss = results["U-NO"]["final_test_loss"]
    mlp_loss = results["MLP"]["final_test_loss"]
    cnn_loss = results["CNN"]["final_test_loss"]

    print(f"1. U-NO achieves {mlp_loss / uno_loss:.1f}x better accuracy than MLP")
    print(f"2. U-NO achieves {cnn_loss / uno_loss:.1f}x better accuracy than CNN")

    # Compare efficiency
    uno_params = results["U-NO"]["n_params"]
    mlp_params = results["MLP"]["n_params"]
    cnn_params = results["CNN"]["n_params"]

    print(f"3. U-NO uses {mlp_params / uno_params:.1f}x fewer parameters than MLP")
    print(f"4. U-NO uses {cnn_params / uno_params:.1f}x fewer parameters than CNN")

    print("\nWhy U-NO is superior:")
    print("- Fourier layers capture global dependencies efficiently")
    print("- U-Net structure preserves multi-scale information")
    print("- Spectral bias helps learn smooth operators")
    print("- Skip connections enable better gradient flow")

    # Create visualization
    fig = plot_comparison(results, a_test, u_test)
    plt.savefig("uno_comparison.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to 'uno_comparison.png'")
    plt.show()


if __name__ == "__main__":
    main()
