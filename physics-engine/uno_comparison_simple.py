"""Simple comparison of U-NO with naive approaches (no plotting)
===========================================================
This script compares the U-shaped Neural Operator (U-NO) with simpler baseline models
without requiring matplotlib for visualization.
"""

import time
from functools import partial
from typing import Dict, Any
import sys

import jax
import jax.numpy as jnp
import optax

Array = jax.Array

# Import the U-NO implementation and utilities from the demo
# Python doesn't allow hyphens in module names, so we need to import it differently
import importlib.util

spec = importlib.util.spec_from_file_location("small_uno_demo", "small-uno-demo.py")
small_uno_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(small_uno_demo)

# Now import what we need
grid = small_uno_demo.grid
make_dataset = small_uno_demo.make_dataset
mse = small_uno_demo.mse
init_uno = small_uno_demo.init_uno
uno_forward = small_uno_demo.uno_forward
init_dense = small_uno_demo.init_dense
dense = small_uno_demo.dense

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
    # Handle both single and batched inputs
    if x.ndim == 3:
        # Single input (H, W, C)
        H, W, C = x.shape
        x_flat = x.reshape(-1)  # (H*W*C,)

        # Forward through layers
        for i, (w, b) in enumerate(params["layers"][:-1]):
            x_flat = jnp.dot(x_flat, w) + b
            x_flat = jax.nn.gelu(x_flat)

        # Output layer (no activation)
        w, b = params["layers"][-1]
        x_flat = jnp.dot(x_flat, w) + b

        # Reshape back to spatial dimensions
        return x_flat.reshape(H, W, 1)
    else:
        # Batched input (B, H, W, C)
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
    # Add batch dimension if needed
    single_input = False
    if x.ndim == 3:
        single_input = True
        x = x[None, ...]  # Add batch dimension

    # Lift
    v = jax.nn.gelu(dense(params["lift"], x))

    # Conv layers
    for conv_params in params["conv_layers"]:
        v = jax.nn.gelu(conv2d(conv_params, v))

    # Project
    v = dense(params["proj"], v)

    # Remove batch dimension if we added it
    if single_input:
        v = v[0]

    return v


# ------------------------------------------------------------
#  Training utilities
# ------------------------------------------------------------


def train_step(params, opt_state, batch_a, batch_u, forward_fn, optimizer):
    """Generic training step."""

    def loss_fn(p):
        pred = forward_fn(p, batch_a)
        return mse(pred, batch_u)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# JIT compile with static arguments
train_step_jit = jax.jit(train_step, static_argnames=["forward_fn", "optimizer"])


def count_params(params):
    """Count total number of parameters."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def train_model(
    key: Array,
    init_fn,
    forward_fn,
    model_name: str,
    epochs: int = 300,
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
    train_losses = []
    test_losses = []
    t0 = time.time()

    for step in range(epochs):
        # Get batch
        idx = jnp.arange(batch_size) + (step * batch_size) % (train_size - batch_size)
        batch_a, batch_u = a_train[idx], u_train[idx]

        # Update
        params, opt_state, loss = train_step_jit(
            params, opt_state, batch_a, batch_u, forward_fn, optimizer
        )

        # Log
        if step % 50 == 0 or step == epochs - 1:
            test_pred = jax.vmap(forward_fn, in_axes=(None, 0))(params, a_test)
            test_loss = mse(test_pred, u_test)
            train_losses.append(float(loss))
            test_losses.append(float(test_loss))
            print(
                f"  Step {step:4d} | train MSE {loss:8.4e} | test MSE {test_loss:8.4e}"
            )

    total_time = time.time() - t0
    print(f"  Training time: {total_time:.1f}s")

    # Final evaluation
    final_test_pred = jax.vmap(forward_fn, in_axes=(None, 0))(params, a_test)
    final_test_loss = float(mse(final_test_pred, u_test))

    # Compute prediction quality metrics
    errors = jnp.abs(final_test_pred - u_test)
    max_error = float(jnp.max(errors))
    mean_error = float(jnp.mean(errors))

    return {
        "name": model_name,
        "n_params": n_params,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_test_loss": final_test_loss,
        "total_time": total_time,
        "max_error": max_error,
        "mean_error": mean_error,
    }


# ------------------------------------------------------------
#  Main comparison
# ------------------------------------------------------------


def main():
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Common settings
    n = 32  # spatial resolution
    epochs = 300  # Reduced for faster demo

    print("=" * 70)
    print("COMPARING U-NO WITH NAIVE APPROACHES")
    print("=" * 70)
    print(f"Task: Learning operator a(x,y) -> u(x,y)")
    print(f"      where u = sin(πx)cos(πy) + 0.1 * mean(a)")
    print(f"Spatial resolution: {n}x{n}")
    print(f"Training epochs: {epochs}")
    print("=" * 70)

    results = {}

    # 1. Train naive MLP
    print("\n1. NAIVE MLP (flattens spatial structure)")
    print("   - Destroys spatial locality")
    print("   - Cannot leverage translation invariance")
    print("   - Requires many parameters")
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
    print("\n2. SIMPLE CNN (local convolutions only)")
    print("   - Limited receptive field")
    print("   - Cannot capture global dependencies efficiently")
    print("   - Needs many layers for large receptive field")
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
    print("\n3. U-NO (Fourier layers + U-Net structure)")
    print("   - Fourier layers capture global dependencies")
    print("   - U-Net preserves multi-scale information")
    print("   - Efficient parameter usage")

    # Create a wrapper for U-NO forward that matches our interface
    def uno_forward_wrapper(params, x):
        # Handle single input
        single_input = False
        if x.ndim == 3:
            single_input = True
            x = x[None, ...]  # Add batch dimension

        result = uno_forward(params, x, depth=3, modes=12)

        # Remove batch dimension if we added it
        if single_input:
            result = result[0]

        return result

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
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"{'Model':<15} {'Parameters':<12} {'Test MSE':<12} {'Max Error':<12} {'Time (s)':<10}"
    )
    print("-" * 70)
    for name, res in results.items():
        print(
            f"{name:<15} {res['n_params']:<12,} {res['final_test_loss']:<12.2e} {res['max_error']:<12.2e} {res['total_time']:<10.1f}"
        )

    # Detailed analysis
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Compare accuracy
    uno_loss = results["U-NO"]["final_test_loss"]
    mlp_loss = results["MLP"]["final_test_loss"]
    cnn_loss = results["CNN"]["final_test_loss"]

    print("\nAccuracy Comparison:")
    print(f"  • U-NO is {mlp_loss / uno_loss:.1f}x more accurate than MLP")
    print(f"  • U-NO is {cnn_loss / uno_loss:.1f}x more accurate than CNN")

    # Compare efficiency
    uno_params = results["U-NO"]["n_params"]
    mlp_params = results["MLP"]["n_params"]
    cnn_params = results["CNN"]["n_params"]

    print("\nParameter Efficiency:")
    print(f"  • U-NO uses {mlp_params / uno_params:.1f}x fewer parameters than MLP")
    print(f"  • U-NO uses {cnn_params / uno_params:.1f}x fewer parameters than CNN")

    # Compare errors
    print("\nPrediction Quality:")
    for name, res in results.items():
        print(
            f"  • {name}: max error = {res['max_error']:.3e}, mean error = {res['mean_error']:.3e}"
        )

    print("\n" + "=" * 70)
    print("WHY U-NO IS SUPERIOR")
    print("=" * 70)
    print("1. GLOBAL DEPENDENCIES: Fourier layers efficiently capture long-range")
    print("   interactions that CNNs struggle with")
    print("\n2. SPECTRAL BIAS: Natural for learning smooth operators common in")
    print("   physics and engineering")
    print("\n3. MULTI-SCALE: U-Net structure preserves information at multiple")
    print("   resolutions, crucial for PDEs")
    print("\n4. PARAMETER EFFICIENCY: Achieves better accuracy with fewer")
    print("   parameters than naive approaches")
    print("\n5. INDUCTIVE BIAS: Architecture is specifically designed for")
    print("   operator learning tasks")

    # Show convergence comparison
    print("\n" + "=" * 70)
    print("CONVERGENCE COMPARISON (Test Loss)")
    print("=" * 70)
    print("Step    MLP         CNN         U-NO")
    print("-" * 40)
    n_logs = len(results["U-NO"]["test_losses"])
    for i in range(n_logs):
        step = i * 50
        mlp_l = (
            results["MLP"]["test_losses"][i]
            if i < len(results["MLP"]["test_losses"])
            else float("nan")
        )
        cnn_l = (
            results["CNN"]["test_losses"][i]
            if i < len(results["CNN"]["test_losses"])
            else float("nan")
        )
        uno_l = results["U-NO"]["test_losses"][i]
        print(f"{step:<8}{mlp_l:<12.2e}{cnn_l:<12.2e}{uno_l:<12.2e}")


if __name__ == "__main__":
    main()
