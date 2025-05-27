"""Complex comparison of U-NO with naive approaches
==================================================
This script compares U-NO with naive approaches on a more challenging task
that better demonstrates the advantages of Fourier layers and U-Net structure.

The task involves learning a more complex operator with:
- Multiple frequency components
- Non-local dependencies
- Multi-scale features
"""

import time
from functools import partial
from typing import Dict, Any
import sys

import jax
import jax.numpy as jnp
import optax

Array = jax.Array

# Import the U-NO implementation
import importlib.util

spec = importlib.util.spec_from_file_location("small_uno_demo", "small-uno-demo.py")
small_uno_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(small_uno_demo)

# Import what we need
grid = small_uno_demo.grid
mse = small_uno_demo.mse
init_uno = small_uno_demo.init_uno
uno_forward = small_uno_demo.uno_forward
init_dense = small_uno_demo.init_dense
dense = small_uno_demo.dense

# ------------------------------------------------------------
#  More challenging dataset
# ------------------------------------------------------------


def make_complex_dataset(key: Array, n_samples: int, n: int = 64):
    """Generate a more complex operator learning task.

    The operator maps a -> u where:
    - a is a random field
    - u involves multiple scales, frequencies, and non-local interactions
    """
    xx, yy = grid(n)
    keys = jax.random.split(key, n_samples + 1)

    # Input: smoother random fields with structure
    def generate_structured_field(k):
        # Low frequency component
        k1, k2, k3 = jax.random.split(k, 3)
        low_freq = jax.random.normal(k1, (8, 8, 1))
        low_freq = jax.image.resize(low_freq, (n, n, 1), method="cubic")

        # High frequency component
        high_freq = 0.3 * jax.random.normal(k2, (n, n, 1))

        # Combine
        return low_freq + high_freq

    a = jax.vmap(generate_structured_field)(keys[:-1])  # (N, n, n, 1)

    # Output: complex operator with multiple scales and non-local effects
    def complex_operator(field, xx, yy):
        # Global features
        mean_val = jnp.mean(field)
        std_val = jnp.std(field)

        # Multiple frequency components
        u1 = jnp.sin(2 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy)
        u2 = jnp.sin(4 * jnp.pi * xx) * jnp.sin(4 * jnp.pi * yy)
        u3 = jnp.cos(6 * jnp.pi * (xx + yy))

        # Non-local interaction: field values affect the whole domain
        field_fft = jnp.fft.fft2(field[..., 0])
        field_power = jnp.abs(field_fft[:5, :5]).mean()  # Low frequency power

        # Combine everything
        u = (
            0.5 * u1
            + 0.3 * u2 * mean_val
            + 0.2 * u3 * std_val
            + 0.1 * field_power * jnp.ones_like(u1)
        )

        # Add some of the input field for locality
        u = u + 0.1 * field[..., 0]

        return u[..., None]

    u = jax.vmap(lambda f: complex_operator(f, xx, yy))(a)

    return a, u


# ------------------------------------------------------------
#  Simplified models (reuse from before but adapted)
# ------------------------------------------------------------


def init_mlp(key: Array, input_dim: int, hidden_dims: list, output_dim: int):
    """Initialize MLP."""
    keys = jax.random.split(key, len(hidden_dims) + 1)
    params = {"layers": []}

    in_dim = input_dim
    for i, h_dim in enumerate(hidden_dims):
        w = jax.random.normal(keys[i], (in_dim, h_dim)) / jnp.sqrt(in_dim)
        b = jnp.zeros((h_dim,))
        params["layers"].append((w, b))
        in_dim = h_dim

    w = jax.random.normal(keys[-1], (in_dim, output_dim)) / jnp.sqrt(in_dim)
    b = jnp.zeros((output_dim,))
    params["layers"].append((w, b))

    return params


def mlp_forward(params: Dict[str, Any], x: Array):
    """MLP forward pass."""
    single_input = x.ndim == 3
    if single_input:
        H, W, C = x.shape
        x_flat = x.reshape(-1)
    else:
        B, H, W, C = x.shape
        x_flat = x.reshape(B, -1)

    for i, (w, b) in enumerate(params["layers"][:-1]):
        x_flat = jnp.dot(x_flat, w) + b
        x_flat = jax.nn.gelu(x_flat)

    w, b = params["layers"][-1]
    x_flat = jnp.dot(x_flat, w) + b

    if single_input:
        return x_flat.reshape(H, W, 1)
    else:
        return x_flat.reshape(B, H, W, 1)


# ------------------------------------------------------------
#  Training utilities
# ------------------------------------------------------------


def train_step(params, opt_state, batch_a, batch_u, forward_fn, optimizer):
    """Training step."""

    def loss_fn(p):
        pred = forward_fn(p, batch_a)
        return mse(pred, batch_u)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


train_step_jit = jax.jit(train_step, static_argnames=["forward_fn", "optimizer"])


def count_params(params):
    """Count parameters."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def evaluate_model(params, forward_fn, a_test, u_test):
    """Evaluate model performance."""
    predictions = jax.vmap(forward_fn, in_axes=(None, 0))(params, a_test)
    test_loss = float(mse(predictions, u_test))

    errors = jnp.abs(predictions - u_test)
    max_error = float(jnp.max(errors))
    mean_error = float(jnp.mean(errors))

    # Relative error
    rel_error = float(jnp.mean(errors / (jnp.abs(u_test) + 1e-8)))

    return {
        "test_loss": test_loss,
        "max_error": max_error,
        "mean_error": mean_error,
        "rel_error": rel_error,
        "predictions": predictions,
    }


# ------------------------------------------------------------
#  Main comparison
# ------------------------------------------------------------


def main():
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)

    # Settings
    n = 64  # Higher resolution
    train_size = 256
    test_size = 64
    batch_size = 8

    print("=" * 70)
    print("COMPLEX OPERATOR LEARNING: U-NO vs NAIVE APPROACHES")
    print("=" * 70)
    print(f"Task: Learning a complex operator with:")
    print(f"  - Multiple frequency components")
    print(f"  - Non-local dependencies (FFT-based)")
    print(f"  - Multi-scale features")
    print(f"Spatial resolution: {n}x{n}")
    print("=" * 70)

    # Generate data
    print("\nGenerating complex dataset...")
    a_train, u_train = make_complex_dataset(keys[0], train_size, n)
    a_test, u_test = make_complex_dataset(keys[1], test_size, n)
    print(
        f"Training data range: a=[{float(a_train.min()):.2f}, {float(a_train.max()):.2f}], u=[{float(u_train.min()):.2f}, {float(u_train.max()):.2f}]"
    )

    results = {}

    # 1. MLP - struggles with spatial structure
    print("\n1. TRAINING MLP")
    print("   Expected issues: Cannot capture spatial patterns efficiently")

    mlp_params = init_mlp(keys[2], n * n * 1, [512, 1024, 512], n * n * 1)
    print(f"   Parameters: {count_params(mlp_params):,}")

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(mlp_params)

    # Quick training
    for step in range(200):
        idx = jnp.arange(batch_size) + (step * batch_size) % (train_size - batch_size)
        batch_a, batch_u = a_train[idx], u_train[idx]
        mlp_params, opt_state, loss = train_step_jit(
            mlp_params, opt_state, batch_a, batch_u, mlp_forward, optimizer
        )
        if step % 50 == 0:
            print(f"   Step {step}: loss = {loss:.4e}")

    results["MLP"] = evaluate_model(mlp_params, mlp_forward, a_test, u_test)
    results["MLP"]["n_params"] = count_params(mlp_params)

    # 2. U-NO - designed for this type of task
    print("\n2. TRAINING U-NO")
    print(
        "   Expected advantages: Fourier layers capture frequencies, U-Net preserves scales"
    )

    uno_params = init_uno(keys[3], depth=3, width0=32, modes=16)
    print(f"   Parameters: {count_params(uno_params):,}")

    def uno_wrapper(params, x):
        single = x.ndim == 3
        if single:
            x = x[None, ...]
        out = uno_forward(params, x, depth=3, modes=16)
        if single:
            out = out[0]
        return out

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(uno_params)

    # Training
    for step in range(200):
        idx = jnp.arange(batch_size) + (step * batch_size) % (train_size - batch_size)
        batch_a, batch_u = a_train[idx], u_train[idx]
        uno_params, opt_state, loss = train_step_jit(
            uno_params, opt_state, batch_a, batch_u, uno_wrapper, optimizer
        )
        if step % 50 == 0:
            print(f"   Step {step}: loss = {loss:.4e}")

    results["U-NO"] = evaluate_model(uno_params, uno_wrapper, a_test, u_test)
    results["U-NO"]["n_params"] = count_params(uno_params)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS ON COMPLEX OPERATOR TASK")
    print("=" * 70)
    print(
        f"{'Model':<10} {'Parameters':<12} {'Test MSE':<12} {'Rel Error':<12} {'Max Error':<12}"
    )
    print("-" * 70)

    for name in ["MLP", "U-NO"]:
        res = results[name]
        print(
            f"{name:<10} {res['n_params']:<12,} {res['test_loss']:<12.4e} {res['rel_error']:<12.2%} {res['max_error']:<12.4e}"
        )

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    mlp_loss = results["MLP"]["test_loss"]
    uno_loss = results["U-NO"]["test_loss"]

    print(f"\n1. ACCURACY: U-NO is {mlp_loss / uno_loss:.1f}x more accurate than MLP")

    mlp_params = results["MLP"]["n_params"]
    uno_params = results["U-NO"]["n_params"]

    print(
        f"\n2. EFFICIENCY: Despite having {uno_params / mlp_params:.1f}x more parameters,"
    )
    print(f"   U-NO uses them more effectively for spatial operators")

    print(f"\n3. ERROR DISTRIBUTION:")
    print(f"   - MLP relative error: {results['MLP']['rel_error']:.1%}")
    print(f"   - U-NO relative error: {results['U-NO']['rel_error']:.1%}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("• U-NO excels at complex operators with multiple scales")
    print("• Fourier layers naturally capture frequency content")
    print("• U-Net structure preserves both local and global features")
    print("• MLPs struggle with spatial structure and need many parameters")
    print("• The gap widens with more complex operators and higher resolutions")


if __name__ == "__main__":
    main()
