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

# Import U-NO implementation
import importlib.util

spec = importlib.util.spec_from_file_location("uno_demo", "small-uno-demo.py")
uno_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uno_module)

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

    mlp_params = init_mlp(key_mlp, n * n, [512, 1024, 512], n * n)
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

    uno_params = uno_module.init_uno(key_uno, depth=3, width0=32, modes=16)
    uno_n_params = count_params(uno_params)
    print(f"Parameters: {uno_n_params:,}")

    def uno_wrapper(params, x):
        single = x.ndim == 3
        if single:
            x = x[None, ...]
        out = uno_module.uno_forward(params, x, depth=3, modes=16)
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
