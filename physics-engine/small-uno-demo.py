"""Scaled-up U-NO demo in pure JAX
================================================
A *less* tiny but still self-contained implementation of the
U-shaped Neural Operator (Rahman et al., 2023) with the
recommended architectural upgrades:

* multi-level encoder/decoder (configurable depth)
* widening channel count as we contract the spatial domain
* average-pool down-sampling (antialiased)
* bilinear up-sampling (jax.image.resize)
* skip connections at every resolution
* adaptive Fourier modes per level

It still trains on the same synthetic *sin + mean(a)* task so that
it can be run quickly on CPU/GPU without external datasets.

Prerequisites
-------------
    pip install -U jax jaxlib optax

How to run
----------
    python scaled_uno_demo.py  # will train for a few seconds
"""

import time
from functools import partial
from typing import List, Sequence, Dict, Any

import jax
import jax.numpy as jnp
import jax.image
import optax

Array = jax.Array  # type alias for clarity

# ------------------------------------------------------------
#  Utility: build a tiny synthetic data set (same as before)
# ------------------------------------------------------------


def grid(n: int):
    """Return coordinate arrays of shape (n,n)."""
    x = jnp.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    return xx, yy


def make_dataset(key: Array, n_samples: int, n: int = 32):
    """Generate (a, u) pairs."""
    xx, yy = grid(n)
    base = jnp.sin(jnp.pi * xx) * jnp.cos(jnp.pi * yy)  # (n,n)
    base = base[..., None]  # -> (n,n,1)

    keys = jax.random.split(key, n_samples)
    a = jax.vmap(lambda k: jax.random.normal(k, (n, n, 1)))(keys)  # (N,n,n,1)

    def build_u(field):
        mean = jnp.mean(field)  # scalar (non-local)
        return base + 0.1 * mean

    u = jax.vmap(build_u)(a)  # (N,n,n,1)
    return a, u


# ------------------------------------------------------------
#  Common building blocks
# ------------------------------------------------------------


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


# ------------------------------------------------------------
#  Scaled-up U-NO
# ------------------------------------------------------------


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
#  Training helpers
# ------------------------------------------------------------


def mse(a: Array, b: Array):
    return jnp.mean((a - b) ** 2)


@partial(jax.jit, static_argnames=("forward", "depth", "modes"))
def loss_fn(forward, params, batch_a, batch_u, depth, modes):
    pred = forward(params, batch_a, depth=depth, modes=modes)
    return mse(pred, batch_u)


@partial(jax.jit, static_argnames=("forward", "optimizer", "depth", "modes"))
def update_step(forward, params, opt_state, batch_a, batch_u, optimizer, depth, modes):
    def _loss_for_grad(p, ba, bu):
        pred = forward(p, ba, depth=depth, modes=modes)
        return mse(pred, bu)

    loss, grads = jax.value_and_grad(_loss_for_grad)(params, batch_a, batch_u)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# ------------------------------------------------------------
#  Simple trainer (same synthetic task)
# ------------------------------------------------------------


def train(
    key: Array,
    epochs: int = 2000,
    n: int = 32,
    train_size: int = 512,
    batch: int = 16,
    depth: int = 3,
    width0: int = 32,
    modes: int | Sequence[int] = 12,
):
    # data --------------------------------------------------
    key_train, key_test = jax.random.split(key)
    a_train, u_train = make_dataset(key_train, train_size, n)
    a_test, u_test = make_dataset(key_test, 64, n)

    # model -------------------------------------------------
    params = init_uno(key, depth=depth, width0=width0, modes=modes)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)

    # batching helper --------------------------------------
    def get_batch(step):
        idx = jnp.arange(batch) + (step * batch) % (train_size - batch)
        return a_train[idx], u_train[idx]

    # loop --------------------------------------------------
    t0 = time.time()
    for step in range(epochs):
        batch_a, batch_u = get_batch(step)
        params, opt_state, loss = update_step(
            uno_forward,
            params,
            opt_state,
            batch_a,
            batch_u,
            optimizer,
            depth,
            modes,
        )

        if step % 50 == 0 or step == epochs - 1:
            test_loss = loss_fn(
                uno_forward, params, a_test, u_test, depth=depth, modes=modes
            )
            print(f"Step {step:4d} | train MSE {loss:8.4e} | test MSE {test_loss:8.4e}")

    print(f"Done in {time.time() - t0:.1f}s")
    return params


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    _ = train(key, epochs=500, depth=3, width0=32, modes=(12, 12, 8, 8))
