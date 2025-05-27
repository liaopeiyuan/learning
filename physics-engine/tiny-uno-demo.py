"""Tiny U-NO demo in pure JAX
=================================
A *very* small U-shaped Neural Operator that learns the
following mapping on the unit square

    u(x,y) = sin(pi x)·cos(pi y) + 0.1 · average(a)

where **a(x,y)** is a random scalar field.
The dependence on *average(a)* makes the target non-local,
so a point-wise CNN cannot nail it, but a Neural Operator can.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import optax


# ------------------------------------------------------------
#  Utility: build a data set
# ------------------------------------------------------------
def grid(n):
    """Return coordinate arrays of shape (n,n)."""
    x = jnp.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    return xx, yy


def make_dataset(key, n_samples, n=32):
    """Generate (a, u) pairs."""
    xx, yy = grid(n)
    base = jnp.sin(jnp.pi * xx) * jnp.cos(jnp.pi * yy)  # (n,n)
    base = base[..., None]  # add channel dim -> (n,n,1)

    keys = jax.random.split(key, n_samples)
    a = jax.vmap(lambda k: jax.random.normal(k, (n, n, 1)))(keys)  # (N,n,n,1)

    def build_u(field):
        mean = jnp.mean(field)  # scalar (non-local!)
        return base + 0.1 * mean

    u = jax.vmap(build_u)(a)  # (N,n,n,1)
    return a, u


# ------------------------------------------------------------
#  Core building blocks
# ------------------------------------------------------------
def init_dense(key, in_channels, out_channels):
    k1, k2 = jax.random.split(key)
    w = jax.random.normal(k1, (in_channels, out_channels)) / jnp.sqrt(in_channels)
    b = jnp.zeros((out_channels,))
    return (w, b)


def dense(params, x):
    w, b = params
    return jnp.einsum("...c,co->...o", x, w) + b


def init_fft_layer(key, in_ch, out_ch, modes):
    k1, k2, k3 = jax.random.split(key, 3)
    # complex weight for the first `modes` low-frequency coefficients
    weight = (
        jax.random.normal(k1, (modes, modes, out_ch, in_ch))
        + 1j * jax.random.normal(k2, (modes, modes, out_ch, in_ch))
    ) / jnp.sqrt(in_ch * out_ch)
    # point-wise residual matrix
    W_local = jax.random.normal(k3, (in_ch, out_ch)) / jnp.sqrt(in_ch)
    return dict(weight=weight, W_local=W_local)


def fft_layer(params, x, modes):
    """Fourier Neural Operator layer (2-D real)."""
    # x: (B,H,W,C_in)
    B, H, W, C_in = x.shape
    C_out = params["W_local"].shape[1]

    v_hat = jnp.fft.rfftn(x, axes=(1, 2))  # (B,H,W//2+1,C_in)
    v_hat_out = jnp.zeros(
        v_hat.shape[:-1] + (C_out,),  # (B,H,W//2+1,C_out)
        dtype=jnp.complex64,
    )

    m = modes
    # low-frequency window
    sl = (slice(None), slice(0, m), slice(0, m))
    v_slice = v_hat[sl + (slice(None),)]  # (B,m,m,C_in)

    # complex multiplication with learned weights
    out_slice = jnp.einsum(
        "bxyc,xyoc->bxyo", v_slice, params["weight"]
    )  # (B,m,m,C_out)

    v_hat_out = v_hat_out.at[sl + (slice(None),)].set(out_slice)
    v_low = jnp.fft.irfftn(v_hat_out, s=(H, W), axes=(1, 2)).real  # back to space

    # local (point-wise) linear map
    v_local = jnp.einsum("...c,co->...o", x, params["W_local"])

    return jax.nn.gelu(v_low + v_local)  # residual + GELU


def downsample(x):
    return x[:, ::2, ::2, :]


def upsample(x, target_hw):
    x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
    return x[:, : target_hw[0], : target_hw[1], :]


# ------------------------------------------------------------
#  Tiny U-NO model
# ------------------------------------------------------------
def init_uno(key, in_ch=1, width=16, modes=8):
    """Depth-2 U-NO: lift -> enc -> bottleneck -> dec -> project"""
    keys = jax.random.split(key, 6)
    params = dict()
    params["lift"] = init_dense(keys[0], in_ch, width)
    params["proj"] = init_dense(keys[1], width, 1)

    # Fourier layers
    params["fft_layers"] = [
        init_fft_layer(keys[2], width, width, modes),  # encoder
        init_fft_layer(keys[3], width, width, modes),  # bottleneck
        init_fft_layer(keys[4], 2 * width, width, modes),  # decoder
    ]
    return params


def uno_forward(params, x, modes=8):
    skips = []
    v = jax.nn.gelu(dense(params["lift"], x))  # (B,H,W,C)

    # encoder -------------------------------------------------
    v = fft_layer(params["fft_layers"][0], v, modes)
    skips.append(v)
    v = downsample(v)

    # bottleneck ---------------------------------------------
    v = fft_layer(params["fft_layers"][1], v, modes)

    # decoder -------------------------------------------------
    v = upsample(v, skips[-1].shape[1:3])
    v = jnp.concatenate([v, skips.pop()], axis=-1)  # channel dim = 2*width
    v = fft_layer(params["fft_layers"][2], v, modes)

    # project -------------------------------------------------
    u = dense(params["proj"], v)  # (B,H,W,1)
    return u


# JIT-compile — makes training much faster
uno_forward = jax.jit(uno_forward)


# ------------------------------------------------------------
#  Training
# ------------------------------------------------------------
def mse(a, b):
    return jnp.mean((a - b) ** 2)


@partial(jax.jit, static_argnums=0)
def loss_fn(forward, params, batch_a, batch_u):
    pred = forward(params, batch_a)
    return mse(pred, batch_u)


@partial(jax.jit, static_argnums=(0, 5))
def update_step(forward, params, opt_state, batch_a, batch_u, optimizer):
    # Define a new loss function for grad calculation that closes over 'forward'
    def _loss_for_grad(p, ba, bu):
        pred = forward(p, ba)
        return mse(pred, bu)

    loss, grads = jax.value_and_grad(_loss_for_grad)(
        params, batch_a, batch_u
    )  # Differentiate w.r.t. params (the first arg of _loss_for_grad)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train(key, epochs=10000, n=32, train_size=512, batch=8):
    # dataset ------------------------------------------------
    key_train, key_test = jax.random.split(key)
    a_train, u_train = make_dataset(key_train, train_size, n)
    a_test, u_test = make_dataset(key_test, 64, n)

    # model -------------------------------------------------
    params = init_uno(key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    # batching helper
    def get_batch(step):
        idx = jnp.arange(batch) + (step * batch) % (train_size - batch)
        return a_train[idx], u_train[idx]

    # training loop ----------------------------------------
    t0 = time.time()
    for step in range(epochs):
        batch_a, batch_u = get_batch(step)
        params, opt_state, loss = update_step(
            uno_forward, params, opt_state, batch_a, batch_u, optimizer
        )

        if step % 40 == 0 or step == epochs - 1:
            test_loss = loss_fn(uno_forward, params, a_test, u_test)
            print(f"Step {step:4d} | train MSE {loss:8.4e} | test MSE {test_loss:8.4e}")

    print(f"Done in {time.time() - t0:.1f}s")
    return params


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    _ = train(key)
