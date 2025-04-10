import jax.numpy as jnp
import jax
import functools as ft
import scipy


def make_gauss_init_sampler(x0, std):
    def sampler(rng, n):
        return x0 + std * jax.random.normal(rng, (n, len(x0)))

    return sampler


@ft.partial(jax.jit, static_argnums=(1,))
def stride_min(arr, stride):
    """strided min over last axis"""
    new_shape = arr.shape[:-1] + (arr.shape[-1] // stride, stride)
    arr = arr.reshape(new_shape)
    return jnp.min(arr, initial=jnp.inf, where=jnp.logical_not(jnp.isnan(arr)), axis=-1)


def plot_percentiles(ax, ts, ys, color, qs=[25, 50, 75], linestyles=["--", "-", "--"]):
    for q, ls in zip(qs, linestyles):
        ax.plot(ts, jnp.percentile(ys, q, axis=1), color=color, linestyle=ls)


def smoothen(ys):
    return scipy.ndimage.gaussian_filter1d(ys, sigma=3, mode="nearest")


def ema(ys, w=0.2):
    ma = ys[0]
    mas = [ma]
    for y in ys[1:]:
        ma = w * y + (1 - w) * ma
        mas.append(ma)
    return jnp.array(mas)
