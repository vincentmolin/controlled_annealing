import ideanneal
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

import ott
import ot as pot


def test_langevin():
    xs = jr.normal(jr.key(0), (10, 2))
    u = lambda x: jnp.sum(x**2, axis=-1) / 2
    beta = lambda t: t + 1

    entropic_transporter = ideanneal.transport.EntropicTransporter(u, beta)
    exact_transporter = ideanneal.transport.ExactTransporter(u, beta)

    lange = ideanneal.langevin.Langevin(u, beta)

    T = 5.0
    dt = 0.05
    t0 = 0.0
    rng = jr.key(0)

    t = t0
    x = xs
    while t < T:
        rng, key = jr.split(rng)
        x = lange.step(key, t, dt, x)
        t += dt
