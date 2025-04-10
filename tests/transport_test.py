import ideanneal
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr


def test_transport(nextkey):
    xs = jr.normal(nextkey(), (10, 2))
    a = jnp.ones(10) / 10
    b = jr.uniform(nextkey(), (10,)) + 1e-3
    b = b / jnp.sum(b)

    u = lambda x: jnp.sum(x**2, axis=1) / 2
    beta = lambda t: t

    entropic_transporter = ideanneal.transport.EntropicTransporter(u, beta)
    exact_transporter = ideanneal.transport.ExactTransporter(u, beta)

    ent_T0 = entropic_transporter(1.0, 0.5, xs)
    exact_T0 = exact_transporter(1.0, 0.5, xs)

    assert ent_T0.shape == xs.shape
    assert exact_T0.shape == xs.shape

    print(ent_T0 - exact_T0)


def test_fast_call(nextkey):
    xs = jr.normal(nextkey(), (10, 2))
    a = jnp.ones(10) / 10
    b = jr.uniform(nextkey(), (10,)) + 1e-3
    b = b / jnp.sum(b)

    u = lambda x: jnp.sum(x**2, axis=1) / 2
    beta = lambda t: t

    exact_transporter = ideanneal.transport.ExactTransporter(u, beta)

    fst_call = exact_transporter.get_jit_call()
    t = 1.0
    h = 0.5
    x = xs
    exact = fst_call(t, h, x)
    assert exact.shape == xs.shape
