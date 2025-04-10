import ideanneal
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from pdmpx.queues import SimpleFactorQueue
from pdmpx.timers import ConstantRateTimer
from ideanneal.pdmp import (
    ControlledPDSA,
    BounceTimer,
    BounceKernel,
    ParticleCloud,
    RefreshmentKernel,
)

import ott
import ot as pot


def test_bounce_kernels(nextkey):
    N = 3
    x0 = jnp.array([[1.0, 1.0], [-1.0, -1.0], [2.0, 0.0]])
    y0 = jnp.array([-x0[0], -x0[1], x0[2]])
    v0 = jnp.zeros(x0.shape)
    p0 = ParticleCloud(0.0, x0, y0, v0)

    u = lambda x: jnp.sum(x**2, axis=-1) / 2
    beta = lambda t: 1.0

    bt = BounceTimer(u, beta, valid_time=20.0)

    ev = bt(nextkey(), p0)

    assert jnp.all(ev.params["event_filter"] == jnp.array([False, False, True]))

    bk = BounceKernel(u)

    ns = bk(nextkey(), p0, ev)

    assert jnp.allclose(ns.y[0], p0.y[0])
    assert jnp.allclose(ns.y[1], p0.y[1])
    assert jnp.allclose(ns.y[2], -p0.y[2])

    p1 = ParticleCloud(p0.t, p0.x, jnp.array([-x0[0], -x0[1], [2.0, 2.0]]), p0.v)
    ev1 = bt(nextkey(), p1)
    ns1 = bk(nextkey(), p1, ev1)
    assert jnp.allclose(ns1.y[2], jnp.array([-2.0, 2.0]))


def test_factor_queue(nextkey):
    N = 3
    x0 = jnp.array([[1.0, 1.0], [-1.0, -1.0], [2.0, 0.0]])
    y0 = jnp.array([-x0[0], -x0[1], x0[2]])
    v0 = jnp.zeros(x0.shape)
    p0 = ParticleCloud(0.0, x0, y0, v0)

    u = lambda x: jnp.sum(x**2, axis=-1) / 2
    beta = lambda t: 1.0 + t

    pdmp = ControlledPDSA(u, beta, valid_time=20.0, refreshment_rate=0.0)
    sfq = pdmp.factor_queue
    ev = sfq.timer(nextkey(), p0)
    ns = sfq.kernel(nextkey(), p0, ev)

    assert jnp.allclose(ns.y[0], p0.y[0])
    assert jnp.allclose(ns.y[1], p0.y[1])
    assert jnp.allclose(ns.y[2], -p0.y[2])

    p1 = ParticleCloud(p0.t, p0.x, jnp.array([-x0[0], -x0[1], [2.0, 2.0]]), p0.v)
    ev1 = sfq.timer(nextkey(), p1)
    ns1 = sfq.kernel(nextkey(), p1, ev1)
    assert jnp.allclose(ns1.y[2], jnp.array([-2.0, 2.0]))


def test_as_bps(nextkey):
    N = 5

    x0 = jr.normal(nextkey(), (N, 2))
    y0 = jr.normal(nextkey(), (N, 2))
    v0 = jnp.zeros(x0.shape)

    u = lambda x: jnp.sum(x**2, axis=-1) / 2
    beta = lambda t: 1.0

    pdmp = ControlledPDSA(
        u, beta, 0.5, 1.0 / N, normalized_velocities=True
    )


def test_pdmp(nextkey):
    N = 3
    x0 = jr.normal(nextkey(), (N, 2))
    y0 = jr.normal(nextkey(), (N, 2))
    v0 = jnp.zeros(x0.shape)

    u = lambda x: jnp.sum(x**2, axis=-1) / 2
    beta = lambda t: t + 1

    entropic_transporter = ideanneal.transport.EntropicTransporter(u, beta)
    exact_transporter = ideanneal.transport.ExactTransporter(u, beta)

    pdmp = ControlledPDSA(u, beta, 0.5, 0.1)

    p0 = ParticleCloud(0.0, x0, y0, v0)
    p1 = pdmp.simulate_controlled_leg(nextkey(), 0.0, p0, 0.5)
