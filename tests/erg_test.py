import scipy.stats.tests
from ideanneal.pdmp import ControlledPDSA, ParticleCloud  # , simulate_controlled_leg
from pdmpx.utils.func import rng_wrap
from pdmpx.utils import discretize_trajectory
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.stats


def simulate_pdsa(
    rng, pdmp: ControlledPDSA, p0: ParticleCloud, T: float, maxiter=10000
):
    step = jax.jit(
        rng_wrap(lambda *args: pdmp.get_next_event_halted(*args, return_event=True))
    )

    p = p0
    ps = [p]
    tevs = []
    t = p0.t
    ts = [t]
    for i in range(maxiter):
        rng, p, dt, dirty, done, tev = step(rng, p, T - t)
        t += dt
        ts.append(t)
        if dirty:
            ps.append(p)
            tevs.append(tev)
        if done:
            ps.append(p)
            tevs.append(tev)
            break
        if i == maxiter - 1:
            print("maxiter reached in simulate")
    return ps, ts, tevs


def test_gaussian(nextkey):
    u = lambda x: jnp.sum(x**2, axis=-1) / 2
    pdmp = ControlledPDSA(
        u,
        beta=lambda t: 1.0,
        valid_time=100.0,
        normalized_velocities=True,
        refreshment_rate=0.5,
    )
    Tm = 500.0

    for N in [1, 5, 10]:
        T = Tm / N
        print(f"N={N}")
        x0 = jr.normal(nextkey(), (N, 2))
        y0 = jr.normal(nextkey(), (N, 2))
        v0 = jnp.zeros(x0.shape)
        p0 = ParticleCloud(0.0, x0, y0 / jnp.linalg.norm(y0, axis=1, keepdims=True), v0)
        ps, ts, tevs = simulate_pdsa(nextkey(), pdmp, p0, T, maxiter=1000)
        xs = discretize_trajectory(
            jnp.stack([p.x for p in ps]),
            jnp.stack([p.t for p in ps]),
            T,
            dt=N * T / (25),
        )
        xflat = jnp.ravel(xs)
        print(f"xflat mean {jnp.mean(xflat)}, std {jnp.std(xflat)}")
        test = scipy.stats.kstest(xflat, "norm", (0.0, 1.0))
        print(test.pvalue)
        assert test.pvalue >= 0.01

        xxs = jnp.reshape(xs, (-1, 2))
        # cov = jnp.cov(xxs.T)
        # print(cov)
        # print(jnp.mean(xxs, axis=0))
        nxs = jnp.sum(xxs**2, axis=1)
        ch2test = scipy.stats.kstest(nxs, "chi2", (2,))
        print(ch2test.pvalue)
        assert ch2test.pvalue >= 0.01
        # assert jnp.max(jnp.abs(jnp.eye(2) - cov)) < 0.2
        # assert jnp.max(jnp.abs(jnp.mean(xxs, axis=0))) < 0.1
