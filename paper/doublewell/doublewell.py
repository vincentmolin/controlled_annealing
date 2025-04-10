# %% Double well experiments
import sys
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import scipy
from ideanneal.util import RejectionSampler
import scipy.optimize

sys.path.append("../")

from common import (
    make_langevin_runner,
    make_controlled_langevin_runner,
    make_pdsa_runner,
    make_controlled_pdsa_runner,
)


class DoubleWell:
    def __init__(self):
        res = scipy.optimize.minimize_scalar(self.u_raw, bounds=[-2, -0.5])  # , -1.2)
        self.u_shift = res.fun

    def u_raw(self, x):
        return x**2 / 2 + jnp.cos(2 * (x - 1 / 4))

    def u(self, x):
        """
        u shifted so that min u = 0 ≃ u(-1.04)
        stabilizes numerical integration
        """
        return jnp.squeeze(x**2 / 2 + jnp.cos(2 * (x - 1 / 4)) - self.u_shift)


class Beta0Sampler:
    def __init__(self, u, beta0, C):
        self.rejection_sampler = RejectionSampler(
            u, beta0, scale=1 / np.sqrt(beta0), C=C
        )

    def __call__(self, rng, N):
        uint32seed = jr.bits(rng, dtype=np.uint32)
        rvs = self.rejection_sampler.sample(uint32seed, N)
        return jnp.expand_dims(rvs, 1)


T = 25
beta = lambda t: 0.25 + 25 * (t / T) ** 2
beta_inv = lambda b: np.sqrt(25.0 * (b - 0.25))
dw = DoubleWell()
u = dw.u
beta0sampler = Beta0Sampler(u, beta(0.0), C=0.5)


def normtime(simts):
    return simts / T


def langevin_experiment(N=500, ITER=25, t0=0.0, dt=0.025, tend=T, rng=jr.key(123)):
    runner = make_langevin_runner(u, beta, beta0sampler, t0, dt, tend)
    ts, Xs = runner(N, ITER, rng=rng)
    return ts, Xs


def controlled_langevin_experiment(
    N=10, ITER=1000, t0=0.0, dt=0.025, tend=T, steps_per_v=20, rng=jr.key(127)
):
    runner = make_controlled_langevin_runner(
        u, beta, beta0sampler, t0, dt, tend, steps_per_v=steps_per_v
    )
    ts, Xs = runner(N, ITER, rng=rng)
    return ts, Xs


def pdsa_experiment(
    N=5,
    ITER=2000,
    t0=0.0,
    tend=T,
    valid_time=0.2,
    refreshment_rate=0.1,
    normalized_velocities=True,
    rng=jr.key(1245),
):
    runner = make_pdsa_runner(
        u,
        beta,
        beta0sampler,
        t0,
        tend,
        valid_time,
        refreshment_rate,
        normalized_velocities,
    )
    trajs, _ = runner(N, ITER, rng)
    return trajs


def controlled_pdsa_experiment(
    N=10,
    ITER=500,
    t0=0.0,
    tend=T,
    valid_time=0.1,
    normalized_velocities=True,
    refreshment_rate=0.1,
    v_interval=0.5,
    rng=jr.key(1245),
):
    runner = make_controlled_pdsa_runner(
        u,
        beta,
        beta0sampler,
        t0,
        tend,
        valid_time,
        normalized_velocities,
        refreshment_rate,
        v_interval,
    )
    trajs, _ = runner(N, ITER, rng)
    return trajs
