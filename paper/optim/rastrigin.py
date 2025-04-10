import numpy as np
import jax.numpy as jnp
import jax
import jax.random as jr
import functools as ft
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import scipy
import os

from optim import (
    make_gauss_init_sampler,
    stride_min,
    ema,
)


sys.path.append("../")
from common import (
    Cacheman,
    ensuredir,
    save_plot_data,
    make_langevin_runner,
    make_controlled_langevin_runner,
    make_pdsa_runner,
    make_controlled_pdsa_runner,
    discretize_trajs,
)

cm = Cacheman("cache")


class Rastrigin:
    def __init__(self, A=1):
        self.obj = lambda x: x.shape[0] + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x))

    def __call__(self, x: jax.Array):
        if x.ndim == 1:
            x = x[None, :]
        return jnp.squeeze(jax.vmap(self.obj)(x))


EXTERNAL_SAVE = False
REFRESHALL = False

ddim = 10
init_sampler = make_gauss_init_sampler(3 * jnp.ones((ddim,)), 0.05)
t0 = 0.0
dt = 0.005
STEPS = 500
T = dt * STEPS
beta = lambda t: 0.1 + 5 * t / T
PDSA_TIMESCALE = 20  # to get comparable trajectory lengths
pdsa_beta = lambda t: beta(t / PDSA_TIMESCALE)
pdsa_T = T * PDSA_TIMESCALE


rastrigin_langevin_runner = cm.persist(
    make_langevin_runner(
        Rastrigin(), beta, init_sampler, t0, dt, T, "rastrigin_langevin"
    )
)
rastrigin_cl_runner = cm.persist(
    make_controlled_langevin_runner(
        Rastrigin(),
        beta,
        init_sampler,
        t0,
        dt,
        T,
        steps_per_v=STEPS // 25,
        fn_name="rastrigin_controlled_langevin",
    )
)
rastrigin_pdsa_runner = cm.persist(
    make_pdsa_runner(
        Rastrigin(),
        pdsa_beta,
        init_sampler,
        t0,
        pdsa_T,
        valid_time=0.1,
        normalized_velocities=True,
        refreshment_rate=T / 10,
        fn_name="rastrigin_pdsa",
    )
)
rastrigin_cpdsa_runner = cm.persist(
    make_controlled_pdsa_runner(
        Rastrigin(),
        pdsa_beta,
        init_sampler,
        t0,
        pdsa_T,
        valid_time=0.1,
        normalized_velocities=True,
        refreshment_rate=T / 10,
        v_interval=T / 25,
        fn_name="rastrigin_controlled_pdsa",
    )
)

rastrigin_x_optim = jnp.zeros((ddim,))
N = 5
ITER = 2000

ts, Xs = rastrigin_langevin_runner(
    N, ITER, rng=jr.key(0), refresh=REFRESHALL
)  # , refresh=True)
tcs, Xcs = rastrigin_cl_runner(
    N, ITER, rng=jr.key(1), refresh=REFRESHALL
)  # , refresh=True)
trajs_ps, ts_ps = rastrigin_pdsa_runner(N, ITER, rng=jr.key(3), refresh=REFRESHALL)
Xps = discretize_trajs(trajs_ps, jnp.linspace(0, pdsa_T, len(ts)))
trajs_cps, ts_cps = rastrigin_cpdsa_runner(N, ITER, rng=jr.key(2), refresh=REFRESHALL)
Xcps = discretize_trajs(trajs_cps, jnp.linspace(0, pdsa_T, len(ts)))

u = jax.vmap(Rastrigin())
uXs = u(Xs)
uXcs = u(Xcs)
uXps = u(Xps)
uXcps = u(Xcps)

uXs_bst = stride_min(uXs, N)
uXs_bst_10N = stride_min(uXs, N * 10)
uXcs_bst = stride_min(uXcs, N)
uXps_bst = stride_min(uXps, N)
uXcps_bst = stride_min(uXcps, N)


rastrigin_plot_data = {
    "t": np.linspace(0, 1, len(ts)),
}
rastrigin_at_tmiddle = {}
rastrigin_at_tend = {}

for ubst, nm in [
    (uXs_bst, "langevin_bo5"),
    (uXcs_bst, "clange5_bo5"),
    (uXs_bst_10N, "langevin_bo50"),
    (uXps_bst, "pdsa_bo5"),
    (uXcps_bst, "cpdsa5_bo5"),
]:
    key = nm
    ubst_mean = jnp.mean(ubst, axis=1)
    ubst_med = jnp.median(ubst, axis=1)
    ubst_std = jnp.std(ubst, axis=1)
    ubst_p25 = jnp.percentile(ubst, 25, axis=1)
    ubst_p75 = jnp.percentile(ubst, 75, axis=1)
    rastrigin_plot_data[key + "_mean"] = ubst_mean
    rastrigin_plot_data[key + "_mean_smooth"] = ema(ubst_mean, 0.2)
    rastrigin_plot_data[key + "_med"] = ubst_mean
    rastrigin_plot_data[key + "_med_smooth"] = ema(ubst_mean, 0.2)
    rastrigin_plot_data[key + "_std"] = ubst_std
    rastrigin_plot_data[key + "_p25"] = ubst_p25
    rastrigin_plot_data[key + "_p25_smooth"] = ema(ubst_p25, 0.2)
    rastrigin_plot_data[key + "_p75"] = ubst_p75
    rastrigin_plot_data[key + "_p75_smooth"] = ema(ubst_p75, 0.2)

    if nm != "langevin_bo50":
        rastrigin_at_tmiddle[key] = ubst[STEPS // 2 + 1]
        rastrigin_at_tend[key] = ubst[-1]

save_plot_data(
    rastrigin_plot_data, "figs/raw/rastrigin.dat", external_save=EXTERNAL_SAVE
)
save_plot_data(
    rastrigin_at_tmiddle, "figs/raw/rastrigin_tmid.dat", external_save=EXTERNAL_SAVE
)
save_plot_data(
    rastrigin_at_tend, "figs/raw/rastrigin_tend.dat", external_save=EXTERNAL_SAVE
)
