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


class Rosenbrock:
    def __init__(self, c=5):
        self.c = c
        self.obj = lambda x: jnp.sum(
            self.c * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2
        )

    def __call__(self, x: jax.Array):
        if x.ndim == 1:
            x = x[None, :]
        return jnp.squeeze(jax.vmap(self.obj)(x))


EXTERNAL_SAVE = False
REFRESHALL = False

ddim = 10
init_sampler = make_gauss_init_sampler(jnp.array([-1.5, *jnp.ones((ddim - 1,))]), 0.05)
t0 = 0.0
dt = 0.01
STEPS = 500
T = dt * STEPS
beta = lambda t: 0.1 + 10.0 * t / T

PDSA_TIMESCALE = 10  # to get comparable trajectory lengths
pdsa_beta = lambda t: beta(t / PDSA_TIMESCALE)
pdsa_T = T * PDSA_TIMESCALE

rosenbrock_langevin_runner = cm.persist(
    make_langevin_runner(
        Rosenbrock(), beta, init_sampler, t0, dt, T, "rosenbrock_langevin"
    )
)
rosenbrock_cl_runner = cm.persist(
    make_controlled_langevin_runner(
        Rosenbrock(),
        beta,
        init_sampler,
        t0,
        dt,
        T,
        steps_per_v=STEPS // 25,
        fn_name="rosenbrock_controlled_langevin",
    )
)
rosenbrock_pdsa_runner = cm.persist(
    make_pdsa_runner(
        Rosenbrock(),
        pdsa_beta,
        init_sampler,
        t0,
        pdsa_T,
        valid_time=0.1,
        normalized_velocities=True,
        refreshment_rate=T / 10,
        fn_name="rosenbrock_pdsa",
    )
)
rosenbrock_cpdsa_runner = cm.persist(
    make_controlled_pdsa_runner(
        Rosenbrock(),
        pdsa_beta,
        init_sampler,
        t0,
        pdsa_T,
        valid_time=0.1,
        normalized_velocities=True,
        refreshment_rate=T / 10,
        v_interval=T / 25,
        fn_name="rosenbrock_controlled_pdsa",
    )
)
rosenbrock_x_optim = jnp.ones((ddim,))


N = 5
ITER = 2000
ts, Xs = rosenbrock_langevin_runner(N, ITER, rng=jr.key(0), refresh=REFRESHALL)
tcs, Xcs = rosenbrock_cl_runner(N, ITER, rng=jr.key(1), refresh=REFRESHALL)
trajs_ps, ts_ps = rosenbrock_pdsa_runner(N, ITER, rng=jr.key(3), refresh=REFRESHALL)
Xps = discretize_trajs(trajs_ps, np.linspace(t0, pdsa_T, len(ts)))
trajs_cps, ts_cps = rosenbrock_cpdsa_runner(N, ITER, rng=jr.key(2), refresh=REFRESHALL)
Xcps = discretize_trajs(trajs_cps, np.linspace(t0, pdsa_T, len(ts)))


# %% Rosenbrock plot


uXs = jax.vmap(Rosenbrock())(Xs)
uXcs = jax.vmap(Rosenbrock())(Xcs)
uXps = jax.vmap(Rosenbrock())(Xps)
uXcps = jax.vmap(Rosenbrock())(Xcps)

uXs_bst = stride_min(uXs, N)
uXcs_bst = stride_min(uXcs, N)
uXps_bst = stride_min(uXps, N)
uXcps_bst = stride_min(uXcps, N)

rosenbrock_plot_data = {
    "t": np.linspace(0, 1, len(ts)),
}
rosenbrock_at_tmiddle = {}
rosenbrock_at_tend = {}

for ubst, nm in [
    (uXs_bst, "langevin"),
    (uXcs_bst, "clange5"),
    (uXps_bst, "pdsa"),
    (uXcps_bst, "cpdsa5"),
]:
    key = f"{nm}_bo5"
    ubst_mean = jnp.mean(ubst, axis=1)
    ubst_med = jnp.median(ubst, axis=1)
    ubst_std = jnp.std(ubst, axis=1)
    rosenbrock_plot_data[key + "_mean"] = ubst_mean
    rosenbrock_plot_data[key + "_mean_smooth"] = ema(ubst_mean, 0.2)
    rosenbrock_plot_data[key + "_med"] = ubst_mean
    rosenbrock_plot_data[key + "_med_smooth"] = ema(ubst_mean, 0.2)
    rosenbrock_plot_data[key + "_std"] = ubst_std

    rosenbrock_at_tmiddle[key] = ubst[STEPS // 2 + 1]
    rosenbrock_at_tend[key] = ubst[-1]

save_plot_data(
    rosenbrock_plot_data, "figs/raw/rosenbrock.dat", external_save=EXTERNAL_SAVE
)
save_plot_data(
    rosenbrock_at_tmiddle, "figs/raw/rosenbrock_tmid.dat", external_save=EXTERNAL_SAVE
)
save_plot_data(
    rosenbrock_at_tend, "figs/raw/rosenbrock_tend.dat", external_save=EXTERNAL_SAVE
)


if False == True:
    FIGSIZE = (2.8, 2.5)
    QUANTILES = False
    SMOOTH = True
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if QUANTILES:
        for i, ys in enumerate([uXs_bst, uXcs_bst, uXps_bst, uXcps_bst]):
            ax.fill_between(
                ts,
                jnp.percentile(ys, 25, axis=1),
                jnp.percentile(ys, 75, axis=1),
                alpha=0.25,
                color=f"C{i}",
            )

    for i, (ys, lab) in enumerate(
        [
            (uXs_bst, r"L"),
            (uXcs_bst, r"CL, $n=5$"),
            (uXps_bst, r"PDSA"),
            (uXcps_bst, r"CPDSA, $n=5$"),
        ]
    ):
        ys = jnp.mean(ys, axis=1)
        if SMOOTH:
            ys = ema(ys, 0.2)
            # ys = smoothen(ys)
        ax.plot(ts, ys, color=f"C{i}", label=lab)
    ax.set(xlabel=r"$t$", ylabel=r"$U_1$")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()

    df = np.stack(list(rosenbrock_at_tend.values())).transpose()
    plt.violinplot(df[:, 0])

    rat_l = np.array(rosenbrock_at_tend["langevin_bo5"])
    jnp.mean(jnp.array([jnp.nan, 1.23]))
    np.mean(df[:, 0])
    plt.violinplot(df, showextrema=False, showmedians=True)
    plt.violinplot(
        [np.log(vals) for vals in rosenbrock_at_tmiddle.values()], showextrema=False
    )
