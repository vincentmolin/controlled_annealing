# %% Double well experiments
from importlib import reload
import sys
import os
import numpy as np
import jax.numpy as jnp
import jax
import jax.random as jr
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage
import doublewell
import hist
from doublewell import u, beta, beta0sampler, T, normtime

sys.path.append("../")
import common

reload(common)
from common import (
    Cacheman,
    save_plot_data,
)

cm = Cacheman("cache")
langevin_experiment = cm.persist(doublewell.langevin_experiment)
controlled_langevin_experiment = cm.persist(doublewell.controlled_langevin_experiment)
pdsa_experiment = cm.persist(doublewell.pdsa_experiment)
controlled_pdsa_experiment = cm.persist(doublewell.controlled_pdsa_experiment)


def _cle(N, ITER=1000, refresh=False):
    return controlled_langevin_experiment(
        N=N, ITER=ITER, persist_as=f"cle_{N}", refresh=refresh
    )


# %% Experiments

tm = np.linspace(0, T, 251)
xm = np.linspace(-3, 3, 250)  # histogram mesh


# %% Plot configuration

REFRESHALL = False
EXPORT = True
EXTERNAL_SAVE = False

plt.style.use("seaborn-v0_8-paper")


# %% W2, ||v|| plot

H_ground_truth = hist.ground_truth(u, beta, tm, xm, norm="dens")
tm_plot = np.linspace(0, 1, len(tm))

ts, Xs = langevin_experiment(N=500, ITER=20)
Hlange = hist.discretized(Xs, xm)

smooth = lambda x: scipy.ndimage.gaussian_filter1d(
    np.concat([np.zeros((5,)), x]), sigma=3, mode="nearest"
)[5:]

Ns = [2, 5, 10]

w2_plot_data = {
    "t": normtime(tm),
    "n1": hist.wass2_auto(Hlange, H_ground_truth, xm, tm),
}

for N in [2, 5, 10]:
    _tc, _Xc = _cle(N)
    _Hc = hist.discretized(_Xc, xm)
    w2_plot_data[f"n{N}"] = hist.wass2_auto(_Hc, H_ground_truth, xm, tm)

for N in [1, *Ns]:
    w2_plot_data[f"n{N}_smooth"] = smooth(w2_plot_data[f"n{N}"])

save_plot_data(w2_plot_data, "figs/doublewell/raw/w2.dat", external_save=EXTERNAL_SAVE)

v_plot_data = {
    "t": normtime(tm[1:]),
    "normv": hist.wass2_auto(H_ground_truth[:, 1:], H_ground_truth[:, :-1], xm, tm[1:])
    / (tm[1] - tm[0]),
}

save_plot_data(v_plot_data, "figs/doublewell/raw/v.dat", external_save=EXTERNAL_SAVE)

### Inspection

if False == True:

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    vax = ax.twinx()
    vax.fill_between(
        v_plot_data["t"],
        np.zeros(len(tm) - 1),
        v_plot_data["normv"],
        alpha=0.225,
        label="v",
    )
    ax.plot(normtime(tm), w2_plot_data["n1_smooth"], label=r"$n=1$")
    for N in [2, 5, 10]:
        _tc, _Xc = _cle(N)
        _Hc = hist.discretized(_Xc, xm)
        w2nN = hist.wass2_auto(_Hc, H_ground_truth, xm, tm)
        ax.plot(normtime(tm), smooth(w2nN), label=f"$n={N}$")
    ax.set(
        xlabel=r"$t$",
        ylabel=r"$\mathcal{W}_2(\cdot, \mu_t)$",
        # title=r"Distance to the Gibbs curve",
        yticks=[0.0, 1.0],
        yticklabels=["$0$", "$1$"],
        xticks=[0.0, 1.0],
        xticklabels=["$0$", "$1$"],
        xlim=(0.0, 1.0),
        ylim=(0.0, ax.get_ylim()[1]),
    )
    vax.set(
        xlabel=r"$t$",
        ylabel=r"$|\mu'|(t)$",
        xlim=(0.0, 1.0),
        ylim=(0.0, 0.6),
        xticks=[],  # [0.0, 1.0],
        xticklabels=[],  # ["$0$", "$1$"],
        yticks=[0.0],
        yticklabels=[],
    )
    ax.legend()

    fig, ax = plt.subplots(figsize=(2.4, 2))
    ax.plot(
        normtime(tm[1:]),
        hist.wass2_auto(H_ground_truth[:, 1:], H_ground_truth[:, :-1], xm, tm[1:])
        / (tm[1] - tm[0]),
    )
    ax.set(
        xlabel=r"$t$",
        ylabel=r"$|\mu'|(t)$",
        xlim=(0.0, 1.0),
        ylim=(0.0, ax.get_ylim()[1]),
        xticks=[0.0, 1.0],
        xticklabels=["$0$", "$1$"],
        yticks=[0.0],
        yticklabels=[],
    )
    fig.tight_layout()
