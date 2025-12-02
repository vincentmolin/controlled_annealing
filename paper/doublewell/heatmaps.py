# %% Double well experiments
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax.random as jr

import doublewell
import hist
from doublewell import u, beta, beta_inv, T, normtime

sys.path.append("../")

from common import (
    Cacheman,
    ensuredir,
    save_plot_data,
    discretize_trajs,
    EXTERNAL_SAVE_PATH,
    make_gauss_init_sampler,
)

cm = Cacheman("cache")
langevin_experiment = cm.persist(doublewell.langevin_experiment)
controlled_langevin_experiment = cm.persist(doublewell.controlled_langevin_experiment)
pdsa_experiment = cm.persist(doublewell.pdsa_experiment)
controlled_pdsa_experiment = cm.persist(doublewell.controlled_pdsa_experiment)

# %% Plot configuration
EXPORT = True
EXTERNAL_SAVE = False
REFRESHALL = False

if EXTERNAL_SAVE:
    assert EXTERNAL_SAVE_PATH is not None
    SAVE_PATH = os.path.join(EXTERNAL_SAVE_PATH, "figs/doublewell")
else:
    SAVE_PATH = "figs/"

plt.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 300  # increase dpi for rasterized heatmaps


def single_hist_plot_just_image(ts, xm, H, fn=None, size=(2, 2), dpi=300, export=False):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(*size)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    hist.plot(ax, normtime(ts), xm, H)

    if export:
        dir = os.path.join(SAVE_PATH, "hists")
        ensuredir(dir)
        fig.savefig(os.path.join(dir, f"{fn}.png"), dpi=dpi)
    return fig, ax


def write_plot_data(d, fn, subdir="raw"):
    path = os.path.join(SAVE_PATH, subdir)
    ensuredir(path)
    save_plot_data(d, os.path.join(path, fn))


# %% Experiments 1

tm = np.linspace(0, T, 251)
xm = np.linspace(-3, 3, 250)  # histogram mesh

H_ground_truth = hist.ground_truth(u, beta, tm, xm, norm="dens")

ts, Xs = langevin_experiment(N=500, ITER=40, refresh=REFRESHALL)
Hlange = hist.discretized(Xs, xm)
tcs5, Xcs5 = controlled_langevin_experiment(
    N=5, ITER=4000, persist_as="cle_5", refresh=REFRESHALL
)
Hclang5 = hist.discretized(Xcs5, xm)
tcs, Xcs = controlled_langevin_experiment(
    N=10, ITER=2000, persist_as="cle_10", refresh=REFRESHALL
)
Hclang10 = hist.discretized(Xcs, xm)

tmp = ts
xmp = xm

trajs = pdsa_experiment(ITER=4000, normalized_velocities=True, refresh=REFRESHALL)
Xpdsa = discretize_trajs(trajs, tmp)
Hpdsa = hist.discretized(Xpdsa, xmp)

ctrajs = controlled_pdsa_experiment(
    N=10,
    ITER=2000,
    normalized_velocities=True,
    persist_as="cpdsa_10",
    refresh=REFRESHALL,
)  #
Xcpdsa = discretize_trajs(ctrajs, tmp)
Hcpdsa = hist.discretized(Xcpdsa, xmp)

ctrajs5 = controlled_pdsa_experiment(
    N=5, ITER=4000, normalized_velocities=True, persist_as="cpdsa_5", refresh=REFRESHALL
)  #
Xcpdsa5 = discretize_trajs(ctrajs5, tmp)
Hcpdsa5 = hist.discretized(Xcpdsa5, xmp)

single_hist_plot_just_image(tm, xm, H_ground_truth, fn="ground_truth", export=EXPORT)
single_hist_plot_just_image(ts, xm, Hlange, fn="langevin", export=EXPORT)
single_hist_plot_just_image(ts, xm, Hclang5, "clangevin_5", export=EXPORT)
single_hist_plot_just_image(ts, xm, Hclang10, fn="clangevin_10", export=EXPORT)
single_hist_plot_just_image(tmp, xmp, Hpdsa, "pdsa", export=EXPORT)
single_hist_plot_just_image(tmp, xmp, Hcpdsa, "cpdsa_10", export=EXPORT)
single_hist_plot_just_image(tmp, xmp, Hcpdsa5, "cpdsa_5", export=EXPORT)

# %% Experiments, offset start

tm = np.linspace(0, T, 251)
xm = np.linspace(-3, 3, 250)  # histogram mesh

std_init = 0.5
x0_inits = np.array([0.0, 1.0, 2.0, 3.0])[:, None]
N = 5
ITER = 1000

Hcs = []
Xcs = []

for x0_init in x0_inits:
    init_sampler = make_gauss_init_sampler(x0_init, std_init)
    ts, Xc = controlled_langevin_experiment(
        init_sampler=init_sampler,
        N=N,
        ITER=ITER,
        persist_as=f"cle_5_offset_{x0_init[0]:.1f}",
        refresh=REFRESHALL,
    )
    Xcs.append(Xc)
    Hcs.append(hist.discretized(Xc, xm))


fig, ax = plt.subplots(figsize=(8, 8))
hist.plot(ax, ts, xm, Hcs[3])
fig

traj_id = 15
traj = Xcs[3][:, traj_id * N : (traj_id + 1) * N, 0]
plt.plot(ts, traj)

Xcs[0].shape

# %%


tm = np.linspace(0, T, 251)
xm = np.linspace(-3, 3, 250)  # histogram mesh

std_init = 0.5
x0_inits = np.array([0.0, 1.0, 2.0, 3.0])[:, None]
N = 5
ITER = 1000
tend = 25.0
beta_t0 = 1.0
toffset = beta_inv(beta_t0)
beta_colder = lambda t: beta(toffset + t * (1 - toffset / tend))
plt.plot(tm, beta(tm))
plt.plot(tm, beta_colder(tm))

Hcs = []
Xcs = []

for x0_init in x0_inits:
    init_sampler = make_gauss_init_sampler(x0_init, std_init)
    ts, Xc = controlled_langevin_experiment(
        init_sampler=init_sampler,
        beta=beta_colder,
        N=N,
        ITER=ITER,
        persist_as=f"cle_5_colder_offset_{x0_init[0]:.1f}",
        refresh=REFRESHALL,
    )
    Xcs.append(Xc)
    Hcs.append(hist.discretized(Xc, xm))


fig, ax = plt.subplots(figsize=(8, 8))
hist.plot(ax, ts, xm, Hcs[3])
fig

traj_id = 15
traj = Xcs[3][:, traj_id * N : (traj_id + 1) * N, 0]
plt.plot(ts, traj)

Xcs[0].shape
