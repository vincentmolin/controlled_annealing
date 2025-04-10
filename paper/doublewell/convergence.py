import sys
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import hist
from doublewell import u, beta, T

sys.path.append("../")

from common import (
    ensuredir,
    save_plot_data,
    discretize_trajs,
)

# %% Convergence

EXTERNAL_SAVE = False
DIR = "doublewell/raw" if EXTERNAL_SAVE else "raw"
ITER = 1000

cl_default = ({"ITER": ITER}, {"t0": 0.0, "tend": T, "dt": 0.025})
cl_params = [
    ({"N": 5}, {"steps_per_v": 20}),
    ({"N": 5}, {"steps_per_v": 4}),
    ({"N": 5}, {"steps_per_v": 1}),
    ({"N": 10}, {"steps_per_v": 20}),
    ({"N": 10}, {"steps_per_v": 4}),
    ({"N": 10}, {"steps_per_v": 1}),
    ({"N": 25}, {"steps_per_v": 20}),
    ({"N": 25}, {"steps_per_v": 4}),
    ({"N": 25}, {"steps_per_v": 1}),
    ({"N": 50}, {"steps_per_v": 20}),
    ({"N": 50}, {"steps_per_v": 4}),
    ({"N": 50}, {"steps_per_v": 1}),
]
pd_default = (
    {"ITER": ITER},
    {
        "t0": 0.0,
        "tend": T,
        "refreshment_rate": 0.1,
        "normalized_velocities": True,
        "valid_time": 0.05,
    },
)
pd_params = [
    ({"N": 5}, {"v_interval": T / 50}),
    ({"N": 5}, {"v_interval": T / 250}),
    ({"N": 5}, {"v_interval": T / 1000}),
    ({"N": 10}, {"v_interval": T / 50}),
    ({"N": 10}, {"v_interval": T / 250}),
    ({"N": 10}, {"v_interval": T / 1000}),
    ({"N": 25}, {"v_interval": T / 50}),
    ({"N": 25}, {"v_interval": T / 250}),
    ({"N": 25}, {"v_interval": T / 1000}),
    ({"N": 50}, {"v_interval": T / 50}),
    ({"N": 50}, {"v_interval": T / 250}),
    ({"N": 50}, {"v_interval": T / 1000}),
]

cl_jobs = []
for p, kw in cl_params:
    p: Dict[str, Any] = {**(cl_default[0]), **p, "rng": len(cl_jobs)}
    kw = {**(cl_default[1]), **kw}
    N = p["N"]
    vsteps = int(kw["tend"] / kw["dt"]) // kw["steps_per_v"]
    p["name"] = f"cle_n{N}_v{vsteps}"
    p["type"] = "cle"
    p["kwargs"] = kw
    cl_jobs.append(p)

pd_jobs = []
for p, kw in pd_params:
    p: Dict[str, Any] = {**(pd_default[0]), **p, "rng": len(cl_jobs) + len(pd_jobs)}
    kw = {**(pd_default[1]), **kw}
    N = p["N"]
    vsteps = int(kw["tend"] / kw["v_interval"])
    p["name"] = f"cpdsa_n{N}_v{vsteps}"
    p["type"] = "cpdsa"
    p["kwargs"] = kw
    pd_jobs.append(p)

jobs = pd_jobs + cl_jobs

ensuredir("mpi")
with open("mpi/jobs.pkl", "wb") as f:
    pickle.dump(jobs, f)

# .... mpiexec -n 8 python mpi.py

xm = np.linspace(-3, 3, 250)
tm = np.linspace(0, T, 251)
H_ground_truth = hist.ground_truth(u, beta, tm, xm)


def tmpplt(ts, Xs):
    H = hist.discretized(Xs, xm)
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    H = hist.discretized(Xs, xm)
    hist.plot(ax, ts, xm, H)
    return fig, ax


def loadpath(pth):
    with open(pth, "rb") as f:
        return pickle.load(f)


def getjob(job):
    return loadpath("mpi/" + job["name"])  # + ".pkl")


wjs = []
w2s = []
for job in jobs:
    wj = getjob(job)
    if wj["type"] == "cle":
        ts, Xs = wj["res"]
    else:
        trajs = wj["res"][0]
        ts = tm
        try:
            Xs = discretize_trajs(trajs, ts)
        except:  # noqa
            print(f"run {wj['name']} malformed")
            probs = []
            for i, traj in enumerate(trajs):
                fltr = np.array([True, *(traj.ts[1:] != traj.ts[:-1])])
                flts = traj.ts[fltr]
                if np.any(flts[1:] < flts[:-1]):
                    probs.append(i)
            print("found", len(probs), "offending trajectories of", len(trajs))
            fltr_trajs = [trajs[i] for i in range(len(trajs)) if i not in probs]
            Xs = discretize_trajs(fltr_trajs, ts)
    wj["ts"] = ts
    wj["Xs"] = Xs
    H = hist.discretized(Xs, xm)
    wj["H"] = H
    wjs.append(wj)
    w2 = hist.wass2_auto(H, H_ground_truth, xm, tm)
    w2s.append(w2)


maxs = [np.max(w2) for w2 in w2s]
for idx in np.argsort(maxs):
    print(jobs[idx]["name"] + f"\t {maxs[idx]}")

avgs = [np.mean(w2) for w2 in w2s]
for idx in np.argsort(avgs):
    print(jobs[idx]["name"] + f"\t {avgs[idx]}")

plot_data_v_cols_n_rows = {"n": [5, 10, 25, 50]}
for n in plot_data_v_cols_n_rows["n"]:
    for job in jobs:
        if job["N"] == n:
            key = "_".join(job["name"].split("_")[0::2])
            plot_data_v_cols_n_rows.setdefault(key + "_max", [])
            plot_data_v_cols_n_rows.setdefault(key + "_mean", [])
            plot_data_v_cols_n_rows[key + "_mean"].append(avgs[jobs.index(job)])
            plot_data_v_cols_n_rows[key + "_max"].append(maxs[jobs.index(job)])

save_plot_data(
    plot_data_v_cols_n_rows,
    os.path.join(DIR, "conv_v_cols_n_rows.dat"),
    external_save=EXTERNAL_SAVE,
)

fig, ax = plt.subplots()
for i, vn in enumerate(["v50", "v250", "v1000"]):
    jobsv = [job for job in jobs if job["name"].endswith(vn)]
    cjobs = [job for job in jobsv if job["type"] == "cle"]
    ns = [job["N"] for job in cjobs]
    cw2ns = [w2s[jobs.index(job)] for job in cjobs]
    pjobs = [job for job in jobsv if job["type"] == "cpdsa"]
    pns = [job["N"] for job in pjobs]
    pw2ns = [w2s[jobs.index(job)] for job in pjobs]

    ax.plot(ns, [np.max(cw2n) for cw2n in cw2ns], "o-", c=f"C{i}", label="CLE " + vn)
    ax.plot(ns, [np.max(pw2n) for pw2n in pw2ns], "x--", c=f"C{i}", label="CPDSA " + vn)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks([5, 10, 25, 50])
ax.set_xticklabels(["5", "10", "25", "50"])
ax.legend()

fig, axss = plt.subplots(4, 3, figsize=(10, 14))
axs = axss.flatten()
i = 0
for wj in wjs:
    if wj["type"] == "cpdsa":
        hist.plot(axs[i], tm, xm, wj["H"])
        axs[i].set_title(wj["name"])
        i += 1

fig, axss = plt.subplots(4, 3, figsize=(10, 14))
axs = axss.flatten()
i = 0
for wj in wjs:
    if wj["type"] == "cle":
        hist.plot(axs[i], wj["ts"], xm, wj["H"])
        axs[i].set_title(wj["name"])
        i += 1
