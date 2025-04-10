from mpi4py import MPI

import sys
import os
import pickle
import json
import jax.random as jr
from doublewell import u, beta, T, beta0sampler  # pyright: ignore

sys.path.append("../")

from common import (  # pyright: ignore
    make_controlled_langevin_runner,
    make_controlled_pdsa_runner,
)


def compute(job):
    t0 = job["kwargs"]["t0"]

    if job["type"] == "cle":
        runner = make_controlled_langevin_runner(
            u,
            beta,
            beta0sampler,
            t0,
            **job["kwargs"],
            progress_bar=False,
        )
        ts, Xs = runner(job["N"], job["ITER"], rng=jr.key(job["rng"]))
        return ts, Xs
    elif job["type"] == "cpdsa":
        runner = make_controlled_pdsa_runner(
            u,
            beta,
            beta0sampler,
            **job["kwargs"],
            progress_bar=False,
        )
        trajs = runner(job["N"], job["ITER"], rng=jr.key(job["rng"]))
        return trajs
    else:
        print("ill specified job", job)


def jobid(job):
    return hash(json.dumps(job, sort_keys=True))


def work(job):
    if os.path.exists("mpi/" + job["name"] + ".id"):
        with open("mpi/" + job["name"] + ".id", "r") as f:
            oldid = int(f.read())
        if oldid == jobid(job):
            print("job already done", job)
            return
    try:
        out = compute(job)
        wj = {**job, "res": out}
        print(f"Writing job {job['name']} to disk")
        with open(f"mpi/{job['name']}", "wb") as f:
            pickle.dump(wj, f)
        with open("mpi/" + job["name"] + ".id", "w") as f:
            f.write(str(jobid(job)))
    except:
        print("failed on ", job)
    return


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()

    if rank == 0:
        with open("mpi/jobs.pkl", "rb") as f:
            jobs = pickle.load(f)
    else:
        jobs = None
    jobs = comm.bcast(jobs, root=0)

    job_index = rank

    while job_index < len(jobs):
        job = jobs[job_index]
        print(f"Worker {rank} starting {job['name']}, {job_index} of {len(jobs)}")
        wj = work(jobs[job_index])
        job_index += commsize
