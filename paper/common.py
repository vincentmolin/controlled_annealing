import os
import numpy as np
import pdmpx
from pdmpx.utils.func import rng_wrap
import pickle
from typing import Sequence, NamedTuple, List
from ideanneal.pdmp import ParticleCloud, ControlledPDSA
from ideanneal.langevin import Langevin, ControlledLangevin
import ideanneal
import jax
import jax.random as jr
import jax.numpy as jnp
from tqdm.autonotebook import tqdm

EXTERNAL_SAVE_PATH = None
envpath = os.path.join(os.path.dirname(__file__), "paper.env")

if os.path.exists(envpath):
    with open(envpath, "r") as f:
        content = f.read()
    var, val = content.split("=")
    assert var == "EXTERNAL_SAVE_PATH"
    EXTERNAL_SAVE_PATH = val


def save_plot_data(d, fn, external_save=False):
    """d dict of plot data"""
    if external_save == True:
        if EXTERNAL_SAVE_PATH is None:
            raise ValueError("EXTERNAL_SAVE_PATH not set")
        fn = os.path.join(EXTERNAL_SAVE_PATH, fn)
    elif type(external_save) == str:
        fn = os.path.join(external_save, fn)

    if os.path.dirname(fn):
        ensuredir(os.path.dirname(fn))
    ks = d.keys()
    cols = [d[k] for k in ks]
    l = len(cols[0])
    assert all([len(col) == l for col in cols])
    with open(fn, "w") as f:
        f.write(" ".join(ks) + "\n")
        rows = [" ".join([str(col[i]) for col in cols]) for i in range(l)]
        f.write("\n".join(rows))


def ensuredir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Cacheman:
    def __init__(self, path="cache"):
        self.path = ensuredir(path)

    def _cache_path(self, f):
        return os.path.join(self.path, f + ".pkl")

    def persist(self, fn):
        def wrapper(*args, persist_as=None, refresh=False, **kwargs):
            cache = self._cache_path(persist_as or fn.__name__)
            if os.path.exists(cache) and not refresh:
                with open(cache, "rb") as file:
                    return pickle.load(file)
            result = fn(*args, **kwargs)
            with open(cache, "wb") as file:
                pickle.dump(result, file)
            return result

        wrapper.__name__ = fn.__name__
        return wrapper


def run_langevin(rng, rstep, x0, t0, dt, tend):
    """
    rstep: rng, t, dt, x -> rng, x
    """
    t = t0
    ts = [t]
    x = x0
    xs = [x]
    while t < tend:
        rng, x = rstep(rng, t, dt, x)
        t += dt
        ts.append(t)
        xs.append(x)
    return ts, xs


def make_langevin_runner(
    u, beta, beta0sampler, t0=0.0, dt=0.025, tend=25.0, fn_name=None, progress_bar=True
):
    pbar = tqdm if progress_bar else (lambda x: x)

    def langevin_runner(N, ITER, rng=jr.key(0)):
        langevin = Langevin(u, beta)
        rstep = jax.jit(rng_wrap(lambda *a, **k: (langevin.step(*a, **k),)))
        xss = []
        for i in pbar(range(ITER)):
            rng, ikey, key = jr.split(rng, 3)
            x0 = beta0sampler(ikey, N)
            ts, xs = run_langevin(key, rstep, x0, t0, dt, tend)
            ts = np.array(ts)
            xs = np.stack(xs)
            xss.append(xs)

        Xs = np.column_stack(xss)
        return ts, Xs

    if fn_name:
        langevin_runner.__name__ = fn_name
    return langevin_runner


def run_controlled_langevin(rng, rcstep, compute_v, x0, t0, dt, tend, steps_per_v):
    h = steps_per_v * dt
    t = t0
    ts = [t]
    x = x0
    xs = [x]
    while t < tend:
        v = compute_v(t, h, x)
        for i in range(steps_per_v):
            rng, x = rcstep(rng, t, dt, x, v)
            t += dt
            ts.append(t)
            xs.append(x)
            if t > tend:
                break
    return ts, xs


def make_controlled_langevin_runner(
    u,
    beta,
    beta0sampler,
    t0=0.0,
    dt=0.025,
    tend=25.0,
    steps_per_v=10,
    fn_name=None,
    progress_bar=True,
):
    pbar = tqdm if progress_bar else (lambda x: x)
    transporter = ideanneal.transport.ExactTransporter(u, beta)
    clange = ControlledLangevin(u, beta, transporter)
    fst_call = transporter.get_jit_call()
    compute_v = lambda t, h, x: fst_call(t, h, x) / h
    rcstep = jax.jit(rng_wrap(lambda *a, **k: (clange.step(*a, **k),)))

    def controlled_langevin_runner(N, ITER, rng=jr.key(0)):
        xss = []
        for i in pbar(range(ITER)):
            rng, ikey, key = jr.split(rng, 3)
            x0 = beta0sampler(ikey, N)
            ts, xs = run_controlled_langevin(
                key, rcstep, compute_v, x0, t0, dt, tend, steps_per_v
            )
            ts = np.array(ts)
            xs = np.stack(xs)
            xss.append(xs)

        Xs = np.column_stack(xss)
        return ts, Xs

    if fn_name:
        controlled_langevin_runner.__name__ = fn_name
    return controlled_langevin_runner


def run_pdsa(rng, rstep, p0: ParticleCloud, tend: float, maxiter=10000):
    p = p0
    ps = [p]
    tevs = []
    t = p0.t
    ts = [t]
    for i in range(maxiter):
        rng, p, dt, dirty, done, tev = rstep(rng, p, tend - t)
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


def make_pdsa_runner(
    u,
    beta,
    beta0sampler,
    t0=0.0,
    tend=25.0,
    valid_time=0.1,
    refreshment_rate=0.1,
    normalized_velocities=True,
    fn_name=None,
    progress_bar=True,
):
    pbar = tqdm if progress_bar else (lambda x: x)
    cpdsa = ControlledPDSA(
        u,
        beta,
        valid_time=valid_time,
        refreshment_rate=refreshment_rate,
        normalized_velocities=normalized_velocities,
    )
    step = jax.jit(
        rng_wrap(lambda *args: cpdsa.get_next_event_halted(*args, return_event=True))
    )

    def pdsa_runner(N, ITER, rng=jr.key(0)):
        trajs = []
        tss = []
        for i in pbar(range(ITER)):
            rng, key, ikey = jr.split(rng, 3)
            x0 = beta0sampler(ikey, N)
            y0 = jr.normal(key, x0.shape)
            if normalized_velocities:
                y0 = y0 / jnp.linalg.norm(y0, axis=1, keepdims=True)
            p0 = ParticleCloud(t0, x0, y0, jnp.zeros(x0.shape))
            ps, ts, _ = run_pdsa(key, step, p0, tend, 10000)
            trajs.append(to_traj(ps))
            tss.append(np.array(ts))
        return trajs, tss

    if fn_name:
        pdsa_runner.__name__ = fn_name
    return pdsa_runner


def run_controlled_pdsa(
    rng,
    rstep,
    compute_v,
    p0: ParticleCloud,
    tend: float,
    v_interval: float,
    maxiter=10000,
):
    v0 = compute_v(p0.t, v_interval, p0.x)
    p = ParticleCloud(p0.t, p0.x, p0.y, v0)
    ps = [p]
    tevs = []
    t = p0.t
    ts = [t]
    next_v = v_interval + p0.t

    for i in range(maxiter):
        rng, p, dt, dirty, done, tev = rstep(rng, p, next_v - t)
        t += dt
        if t >= tend:
            ps.append(p)
            tevs.append(tev)
            break
        elif t >= next_v:
            t = next_v
            v = compute_v(t, v_interval, p.x)
            p = ParticleCloud(t, p.x, p.y, v)
            next_v += v_interval
            dirty = True
        if dirty:
            ps.append(p)
            tevs.append(tev)
        if i == maxiter - 1:
            print("maxiter reached in simulate")
        ts.append(t)
    return ps, ts, tevs


def make_controlled_pdsa_runner(
    u,
    beta,
    beta0sampler,
    t0=0.0,
    tend=25.0,
    valid_time=0.1,
    normalized_velocities=True,
    refreshment_rate=0.1,
    v_interval=0.5,
    fn_name=None,
    progress_bar=True,
):
    pbar = tqdm if progress_bar else (lambda x: x)
    cpdsa = ControlledPDSA(
        u,
        beta,
        valid_time=valid_time,
        refreshment_rate=refreshment_rate,
        normalized_velocities=normalized_velocities,
    )
    step = jax.jit(
        rng_wrap(lambda *args: cpdsa.get_next_event_halted(*args, return_event=True))
    )

    transporter = ideanneal.transport.ExactTransporter(u, beta)
    fst_call = transporter.get_jit_call()
    compute_v = lambda t, h, x: fst_call(t, h, x) / h

    def controlled_pdsa_runner(N, ITER, rng=jr.key(0)):
        trajs = []
        tss = []
        for i in pbar(range(ITER)):
            rng, key, ikey = jr.split(rng, 3)
            x0 = beta0sampler(ikey, N)
            y0 = jr.normal(key, x0.shape)
            if normalized_velocities:
                y0 = y0 / jnp.linalg.norm(y0, axis=1, keepdims=True)
            p0 = ParticleCloud(t0, x0, y0, jnp.zeros(x0.shape))
            ps, ts, _ = run_controlled_pdsa(
                key, step, compute_v, p0, tend, v_interval=v_interval, maxiter=10000
            )
            trajs.append(to_traj(ps))
            tss.append(np.array(ts))
        return trajs, tss

    if fn_name:
        controlled_pdsa_runner.__name__ = fn_name
    return controlled_pdsa_runner


class PDSATraj(NamedTuple):
    ts: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    vs: np.ndarray


def to_traj(ps: Sequence[ParticleCloud]):
    ts = np.array([p.t for p in ps])
    xs = np.array([p.x for p in ps])
    ys = np.array([p.y for p in ps])
    vs = np.array([p.v for p in ps])
    return PDSATraj(ts, xs, ys, vs)


def discretize_trajs(trajs: Sequence[PDSATraj], tm, eps=1e-5):
    """
    trajs: [PDSATraj]
    tm: np.ndarray
    eps: tolerance for float32 errors in traj.ts
    """
    Xts = []
    if isinstance(trajs, PDSATraj):
        trajs = [trajs]
    for traj in trajs:
        tdfs = np.diff(traj.ts)
        fltr = np.array([True, *(np.abs(tdfs) >= eps)])
        xt = pdmpx.utils.discretize_trajectory(
            traj.xs[fltr], traj.ts[fltr], T=None, mesh=tm
        )
        Xts.append(np.squeeze(xt))
    return np.column_stack(Xts)
