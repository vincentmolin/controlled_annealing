import numpy as np
import ideanneal
import matplotlib as mpl
import matplotlib.pyplot as plt


def ground_truth(
    u,
    beta,
    tm=np.linspace(0, 25, 251),
    xm=np.linspace(-3, 3, 250),
    norm="dens",
):
    if norm == "prob":
        normalize = lambda p: p / np.sum(p)
    elif norm == "max":
        normalize = lambda p: p / np.max(p)
    elif norm == "dens":
        normalize = lambda p: p / (np.sum(p) * (xm[1] - xm[0]))
    else:
        raise NotImplementedError
    hsts = []
    for t in tm:
        p = np.exp(-beta(t) * u(xm))
        p = normalize(p)
        hsts.append(p)
    H = np.stack(hsts).T
    return H


def discretized(Xs, xm=np.linspace(-3, 3, 250)):
    """
    Xs.shape = [ts, N]
    xm are bin centers for compatibility with hist_ground_truth
    """
    h = xm[1] - xm[0]
    bins = np.linspace(xm[0] - h / 2, xm[-1] + h / 2, len(xm) + 1)
    hsts = []
    for i in range(Xs.shape[0]):
        hc, _ = np.histogram(Xs[i], bins, density=True)
        hsts.append(hc)
    return np.stack(hsts).T


def reshape(ts, H, t_mesh):
    Hm = np.zeros((H.shape[0], len(t_mesh)))
    for i in range(Hm.shape[0]):
        Hm[i] = np.interp(t_mesh, ts, H[i])
    return Hm


def wass2(H1, H2, xm):
    """
    H1.shape == H2.shape == [xm.shape[0], n_t]
    """
    assert np.all(H1.shape == H2.shape)
    return np.array(
        [
            ideanneal.util.wasserstein2_1d(xm, H1[:, i], H2[:, i])
            for i in range(H1.shape[1])
        ]
    )


def wass2_auto(H, H0, xm0, tm0, tlim=(0.0, 25.0)):
    """
    Computes W2 columnwise between the histogram H and H0, reshaping H if necessary
    """
    if not H.shape[1] == H0.shape[1]:
        ts = np.linspace(*tlim, H.shape[1])
        H = reshape(ts, H, tm0)
    return wass2(H, H0, xm0)


def plot(ax, time_mesh, x_mesh, H, norm=mpl.colors.AsinhNorm(1e-2)):
    """
    Plot the density H on the meshgrid (time_mesh, x_mesh)
    """
    T, X = np.meshgrid(time_mesh, x_mesh)
    cf = ax.pcolormesh(
        T,
        X,
        H,
        cmap="plasma",
        shading="gouraud",
        rasterized=True,
        norm=norm,
    )
    return cf
