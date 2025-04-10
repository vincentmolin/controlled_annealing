# import jax.numpy as jnp
# import jax.random as jr
import numpy as np
import scipy
import ot as pot

# import ott
# import ot
# import pdmpx
# from typing import NamedTuple
# import functools as ft
# from tqdm import tqdm


def wasserstein2_1d(x_mesh, a, b):
    """
    Compute the Wasserstein-2 distance between two discretized 1D distributions
    with densities a and b, respectively, on the mesh x_mesh.
    """
    return np.sqrt(
        pot.lp.emd2_1d(
            x_mesh, x_mesh, a / np.sum(a), b / np.sum(b), metric="sqeuclidean"
        )
    )


class RejectionSampler:
    def __init__(self, u, beta, loc=0.0, scale=1.0, C=1.0):
        """
        Rejection sampler for densities on RR p(x) = exp(-beta u(x)) / Z with a Gaussian reference measure.
        Only useful for demonstration purposes, for instance starting particles at stationarity
        """
        self.u = u
        self.beta = beta
        Z, _ = scipy.integrate.quad(lambda x: np.exp(-beta * u(x)), -np.inf, np.inf)
        self.Z = Z
        self.M = 1.0 + C
        self.p = lambda x: np.exp(-beta * u(x)) / Z  # noqa
        self.proposal = scipy.stats.norm(loc, scale)
        self.g = self.proposal.pdf

    def sample(self, seed, N):
        np.random.seed(seed)
        samps = []
        while len(samps) < N:
            x = self.proposal.rvs()
            U = np.random.rand()
            a = self.p(x) / (self.M * self.g(x))
            if a > 1.0:
                print("Warning: a > 1.0, increase C")
            if U < a:
                samps.append(x)
        return np.array(samps)
