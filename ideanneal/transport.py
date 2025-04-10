import jax.numpy as jnp
import jax
import jax.random as jr
import ot as pot
import ott
from typing import NamedTuple, Callable
import functools as ft


class AbstractTransporter:
    def __call__(self, t, h, pts) -> jnp.ndarray:
        """
        Compute the transport map between pts and pts reweighted,
        at time t and time t+h. Returns the relative _shift_ of each point,
        that is, the Monge map T(x) minus x.

        Example: The identity transport map returns zeros.
        """
        pass


class NullTransporter:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, t, h, pts):
        """
        Compute the null transport map.
        """
        return jnp.zeros(pts.shape)


class ReweightingTransporter:
    def __init__(self, u: Callable, beta: Callable):
        self.u = u
        self.beta = beta

    def _diff_ws(self, t, h, pts):
        """
        Reweight particles, assuming $pts \\sim \\mu_t$.
        """
        log_bw = -self.u(pts) * (self.beta(t + h) - self.beta(t))
        z = jnp.sum(jnp.exp(log_bw))
        return jnp.exp(log_bw) / z


def sqdist(pts):
    """
    pts 2d array with shape == [n_particles, n_dims]
    """
    difs = pts[:, None, :] - pts
    C = jnp.einsum("ijk,ijk->ij", difs, difs)
    return C


class ExactTransporter(ReweightingTransporter):
    def __call__(self, t, h, pts):
        """
        Compute exact transport plan between pts and pts reweighted,
        and returns conditional projections of pts - pts
        """
        n = pts.shape[0]
        wths = jnp.squeeze(self._diff_ws(t, h, pts))
        wts = jnp.ones(pts.shape[0]) / pts.shape[0]

        C = sqdist(pts)  # pot.utils.dist(pts, metric="sqeuclidean")
        G = pot.emd(wts, wths, C)
        return n * G @ pts - pts

    def _set_up(self, t, h, pts):
        n = pts.shape[0]
        wts = jnp.ones(pts.shape[0]) / pts.shape[0]
        wths = jnp.squeeze(self._diff_ws(t, h, pts))
        C = sqdist(pts)
        return n, wts, wths, C

    def _marginalize(self, G, n, pts):
        return n * G @ pts - pts

    def get_jit_call(self):
        set_up = jax.jit(self._set_up)
        marginalize = jax.jit(self._marginalize)

        def call(t, h, pts):
            n, wts, wths, C = set_up(t, h, pts)
            G = pot.emd(wts, wths, C)
            return marginalize(G, n, pts)

        return call


class EntropicTransporter(ReweightingTransporter):
    def __init__(
        self, u: Callable, beta: Callable, epsilon=1e-2, entropy_correction=True
    ):
        super().__init__(u, beta)
        self.epsilon = epsilon
        self.entropy_correction = entropy_correction

    def __call__(self, t, h, pts):
        """
        Compute entropic transport map between pts and pts reweighted.
        """
        wths = jnp.squeeze(self._diff_ws(t, h, pts))
        wts = jnp.ones(pts.shape[0]) / pts.shape[0]

        geom = ott.geometry.pointcloud.PointCloud(pts, epsilon=self.epsilon)
        ot = ott.solvers.linear.solve(geom, wts, wths)
        dual_potentials = ot.to_dual_potentials()
        if not self.entropy_correction:
            return dual_potentials.transport(pts) - pts
        else:
            ot_corr = ott.solvers.linear.solve(geom)
            dual_potentials_corr = ot_corr.to_dual_potentials()
            return dual_potentials.transport(pts) - dual_potentials_corr.transport(pts)

    # def T_one_cloud_entropy_correction(self, t, h, pts, epsilon=1e-3):
    #     """
    #     Compute entropy corrected transport map between pts and pts reweighted
    #     """
    #     wths = jnp.squeeze(self._diff_ws(t, h, pts))
    #     wts = jnp.ones(pts.shape[0]) / pts.shape[0]

    #     geom = ott.geometry.pointcloud.PointCloud(pts, epsilon=epsilon)
    #     ot = ott.solvers.linear.solve(geom, wts, wths)
    #     ot_corr = ott.solvers.linear.solve(geom)
    #     dp = ot.to_dual_potentials()
    #     dp_corr = ot_corr.to_dual_potentials()
    #     return dp.transport(pts) - dp_corr.transport(pts)


class GaussianTransporter(ReweightingTransporter):
    def __call__(self, t, h, pts):
        """
        Compute transport map between gaussian approximations
        of pts and pts reweighted.
        """
        wths = jnp.squeeze(self._diff_ws(t, h, pts))
        Gt = ott.tools.gaussian_mixture.gaussian.Gaussian.from_samples(pts)
        Gth = ott.tools.gaussian_mixture.gaussian.Gaussian.from_samples(pts, wths)
        return Gt.transport(Gth, pts) - pts


class TwoCloudsTransporter(EntropicTransporter):
    # @ft.partial(jax.jit, static_argnames=("correction",))
    def __call__(self, t, h, pts0, pts1, epsilon=1e-2, correction=True):
        """
        Compute the transport map from pts0 at time t to pts1 at time t+h,
        corrected by the transport map from pts0 to pts1 at time t.
        """
        wths = jnp.squeeze(self._diff_ws(t, h, pts1))
        wts = jnp.ones(pts0.shape[0]) / pts0.shape[0]

        geom = ott.geometry.pointcloud.PointCloud(pts0, pts1, epsilon=epsilon)
        oth = ott.solvers.linear.solve(
            geom,
            wts,
            wths,
        )
        t_uncorrected = oth.to_dual_potentials().transport(pts0)
        if not correction:
            return t_uncorrected - pts0
        ot = ott.solvers.linear.solve(geom)
        t_correction = ot.to_dual_potentials().transport(pts0)
        return t_uncorrected - t_correction
