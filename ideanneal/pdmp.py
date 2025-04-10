import jax
import jax.numpy as jnp
import jax.random as jr

from pdmpx.timers import ConstantRateTimer
from pdmpx.poisson_time import ab_poisson_time
from pdmpx.queues import SimpleFactorQueue, Factor
from pdmpx.pdmp import PDMP, AbstractDynamics, AbstractKernel, AbstractTimer, TimerEvent
from pdmpx.utils.func import rng_wrap
from typing import NamedTuple, Callable, Any, Tuple

from .transport import ExactTransporter, EntropicTransporter, NullTransporter

vab_poisson_time = jax.vmap(ab_poisson_time, in_axes=(0, 0, 0))


class ParticleCloud(NamedTuple):
    t: float
    x: jnp.ndarray
    y: jnp.ndarray
    v: jnp.ndarray


class LinearDynamics(AbstractDynamics):
    def forward(self, dt: float, p: ParticleCloud):
        return ParticleCloud(
            t=p.t + dt,
            x=p.x + dt * (p.y + p.v),
            y=p.y,
            v=p.v,
        )


class BounceTimer(AbstractTimer):
    def __init__(self, u: Callable, beta: Callable, valid_time: float):
        self.u = u
        self.beta = beta
        self.valid_time = valid_time
        self.dynamics = LinearDynamics()

    def rate_fn(self, dt: float, px: ParticleCloud):
        pt = self.dynamics.forward(dt, px)
        grads = jax.vmap(jax.grad(self.u))(pt.x)
        dpot = jnp.sum(jnp.reshape(grads * pt.y, (pt.y.shape[0], -1)), axis=-1)
        return dpot

    def __call__(self, rng: Any, px: ParticleCloud) -> TimerEvent:
        ax, bx = jax.jvp(
            lambda dt: self.beta(px.t + dt) * self.rate_fn(dt, px), (0.0,), (1.0,)
        )
        runif = jr.uniform(rng, ax.shape)

        evx = vab_poisson_time(runif, ax, bx)
        evt = jnp.minimum(jnp.min(evx), self.valid_time)
        yfilx = evx == evt

        event = jax.lax.cond(
            evt < self.valid_time,
            lambda: TimerEvent(evt, dirty=1.0, params={"event_filter": yfilx}),
            lambda: TimerEvent(
                self.valid_time, dirty=0.0, params={"event_filter": yfilx}
            ),
        )
        return event


class BounceKernel(AbstractKernel):
    def __init__(self, u: Callable):
        self.u = u

    def __call__(
        self, rng, px: ParticleCloud, timer_event: TimerEvent
    ) -> ParticleCloud:
        """
        Vectorized reflection kernel.
        """

        def _refl(x, y):
            grad = jax.grad(self.u)(x)
            return y - 2 * jnp.dot(grad, y) * grad / jnp.dot(grad, grad)

        refl_ys = jax.vmap(_refl, in_axes=(0, 0))(px.x, px.y)
        yfilx = timer_event.params["event_filter"]
        ys = jnp.where(yfilx[..., None], refl_ys, px.y)
        return ParticleCloud(t=px.t, x=px.x, y=ys, v=px.v)


class RefreshmentKernel(AbstractKernel):
    def __init__(self, normalized=False):
        self.normalized = float(normalized)

    def __call__(self, rng, state, timer_event):
        y = jr.normal(rng, state.y.shape)
        if self.normalized:
            y = y / jnp.linalg.norm(y, axis=-1, keepdims=True)
            y *= self.normalized
        return ParticleCloud(t=state.t, x=state.x, y=y, v=state.v)


class ControlledPDSA(PDMP):
    def __init__(
        self,
        u,
        beta,
        valid_time,
        refreshment_rate=0.1,
        transporter=NullTransporter(),
        normalized_velocities=False,
    ):
        self.u = u
        self.beta = beta
        self.valid_time = valid_time
        self.normalized_velocities = normalized_velocities
        self.transporter = transporter

        self.bounce_timer = BounceTimer(self.u, self.beta, self.valid_time)
        self.bounce_kernel = BounceKernel(self.u)
        self.refreshment_kernel = RefreshmentKernel(self.normalized_velocities)
        self.refreshment_timer = ConstantRateTimer(refreshment_rate)

        self.factor_queue = SimpleFactorQueue(
            [
                Factor(self.bounce_timer, self.bounce_kernel),
                Factor(self.refreshment_timer, self.refreshment_kernel),
            ]
        )

        super().__init__(
            LinearDynamics(), self.factor_queue.timer, self.factor_queue.kernel
        )

    def get_next_event_halted(
        self, rng, state, dtlim, return_event=False
    ) -> Tuple[ParticleCloud, float, bool, bool]:
        timer_key, kernel_key = jax.random.split(rng, 2)
        timer_event = self.timer(timer_key, state)
        dt = jnp.minimum(timer_event.time, dtlim)
        state = self.dynamics.forward(dt, state)
        apply_kernel = timer_event.dirty * (dt < dtlim)
        state = jax.lax.cond(
            apply_kernel,
            lambda k, st, te: self.kernel(k, st, te),
            lambda rng, st, te: st,
            kernel_key,
            state,
            timer_event,
        )
        if return_event:
            return state, dt, apply_kernel, dt >= dtlim, timer_event
        else:
            return state, dt, apply_kernel, dt >= dtlim

    def simulate_controlled_leg(
        self, rng: Any, t0: float, p0: ParticleCloud, tlim: float
    ):
        v = self.transporter(t0, tlim - t0, p0.x) / (tlim - t0)
        state = ParticleCloud(t=t0, x=p0.x, y=p0.y, v=v)

        remaining_time = tlim - t0
        _next = jax.jit(rng_wrap(self.get_next_event_halted))
        while remaining_time > 0:
            rng, state, dt, dirty, done = _next(rng, state, remaining_time)
            remaining_time -= dt
            if done:
                break
        return state
