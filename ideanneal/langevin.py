import jax
import jax.numpy as jnp
import jax.random as jr


class Langevin:
    def __init__(self, u, beta, speed=1):
        self.u = u
        self.beta = beta
        self.vu = jax.vmap(u)
        self.vgradu = jax.vmap(jax.grad(u))
        self.speed = speed

    def step(self, rng, t, dt, x) -> jax.Array:
        """
        Langevin step
        """
        noise = jr.normal(rng, x.shape)
        grad = self.vgradu(x)
        if self.speed != 1:
            dt = dt * self.speed
        return x - dt * grad + jnp.sqrt(2 * dt / self.beta(t)) * noise


class ControlledLangevin(Langevin):
    def __init__(self, u, beta, transporter, langevin_speed=1):
        self.transporter = transporter
        super().__init__(u, beta, speed=langevin_speed)

    def step(self, rng, t, dt, x, v):
        """
        Langevin step with transport drift
        """
        return super().step(rng, t, dt, x) + dt * v

    def compute_v(self, t, h, x):
        """
        Compute the drift v at time t
        """
        return self.transporter(t, h, x) / h

    def simulate(self, rng, x0, t0, dt, n_steps, steps_per_v, keep_every=0):
        """
        Simulate the process for n_steps
        """
        t = t0
        x = x0
        h = dt * steps_per_v

        @jax.jit
        def jitstep(rng, t, x, v):
            rng, key = jr.split(rng)
            x = self.step(key, t, dt, x, v)
            t += dt
            return rng, t, x

        if keep_every > 0:
            trajectory = [x]
        for i in range(n_steps):
            if i % steps_per_v == 0:
                v = self.compute_v(t, h, x)
            rng, t, x = jitstep(rng, t, x, v)
            if keep_every > 0 and i % keep_every == 0:
                trajectory.append(x)
        if keep_every > 0:
            return jnp.stack(trajectory)
        else:
            return x
