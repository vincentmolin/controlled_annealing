import jax
import jax.random as jr
import jax.numpy as jnp
import ideanneal
import matplotlib.pyplot as plt


def squeezed_vmap(fn):
    vfn = jax.vmap(fn)
    return lambda x: jnp.squeeze(vfn(x))


@squeezed_vmap
def objective_fn(x):
    return x**2 + jnp.cos(2 * (x - 0.25))


def beta(t):
    return 1.0 + t**2 / 10.0


REPEATS = 25
N = 5
dt = 0.025
t0 = 0.0
tend = 25.0
steps_per_v = int(tend / dt) // 50

langevin = ideanneal.langevin.ControlledLangevin(
    objective_fn, beta, ideanneal.transport.NullTransporter()
)
controlled_langevin = ideanneal.langevin.ControlledLangevin(
    objective_fn, beta, ideanneal.transport.ExactTransporter(objective_fn, beta)
)

langevin_trajs = []
controlled_langevin_trajs = []

rng0, rng1, rng2 = jr.split(jr.key(0), 3)

for i in range(REPEATS):
    print("Run", i, "Langevin")
    rng0, key0 = jr.split(rng0)
    x0 = 0.25 + jr.normal(jr.key(0), (N, 1))

    rng1, key1 = jr.split(rng1)
    langevin_traj = langevin.simulate(
        key1, x0, t0, dt, int(tend / dt), steps_per_v, keep_every=1
    )
    langevin_trajs.append(langevin_traj)

    print("Run", i, "Controlled Langevin")
    rng2, key2 = jr.split(rng2)
    controlled_langevin_traj = controlled_langevin.simulate(
        key2, x0, t0, dt, int(tend / dt), steps_per_v, keep_every=1
    )

    controlled_langevin_trajs.append(controlled_langevin_traj)

fig, ax = plt.subplots()
for ri, traj in enumerate(langevin_trajs):
    for i in range(N):
        ax.plot(traj[:, i, 0], c=f"C{ri}")
ax.set_title("Langevin")
plt.show()


fig, ax = plt.subplots()
for ri, traj in enumerate(controlled_langevin_trajs):
    for i in range(N):
        ax.plot(traj[:, i, 0], c=f"C{ri}")
ax.set_title("Controlled Langevin")
plt.show()
