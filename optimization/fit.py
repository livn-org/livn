import jax
import jax.numpy as jnp
import numpy as np

from optimization.losses import param_prior

__all__ = ["fit"]


def fit(
    env,
    target,
    loss,
    init,
    *,
    duration,
    stimulus=None,
    dt: float = 0.1,
    optimizer=None,
    steps: int = 100,
    learning_rate: float = 1e-2,
    prior=None,
    prior_weight: float = 1.0,
    prior_weights=None,
    run_kwargs=None,
    callback=None,
    jit: bool = True,
):
    """Fit cell parameters by gradient descent through the simulation.

    Args:
        env: a differentiable environment (the ``diffrax`` backend), already ``init()``-ed and with
            the channels the loss needs enabled via ``record_*``
        target: whatever ``loss`` compares against (recorded traces, spike times, a dict of both)
        loss: ``loss(run, target) -> scalar``, called with the :class:`~livn.run.Run` of each
            iteration. See :mod:`optimization.losses`.
        init: ``{name: value}`` starting cell parameters. Scalars apply to every cell, while an
            ``(n_cells,)`` array gives each cell its own value.
        duration: simulated duration per iteration, in ms
        stimulus: the stimulus to drive each iteration with, held fixed
        dt: recording grid in ms. This is the grid the returned channels land on, so it has to be
            the one the target was recorded on and the one the loss assumes. It does not
            constrain the integrator, which adapts within each segment regardless.
        optimizer: an optax optimizer; defaults to ``optax.adam(learning_rate)``
        steps: number of update steps
        learning_rate: step size for the default optimizer, ignored when ``optimizer`` is given
        prior: optional ``{name: value}`` to regularize toward, via
            :func:`~optimization.losses.param_prior`
        prior_weight: scalar multiplier on the prior term
        prior_weights: optional per-parameter weights inside the prior term
        run_kwargs: extra keyword arguments for ``env.run``
        callback: optional ``callback(step, theta, value)``, called after each step
        jit: compile the value-and-gradient step (set ``False`` to debug the inner objective)

    Returns:
        ``(theta, history)``. ``history["loss"]`` has ``steps + 1`` entries, one per step plus a
        final evaluation at the returned ``theta``, so the last entry is the loss of what you get
        back. ``history["params"][name]`` tracks each parameter over the same points.
    """
    import optax

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    run_kwargs = {"dt": dt, **(run_kwargs or {})}

    def simulate(theta):
        return env.cells.set_params(theta).run(duration, stimulus, **run_kwargs)

    def objective(theta):
        value = loss(simulate(theta), target)
        if prior is not None:
            value = value + prior_weight * param_prior(theta, prior, prior_weights)
        return value

    value_and_grad = jax.value_and_grad(objective)
    if jit:
        import equinox as eqx

        value_and_grad = eqx.filter_jit(value_and_grad)

    theta = {name: jnp.asarray(value, dtype=float) for name, value in init.items()}
    opt_state = optimizer.init(theta)

    history: dict = {"loss": [], "params": {name: [] for name in theta}}

    def record(theta, value):
        history["loss"].append(float(value))
        for name, array in theta.items():
            history["params"][name].append(np.asarray(array))

    for step in range(int(steps)):
        value, grads = value_and_grad(theta)
        record(theta, value)
        if callback is not None:
            callback(step, theta, float(value))

        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

    record(theta, objective(theta))

    history["loss"] = np.asarray(history["loss"])
    history["params"] = {
        name: np.stack(values) for name, values in history["params"].items()
    }

    return theta, history
