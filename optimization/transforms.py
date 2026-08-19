import jax.numpy as jnp

__all__ = [
    "BIJECTORS",
    "BOUNDS",
    "DEFAULT",
    "bijector_for",
    "bounds_for",
    "pack",
    "unpack",
]


def _identity_fwd(x):
    return x


def _identity_inv(z):
    return z


def _log_fwd(x):
    return jnp.log(x)


def _log_inv(z):
    return jnp.exp(z)


def _logit_fwd(x):
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jnp.log(x) - jnp.log1p(-x)


def _logit_inv(z):
    return 1.0 / (1.0 + jnp.exp(-z))


def _bounded_fwd(x, lo, hi):
    u = jnp.clip((x - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
    return jnp.log(u) - jnp.log1p(-u)


def _bounded_inv(z, lo, hi):
    return lo + (hi - lo) / (1.0 + jnp.exp(-z))


BOUNDS = {
    "E_L": (-40.0, 40.0),  # mV
    "V_threshold_base": (1.0, 150.0),  # mV
    "tau_m": (0.5, 300.0),  # ms
    "g_L": (0.05, 500.0),  # nS
    "t_ref": (0.05, 100.0),  # ms
    "asc_r": (-2.0, 2.0),
    "theta_jump": (-50.0, 50.0),  # mV
    "delta_v": (-50.0, 50.0),  # mV
}


def bounds_for(name, bounds=None):
    return (BOUNDS if bounds is None else bounds).get(name)


#: name -> (constrained -> unconstrained, unconstrained -> constrained)
BIJECTORS = {
    "identity": (_identity_fwd, _identity_inv),
    "log": (_log_fwd, _log_inv),
    "logit": (_logit_fwd, _logit_inv),
    "bounded": (_bounded_fwd, _bounded_inv),
}


DEFAULT = {
    "tau_m": "bounded",
    "g_L": "bounded",
    "t_ref": "bounded",
    "theta_decay_rate": "log",
    "asc_decay_rate_1": "log",
    "asc_decay_rate_2": "log",
    "tau_s": "log",
    "sigma": "log",
    "alpha": "log",
    "b_v": "log",
    "f_v": "logit",
    "E_L": "bounded",
    "V_threshold_base": "bounded",
    "theta_jump": "identity",
    "delta_v": "identity",
    "asc_amp_1": "identity",
    "asc_amp_2": "identity",
    "a_v": "identity",
    "asc_r": "bounded",
}


def bijector_for(name, spec=None) -> str:
    spec = DEFAULT if spec is None else spec
    return spec.get(name, "identity")


def _apply(direction, name, value, spec, bounds):
    kind = bijector_for(name, spec)
    fn = BIJECTORS[kind][direction]
    if kind != "bounded":
        return fn(jnp.asarray(value))
    box = bounds_for(name, bounds)
    if box is None:
        return jnp.asarray(value)
    return fn(jnp.asarray(value), *box)


def pack(params, spec=None, bounds=None) -> dict:
    return {
        name: _apply(0, name, value, spec, bounds) for name, value in params.items()
    }


def unpack(z, spec=None, bounds=None) -> dict:
    return {name: _apply(1, name, value, spec, bounds) for name, value in z.items()}
