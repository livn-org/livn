import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from livn.models.slif.snn import SpikingNeuralNet
from livn.types import Model


class SLIF(Model):
    def diffrax_module(self, connectivity, key):
        tau_s = 1
        beta = 5
        v_th = 1
        v_reset = 1.2
        alpha = 3e-2
        mu = jnp.array([10, 0.0])

        diffusion = False
        sigma = None

        # Cap exp function for numerical stability
        def intensity_fn(v: Float) -> Float:
            return jnp.exp(beta * (jnp.minimum(v - v_th, 10))) / tau_s

        return SpikingNeuralNet(
            num_neurons=len(connectivity),
            intensity_fn=intensity_fn,
            v_reset=v_reset,
            alpha=alpha,
            network=jnp.array(connectivity == 0.0),  # mask for weight update
            read_out_neurons=None,  # all neurons participate
            mu=mu,
            diffusion=diffusion,
            sigma=sigma,
            key=key,
        )

    def diffrax_default_weights(self, system: str):
        return {
            "S1": {
                "EXC_EXC": 1.0978738069534302,
                "EXC_INH": 3.549142360687256,
                "INH_EXC": 21.964923858642578,
                "INH_INH": 0.0,
            },
            "S2": {},
            "S3": {},
            "S4": {},
        }[system]
