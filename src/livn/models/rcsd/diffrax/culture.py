import equinox as eqx
from livn.models.rcsd.diffrax.motoneuron import BoothRinzelKiehn
import jax


class MotoneuronCulture(eqx.Module):
    neurons: BoothRinzelKiehn
    num_neurons: int = eqx.field(static=True)

    def __init__(self, num_neurons: int, params: dict | None = None, key=None):
        # if key is None:
        #     key = jax.random.PRNGKey(0)

        # neuron_keys = jax.random.split(key, num=num_neurons)

        self.num_neurons = num_neurons
        # currently, we only support independent parallel simulation
        #  of individual neurons via vmap
        self.neurons = BoothRinzelKiehn(params)

    def run(
        self, input_current, t0, t1, dt, v0=None, dt_solver=0.01, key=None, **kwargs
    ):
        @jax.vmap
        def solve_many(I_stim):
            t_arr, v_soma, v_dend, i_mem_soma, i_mem_dend = self.neurons.solve(
                t_dur=t1 - t0, I_stim_array=I_stim, dt=dt
            )

            # stack so that after vmap we have shape (num_neurons, 2, T)
            v_both = jax.numpy.stack((v_soma, v_dend), axis=0)
            i_mem_both = jax.numpy.stack((i_mem_soma, i_mem_dend), axis=0)
            return t_arr, v_both, i_mem_both

        # map over neuron dimension -> [num_neurons, 2, T + 1]
        t_arr, v_soma_dend, i_mem_soma_dend = solve_many(input_current.T)

        # No spike detection
        it = None
        tt = None
        # interleave treating compartments as neurons (soma0,dend0,soma1,dend1,...)
        iv = jax.numpy.repeat(jax.numpy.arange(self.num_neurons), 2)
        v = v_soma_dend.reshape(self.num_neurons * 2, -1)
        im = iv
        m = i_mem_soma_dend.reshape(self.num_neurons * 2, -1)

        return it, tt, iv, v, im, m
