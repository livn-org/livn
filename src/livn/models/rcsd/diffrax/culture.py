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

    @property
    def input_mode(self):
        return self.neurons.input_mode

    def run(
        self,
        input_current,
        noise,
        t0,
        t1,
        dt,
        y0=None,
        dt_solver=0.01,
        key=None,
        **kwargs,
    ):
        def solve_single(I_stim, init_state):
            t_arr, v_soma, v_dend, i_mem_soma, i_mem_dend, final_state = (
                self.neurons.solve(
                    t_dur=t1 - t0,
                    I_stim_array=I_stim,
                    dt=dt,
                    dt_solver=dt_solver,
                    y0=init_state,
                    **kwargs,
                )
            )
            v_both = jax.numpy.stack((v_soma, v_dend), axis=0)
            i_mem_both = jax.numpy.stack((i_mem_soma, i_mem_dend), axis=0)
            return t_arr, v_both, i_mem_both, final_state

        if y0 is None:
            solve_many = jax.vmap(solve_single, in_axes=(0, None))
            y0_arg = None
        else:
            solve_many = jax.vmap(solve_single, in_axes=(0, 0))
            y0_arg = y0

        if input_current is None:
            input_current = jax.numpy.zeros(
                [int((t1 - t0) / dt) + 1, self.num_neurons * 2]
            )

        if input_current.ndim == 1:
            input_current = input_current[:, None]

        if input_current.ndim != 2:
            raise ValueError("Expected stimulus array with shape [time, neurons]")

        expected_channels = self.num_neurons * 2
        if input_current.shape[1] != expected_channels:
            raise ValueError(
                "Stimulus channels must provide soma and dendrite currents"
            )

        # group soma and dendrite columns per neuron before vmapping
        per_neuron_stimulus = input_current.reshape(
            input_current.shape[0], self.num_neurons, 2
        )
        per_neuron_stimulus = jax.numpy.swapaxes(per_neuron_stimulus, 0, 1)

        t_arr_batched, v_soma_dend, i_mem_soma_dend, yT = solve_many(
            per_neuron_stimulus, y0_arg
        )

        # no spike detection
        it = jax.numpy.empty((0,), dtype=jax.numpy.int32)
        tt = jax.numpy.empty((0,), dtype=jax.numpy.float32)

        # interleave treating compartments as neurons (soma0,dend0,soma1,dend1,...)
        iv = jax.numpy.repeat(jax.numpy.arange(self.num_neurons), 2)
        v = v_soma_dend.reshape(self.num_neurons * 2, -1)
        im = iv
        mp = i_mem_soma_dend.reshape(self.num_neurons * 2, -1)

        return it, tt, iv, v, im, mp, yT
