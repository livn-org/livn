import equinox as eqx
import jax

from livn.models.rcsd.diffrax.motoneuron import BoothRinzelKiehn


class MotoneuronCulture(eqx.Module):
    neurons: BoothRinzelKiehn
    num_neurons: int = eqx.field(static=True)
    section_names: tuple[str, ...] = eqx.field(static=True)
    opsin_gmax: jax.Array | None

    def __init__(self, num_neurons: int, params: dict | None = None, key=None):
        # if key is None:
        #     key = jax.random.PRNGKey(0)

        # neuron_keys = jax.random.split(key, num=num_neurons)

        self.num_neurons = num_neurons
        self.section_names = ("soma", "dend")
        # currently, we only support independent parallel simulation
        #  of individual neurons via vmap
        self.neurons = BoothRinzelKiehn(params)
        self.opsin_gmax = (
            jax.numpy.full((num_neurons,), self.neurons.chr2_gmax)
            if self.neurons.opsin == "ChR2H134R"
            else None
        )

    @property
    def input_mode(self):
        return self.neurons.input_mode

    def _per_neuron(self, input_current, duration, dt):
        """`[neurons, time, 2]` soma and dendrite columns from `[time, 2 * neurons]`."""
        if input_current is None:
            input_current = jax.numpy.zeros(
                [int(duration / dt) + 1, self.num_neurons * 2]
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

        per_neuron_stimulus = input_current.reshape(
            input_current.shape[0], self.num_neurons, 2
        )
        return jax.numpy.swapaxes(per_neuron_stimulus, 0, 1)

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
        record=None,
        **kwargs,
    ):
        del record  # no auxiliary state to record selectively

        def solve_single(I_stim, init_state, gmax=None):
            neurons = self.neurons
            if gmax is not None:
                neurons = eqx.tree_at(lambda m: m.chr2_gmax, neurons, gmax)
            t_arr, v_soma, v_dend, i_mem_soma, i_mem_dend, final_state = neurons.solve(
                t_dur=t1 - t0,
                I_stim_array=I_stim,
                dt=dt,
                dt_solver=dt_solver,
                y0=init_state,
                **kwargs,
            )
            v_both = jax.numpy.stack((v_soma, v_dend), axis=0)
            i_mem_both = jax.numpy.stack((i_mem_soma, i_mem_dend), axis=0)
            return t_arr, v_both, i_mem_both, final_state

        gmax_axis = None if self.opsin_gmax is None else 0
        if y0 is None:
            solve_many = jax.vmap(solve_single, in_axes=(0, None, gmax_axis))
            y0_arg = None
        else:
            solve_many = jax.vmap(solve_single, in_axes=(0, 0, gmax_axis))
            y0_arg = y0

        if isinstance(self.input_mode, str):
            per_neuron_stimulus = self._per_neuron(input_current, t1 - t0, dt)
        else:
            modes = self.neurons.input_modes
            given = {} if input_current is None else input_current
            if not isinstance(given, dict) or set(given) - set(modes):
                raise ValueError(
                    f"a module of input modes {list(modes)} reads a dict of one "
                    "[time, 2 * neurons] array per mode, not "
                    f"{sorted(given) if isinstance(given, dict) else 'one array'}"
                )
            per_neuron_stimulus = {
                mode: self._per_neuron(given.get(mode), t1 - t0, dt) for mode in modes
            }

        _t_arr_batched, v_soma_dend, i_mem_soma_dend, yT = solve_many(
            per_neuron_stimulus, y0_arg, self.opsin_gmax
        )

        # no spike detection
        it = jax.numpy.empty((0,), dtype=jax.numpy.int32)
        tt = jax.numpy.empty((0,), dtype=jax.numpy.float32)

        # interleave treating compartments as neurons (soma0,dend0,soma1,dend1,...)
        iv = jax.numpy.repeat(jax.numpy.arange(self.num_neurons), 2)
        v = v_soma_dend.reshape(self.num_neurons * 2, -1)
        im = iv
        mp = i_mem_soma_dend.reshape(self.num_neurons * 2, -1)

        return it, tt, iv, v, im, mp, yT, {}
