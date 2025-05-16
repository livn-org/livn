import brian2 as b2

from livn.types import Model


class LIF(Model):
    def brian2_population_group(self, population_name, n, offset, coordinates, prng):
        v_resting = -70 * b2.mV
        population = b2.NeuronGroup(
            n,
            f"""
            dv/dt = ((v_resting - v) + Rm*(I + I_noise) + stim(t, i + {offset}))/tau : volt
            I : amp
            I_noise : amp
            noise_amplitude: 1
            """,
            threshold="v>-55*mV",
            reset="v=v_reset",
            method="euler",
            name=population_name,
            namespace={
                "tau": 10 * b2.ms,
                "Rm": 100 * b2.Mohm,
                "v_reset": -75 * b2.mV,
                "v_resting": v_resting,
            },
        )
        population.v = v_resting

        return population

    def brian2_connection_synapse(self, pre_group, post_group):
        synapse = b2.Synapses(
            pre_group,
            post_group,
            """
            w : 1
            multiplier: 1
            distance: 1
            prefix: 1
            """,
            on_pre="I += prefix * w * multiplier * pA",
        )

        return synapse

    def brian2_noise_op(self, population_group, prng):
        return population_group.run_regularly(
            "I_noise = noise_amplitude*randn()*pA", dt=1 * b2.ms
        )

    def brian2_noise_configure(self, population_group, level):
        population_group.noise_amplitude = level
