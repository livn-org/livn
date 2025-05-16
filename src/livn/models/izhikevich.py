import brian2 as b2

from livn.types import Model


class Izhikevich(Model):
    def brian2_population_group(self, population_name, n, offset, coordinates, prng):
        population = b2.NeuronGroup(
            n,
            f"""
            dv/dt = (c1*v**2 + c2*v + c3 - u + Rm*(I + I_noise) + stim(t, i + {offset}))/ms : volt
            du/dt = (a*(b*v - u))  : volt
            I : amp
            I_noise : amp
            a : Hz
            b : 1
            c : volt
            d : volt
            noise_amplitude: 1
            """,
            threshold="v>30*mV",
            reset="v=c; u+=d",
            method="euler",
            name=population_name,
            namespace={
                "c1": 0.04 * 1 / b2.mV,
                "c2": 5.0,
                "c3": 140 * b2.mV,
                "Rm": 1 * b2.Gohm,  # resistance
            },
        )
        population.v = -65 * b2.mV
        population.u = "b*v"

        if population_name == "EXC":
            re = prng.uniform(size=n)
            population.a = 0.02 / b2.ms
            population.b = 0.2
            population.c = (-65 + 15 * re**2) * b2.mV
            population.d = (8 - 6 * re**2) * b2.mV
        else:
            ri = prng.uniform(size=n)
            population.a = (0.02 + 0.08 * ri) / b2.ms
            population.b = 0.25 - 0.05 * ri
            population.c = -65 * b2.mV
            population.d = 2 * b2.mV

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
        population_group.noise_amplitude = (
            5 * level if population_group.name == "EXC" else 2 * level
        )
