import equinox as eqx
from livn.models.rcsd.diffrax.pinsky_rinzel import PinskyRinzel


class Motoneuron(eqx.Module):
    def __init__(self):
        self.neuron = PinskyRinzel()

    @eqx.filter_jit
    def __call__(
        self,
    ):
        pass

    def run(self, input_current, t0, t1, dt, v0=None, dt_solver=0.01, key=None, **kwargs):
        pass
