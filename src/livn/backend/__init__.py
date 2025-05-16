from livn.backend.common import *
from livn.backend.config import backend

if backend() == "brian2":
    from livn.backend.brian2 import *
elif backend() == "neuron":
    from livn.backend.neuron import *
elif backend() == "diffrax":
    from livn.backend.diffrax import *
else:
    print(f"livn: Unknown backend: {backend()}")
