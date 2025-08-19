# livn

A testbed for learning to interact with in vitro neural networks

## Quickstart

### Using the dataset

[examples/using_the_dataset.py](examples/using_the_dataset.py)
```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
#   "datasets",
# ]
# ///
from livn.system import make
from livn.io import MEA
from datasets import load_dataset

system_name = "S1"

dataset = load_dataset("livn-org/livn", name=system_name)

sample = dataset["train_with_noise"][0]
it = sample["trial_it"][0]
t = sample["trial_t"][0]

# use a multi-electrode array to 'observe' the data
system = make(system_name)
mea = MEA.from_directory(system.uri)

cit, ct = mea.channel_recording(system.neuron_coordinates, it, t)

print("Spikes in channel 0:")
print(ct[0])
```

### Using the livn environment interactively

[examples/run_a_simulation.py](examples/run_a_simulation.py)
```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
# ]
# ///
from livn.env import Env
from livn.system import predefined

env = Env(predefined("S1")).init()

env.apply_model_defaults()
env.record_spikes()
env.record_voltage()

it, t, iv, v = env.run(100)

print("Initial voltages: ", v[:, 0])
```

Note that livn supports three backends that can be chosen before running the script:
```sh
export LIVN_BACKEND=brian2     # default
export LIVN_BACKEND=diffrax    # requires livn[diffrax] dependencies
export LIVN_BACKEND=neuron     # requires livn[neuron] dependencies and MPI (see below)
```


### Parallel execution

NEURON simulations can executed in parallel using `mpirun -n $MPI_RANKS ...`. Checkout a full example [here](examples/parallel_simulation.py).


### Differentiable simulations

When using the diffrax backend, it is possible to differentiate through the simulation environment, including the electrode IO:

```python
import os

os.environ["LIVN_BACKEND"] = "diffrax"

import equinox as eqx
import jax.numpy as jnp
import optax

from livn.env import Env
from livn.system import predefined


@eqx.filter_jit
def systempass(inputs, env, t_end, targets, key):
    # pass through IO and system
    stimulus = env.cell_stimulus(inputs)
    mask, _, gids, v = env.run(t_end, stimulus, unroll="mask")

    return -jnp.mean(v)  # dummy loss: maximize action potentials


@eqx.filter_jit
def make_step(
    env,
    inputs,
    t_end,
    grad_loss,
    optim,
    targets,
    opt_state,
    key,
):
    loss, grads = grad_loss(inputs, env, t_end, targets, key)
    updates, opt_state = optim.update(grads, opt_state)
    new_inputs = eqx.apply_updates(inputs, updates)
    return loss, opt_state, new_inputs


env = Env(predefined("S1")).init()

env.apply_model_defaults()
env.record_spikes()
env.record_voltage()

t_end = 30
inputs = jnp.zeros([t_end, 16]) # zero input
optim = optax.adam(1)
opt_state = optim.init(eqx.filter(inputs, eqx.is_inexact_array))
grad_loss = eqx.filter_value_and_grad(systempass)
targets = []

loss, opt_state, inputs = make_step(
    env,
    inputs,
    t_end,
    grad_loss,
    optim,
    targets,
    opt_state,
    env.run_key
)

print('Gradient for the electrode input: ', inputs)
```

Checkout the full example [here](examples/differentiable_simulation.py).

## Machine learning benchmarks

livn integrates with the [Gymnasium](https://gymnasium.farama.org/) standard interface for RL environments:

```python
import gymnasium as gym

from livn.env import Env
from livn.system import predefined
from livn.integrations.gym import LivnGym

# standard RL environment
benchmark = gym.make("Pendulum-v1")

# livn system
env = Env(predefined("S1")).init()

rl_env = LivnGym(
    benchmark,
    env,
    encoding=encoder, # define callable to encode observations into the system
    decoding=decoder, # define callable to decode actions from the system
)

... # use rl_env like any other gymnasium environment
```

Checkout the full [reinforcement learning examples](benchmarks/rl.py): 

```sh
# Train SAC on the simulated livn system
python benchmarks/rl.py
# Pretrain using the livn dataset as pre-generated replay buffer
python benchmarks/rl.py --replay
```

## Advanced usage for research and development

Clone this repo and get [uv](https://docs.astral.sh/uv/) to `uv sync`.

Note that this repository has different `--package` dependencies.

### Prerequisites

If you use the NEURON backend for large-scale simulation, an MPI and HDF5 installation is required. Note that most of the NEURON backend code is hosted and documented [here](https://github.com/GazzolaLab/MiV-Simulator).

Linux (Debian) 🐧 | Windows (WSL2) 🪟
```sh
$ apt install -y cmake mpich libmpich-dev libhdf5-mpich-dev hdf5-tools
```

Mac 🍎
```sh
$ brew install hdf5-mpi
```

Additionally, if you are interested in generating your own systems, you will  have to compile `neuroh5`. Note that this is **not required** if you download livn's default systems.
```sh
git clone https://github.com/iraikov/neuroh5.git
cd neuroh5
cmake .
make

# add the neuroh5 binaries to your PATH
export PATH="/path/to/neuroh5/bin:$PATH"
```

- [The paper describing the H5 file format](https://www.biorxiv.org/content/10.1101/2021.11.02.466940v1.full)
- [h5py](https://docs.h5py.org/en/stable/) and [neuroh5](https://github.com/iraikov/neuroh5) to write and read the H5 coordinate files used by the simulator
- vscode extension for opening H5 files: `h5web.vscode-h5web`

### Systems

Requires `uv sync --package systems` (see [systems](./systems) directory).

#### Generate a system

System generation reads a YAML configuration to create the HDF5 files containing the neuron locations and connectivity of the network. The following launches a parallel job using 8 cores to generate the S1 system:

```bash
$ export MPI_RANKS=8
$ livn systems mpi generate config=./systems/config/S1.yml --launch
```

> Tip: To use MPI within SLURM, replace `mpi` with `slurm`. [Learn more](https://machinable.org/examples/slurm-execution/)

Once completed, you can manage the files using:

```bash
... --inspect  # list generated files
... --mea      # generate a multi-electrode configuration (mea.json)
... --export   # export generated files to the systems/data directory
```

#### Tuning

To find appropriate parameters, you can leverage black-box optimization:
```sh
export DISTWQ_CONTROLLER_RANK=-1
livn systems mpi tune system=./systems/data/S1 nprocs_per_worker=1 weights=1 stimulate=1 --launch
```

Use `... --inspect` to display the pareto-front of solutions. This will output a command to continue tuning of synaptic noise using the found synaptic weights.

#### Sampling

To sample from the system (i.e. run many parallel simulations for different input features), you can use the sampling operation:
```sh
livn systems mpi sample output_directory=./my-dataset nprocs_per_worker=1 samples=10000 noise=0 --launch
```
Once completed, the samples can be merged using `... --merge`.

### Benchmarks

Requires `uv sync --package benchmarks` (see [benchmarks](./benchmarks) directory).


### Full installation

To install all dependencies and package, you may use:
```sh
uv sync --all-packages --group diffrax
```
