#!/usr/bin/env python3

import os
import sys
import time

import numpy as np
from machinable.utils import save_file
from neuron import h

from livn.backend import backend
from livn.env import Env
from livn.io import MEA
from livn.models.rcsd import ReducedCalciumSomaDendrite
from livn.utils import P

if len(sys.argv) > 1 and sys.argv[1] == "slurm":
    # generate slurm scripts
    max_nodes = 32
    max_ranks_per_node = 144
    power2_per_node = int(np.log2(max_ranks_per_node))
    ranks_per_node = 2**power2_per_node

    for power2 in range(5, 13):
        cores = 2**power2
        if cores > ranks_per_node * max_nodes:
            continue

        nodes = cores / ranks_per_node

        job_name = f"s_{power2}_{cores}"
        job_N = int(max(1, nodes))
        job_ntasks_per_node = int(cores if nodes < 1 else ranks_per_node)

        print(
            f"sbatch -J {job_name} -N {job_N} --ntasks-per-node {job_ntasks_per_node} scaling.sh"
        )

    exit()

assert backend() == "neuron"

t_end = 5_000
system = "systems/graphs/" + sys.argv[1]

system_name = os.path.basename(system)

model = ReducedCalciumSomaDendrite()
io = MEA.from_json(os.path.join(system, "mea.json"))
env = Env(system, model, io).init()

env.apply_model_defaults()

for trials in range(3):
    q = time.time()

    inputs = np.zeros([t_end, 16])

    for r in range(20):
        for c in [1, 2, 3, 4]:
            inputs[t_end // 4 + r, c] = 750

    stimulus = env.cell_stimulus(inputs)

    stimctime = time.time() - q

    env.run(t_end, stimulus=stimulus)

    cputime = env.pc.step_time()

    stats = {
        "stimctime": [stimctime],
        "mkcellstime": [env.mkcellstime],
        "connectgjstime": [env.connectgjstime],
        "connectcellstime": [env.connectcellstime],
        "psolvetime": [env.psolvetime],
        "cputime": [cputime],
        "spike_communication": [env.pc.send_time()],
        "event_handling": [env.pc.event_time()],
        "numerical_integration": [env.pc.integ_time()],
    }

    stats = P.gather(stats)

    cwtime = cputime + env.pc.step_wait()
    maxcw = env.pc.allreduce(cwtime, 2)
    meancomp = env.pc.allreduce(cputime, 1) / P.size()
    gjtime = env.pc.vtransfer_time()
    gjvect = h.Vector()
    env.pc.allgather(gjtime, gjvect)
    meangj = gjvect.mean()
    maxgj = gjvect.max()

    if P.is_root():
        stats = P.merge(stats)

        stats.update(
            {
                "voltage_transfer": [gjtime],
                "load_balance": [meancomp / (maxcw if maxcw > 0 else 1)],
                "mean_voltage_transfer_time": meangj,
                "max_voltage_transfer_time": maxgj,
                "meancomp": cputime,
            }
        )

        save_file(
            [
                os.path.dirname(os.path.abspath(__file__)),
                "results",
                "scaling",
                f"{system_name}_{P.size()}.jsonl",
            ],
            {k: v.tolist() if hasattr(v, "tolist") else v for k, v in stats.items()},
            mode="a+",
        )
