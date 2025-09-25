import copy
import importlib
import os
import random
from functools import partial

import numpy as np
from dmosopt import config
from machinable import Project
from machinable.utils import find_subclass_in_module
from mpi4py import MPI
from pydantic import Field

from livn import io
from livn.env import Env
from livn.types import Model

_, Dmosopt = (
    Project(os.path.dirname(os.path.dirname(__file__)))
    .provider()
    .on_resolve_element("interface.dmosopt")
)


class Evaluation:
    def __init__(self, objective_names):
        self.objective_names = objective_names
        self.features = {n: [] for n in objective_names}
        self.objectives = {n: [] for n in objective_names}

    def push(self, name, objective, feature=None):
        if feature is None:
            feature = objective

        self.objectives[name].append(objective)
        self.features[name].append(feature)

    def result(self):
        objectives = []
        features = []

        for name in self.objective_names:
            objectives.append(np.mean(self.objectives[name]))
            features.append(np.mean(self.features[name]))

        reduced_objectives = np.array(objectives)
        reduced_features = np.asarray(
            [tuple(rf for rf in features)],
            dtype=np.dtype([(name, np.float32) for name in self.features.keys()]),
        )

        return {0: (reduced_objectives, reduced_features)}


def generate_time_windows(n, window_length, max_gap, seed=42):
    random.seed(seed)

    windows = []
    current_time = 0

    for _ in range(n):
        if windows:
            gap = max(random.random() * max_gap, 10)
            current_time += int(gap)

        start_time = current_time
        end_time = start_time + window_length
        windows.append((start_time, end_time))

        current_time = end_time

    # add gaps
    windows = [
        (ws, we, windows[i + 1][0] - we if i < len(windows) - 1 else 0)
        for i, (ws, we) in enumerate(windows)
    ]

    return windows


def evaluate_isi(it, t, r):
    isi_cvs = []
    for neuron in np.unique(it):
        neuron_spikes = t[it == neuron]
        if len(neuron_spikes) > 1:
            isi = np.diff(np.sort(neuron_spikes))
            if np.nan_to_num(np.mean(isi)) > 0:
                cv = np.std(isi) / np.mean(isi)
                isi_cvs.append(cv)

    if isi_cvs:
        mean_cv = np.mean(isi_cvs)
        isi_error = (mean_cv - 1) ** 2
        r.push("isi", isi_error, mean_cv)
    else:
        r.push("isi", 10.0, 0.0)


def evaluate_correlation(it, t, r, trial_length):
    bin_size = 50
    bins = np.arange(0, trial_length + bin_size, bin_size)

    unique_neurons = np.unique(it)
    if len(unique_neurons) < 2:
        r.push("corr", 0.0, -1)
        return

    binned_spikes = np.zeros((len(unique_neurons), len(bins) - 1))
    for i, neuron in enumerate(unique_neurons):
        neuron_spikes = t[it == neuron]
        hist, _ = np.histogram(neuron_spikes, bins=bins)
        binned_spikes[i, :] = hist

    if np.all(binned_spikes.std(axis=1) == 0):
        r.push("corr", 0.0, -1)
        return

    corr_matrix = np.corrcoef(binned_spikes)
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)

    corr_values = corr_matrix[mask]
    if len(corr_values) == 0 or np.all(np.isnan(corr_values)):
        mean_corr = 0.0
    else:
        mean_corr = np.nanmean(corr_values)

    target_min = 0.05
    target_max = 0.4
    target_mid = (target_min + target_max) / 2

    if mean_corr < target_min:
        corr_score = (mean_corr / target_min) ** 2
    elif mean_corr > target_max:
        corr_score = np.exp(-2 * (mean_corr - target_max) / (1 - target_max))
    else:
        corr_score = 1.0 - 0.5 * abs(mean_corr - target_mid) / (target_max - target_min)

    if np.isnan(corr_score):
        r.push("corr", 0.0, -1)
        return

    r.push("corr", -corr_score, corr_score)


def evaluate_stimulation(it, t, r, channel, trial_length):
    active = len(it[t <= 150])
    silent = len(it[t > 150])
    trivial = active == 0 and silent == 0

    # maximize spikes in active window
    r.push(f"a{channel}", -float(active) if not trivial else 999.0)
    # minimize in silent range
    r.push(f"s{channel}", float(silent) if not trivial else 999.0)


def evaluate_dynamics(it, t, r, targets, trial_length):
    # rate
    rate = np.nan_to_num(
        np.mean(np.unique(it, return_counts=True)[1] / (trial_length / 100))
    )
    r.push("rate", (rate - targets["rate"]) ** 2, rate)

    # active fraction of neurons
    active_neurons = np.unique(it).size
    total_neurons = sum(targets["cell_count"].values())
    active = active_neurons / total_neurons
    r.push("active", (targets["active"] - active) ** 2, active)

    # ISI
    evaluate_isi(it, t, r)


def _concat(a):
    if len(a) == 1:
        return a[0]

    if len(a) > 1:
        return np.concatenate(a)

    return []


def feature_dtypes(c):
    return [(f, np.float32) for f in c.config.dopt_params.objective_names]


def objective_names(c):
    return [f for f in c.config.dopt_params.reduce_fun_args[0]]


def get_model(model):
    return find_subclass_in_module(importlib.import_module(model), base_class=Model)()


live_envs = []


def make_env(system, model, subworld_size, comm=None):
    env = Env(
        system,
        model=get_model(model) if model is not None else None,
        io=io.MEA.from_json(os.path.join(system, "mea.json"), comm=False),
        comm=comm,
        subworld_size=subworld_size,
    )
    env.init()
    env.record_spikes()

    return env


def obj_fun_init(system, model, subworld_size, stimulate=False, trials=1, worker=None):
    env = make_env(
        system,
        model,
        subworld_size,
        worker.merged_comm,
    )

    live_envs.append(env)
    return partial(obj_fun, env=env, stimulate=stimulate, trials=trials)


def controller_init(system, model, subworld_size):
    env = Env(
        system,
        model=get_model(model) if model is not None else None,
        io=io.MEA.from_json(os.path.join(system, "mea.json"), comm=False),
        subworld_size=subworld_size,
    )
    live_envs.append(env)


def obj_fun(x, env, stimulate, trials):
    env.clear()
    x_c = copy.deepcopy(x)
    env.set_noise(exc=x_c.pop("noise_exc", 0.0), inh=x_c.pop("noise_inh", 0.0))
    env.set_weights(x_c)

    warmup = 250

    if stimulate:
        trial_length = 500
        t_end = warmup + trial_length
        num_channels = env.io.num_channels

        it = []
        t = []
        for c in range(num_channels):
            inputs = np.zeros([t_end, num_channels])
            for r in range(20):
                inputs[warmup + r, c] = 750

            stimulus = env.cell_stimulus(inputs)

            _it, _t, *_ = env.run(t_end, stimulus=stimulus, root_only=False)

            for ii in _it[_t >= warmup]:
                it.append(ii)
            for tt in _t[_t >= warmup]:
                t.append(tt - warmup)

            env.clear()

        return {
            "it": np.array(it),
            "t": np.array(t),
            "t_end": trial_length * num_channels,
            "stimulate": stimulate,
            "trials": num_channels,
        }

    trial_length = 500
    t_end = warmup + trial_length * trials
    it, t, *_ = env.run(t_end, stimulus=None, root_only=False)

    it, t = it[t >= warmup], t[t >= warmup] - warmup
    t_end = t_end - warmup

    return {
        "it": it,
        "t": t,
        "t_end": t_end,
        "stimulate": stimulate,
        "trials": trials,
    }


def obj_reduce(payload, targets, callback=None):
    it = []
    t = []
    t_end = 0
    trials = None
    stimulate = None
    for result in payload:
        it.append(result[0]["it"])
        t.append(result[0]["t"])
        t_end = max(t_end, result[0]["t_end"])
        trials = result[0]["trials"]
        stimulate = result[0]["stimulate"]

    it, t = _concat(it), _concat(t)

    if callback is not None:
        callback(it=it, t=t, t_end=t_end, trials=trials, targets=targets)

    r = Evaluation(targets["objective_names"])

    if stimulate:
        evaluate_correlation(it, t, r, t_end)
        evaluate_isi(it, t, r)

    trial_length = t_end // trials
    for i in range(trials):
        trial_start = i * trial_length
        trial_end = (i + 1) * trial_length

        mask = (t >= trial_start) & (t < trial_end)

        trial_it = it[mask]
        trial_t = t[mask] - trial_start

        if stimulate:
            evaluate_stimulation(it, t, r, i, trial_length)
        else:
            evaluate_dynamics(trial_it, trial_t, r, targets, trial_length)

    return r.result()


class Sopt(Dmosopt):
    class Config(Dmosopt.Config):
        dopt_params: dict = Field(
            default_factory=lambda: {
                "opt_id": "default",
                "obj_fun_init_name": "interface.sopt.obj_fun_init",
                "obj_fun_init_args": {
                    "system": "???",
                    "model": "???",
                    "subworld_size": "${...nprocs_per_worker}",
                },
                # "objective_names": "${oc.dict.keys: .obj_fun_init_args.target_rates}",
                "objective_names": "interface.sopt.objective_names",
                "feature_dtypes": "interface.sopt.feature_dtypes",
                "controller_init_fun_name": "interface.sopt.controller_init",
                "controller_init_fun_args": {
                    "subworld_size": "${...nprocs_per_worker}",
                    "system": "${..obj_fun_init_args.system}",
                    "model": "${..obj_fun_init_args.model}",
                },
                "reduce_fun_name": "interface.sopt.obj_reduce",
                "reduce_fun_args": "???",
                "problem_parameters": {},
                "optimizer_name": "nsga2",
                "initial_method": "slh",
                "n_initial": 100,
                "initial_maxiter": 0,
                "n_epochs": 25,
                "population_size": 50,
                "num_generations": 100,
                "termination_conditions": True,
                "resample_fraction": 1.0,
                "surrogate_method_name": "gpr",
                "surrogate_method_kwargs": {},
                "feasibility_method_name": None,
                "feasibility_method_kwargs": {},
                "save": True,
            }
        )
        ranks: int = -1

    def evaluate_objective_at(self, x, verbose=False, **reduce_kwargs):
        import logging

        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        class W:
            def __init__(self):
                self.merged_comm = comm

        worker = W()

        p = x
        if not isinstance(p, dict):
            p = self.parameter_vector_to_dict(x)

        logging.basicConfig(level=logging.INFO if verbose else logging.ERROR)
        if "obj_fun_init_name" in self.config.dopt_params:
            kwargs = dict(self.config.dopt_params.obj_fun_init_args)
            kwargs["worker"] = worker
            kwargs["subworld_size"] = size
            obj_fun = config.import_object_by_path(
                self.config.dopt_params.obj_fun_init_name
            )(**kwargs)
        else:
            obj_fun = config.import_object_by_path(self.config.dopt_params.obj_fun_name)

        payload = obj_fun(p)

        gathered_payload = comm.gather(payload, root=0)

        if rank != 0:
            return None

        reduce_fun = config.import_object_by_path(
            self.config.dopt_params.reduce_fun_name
        )
        args = self.config.dopt_params.reduce_fun_args

        return reduce_fun([{0: p} for p in gathered_payload], *args, **reduce_kwargs)

    def on_finish(self, success: bool):
        for env in live_envs:
            if hasattr(env, "pc"):
                env.pc.done()

    def on_after_dispatch(self, success: bool):
        if success:
            # ensure shutdown
            MPI.COMM_WORLD.Abort()
