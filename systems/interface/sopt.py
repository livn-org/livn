import importlib
import os
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

live_envs = []


class Evaluation:
    def __init__(self):
        self.features = {}
        self.objectives = {}
        self.constraints = {}

    def push(self, name, objective, feature=None):
        if feature is None:
            feature = objective

        self.objectives.setdefault(name, [])
        self.objectives[name].append(objective)
        self.features.setdefault(name, [])
        self.features[name].append(feature)

    def push_constraint(self, name, value, feature=None):
        self.constraints.setdefault(name, [])
        self.constraints[name].append(value)

    def result(self):
        objectives = []
        features = []

        for name in self.objectives:
            objectives.append(np.mean(self.objectives[name]))
            features.append(np.mean(self.features[name]))

        reduced_objectives = np.array(objectives)
        reduced_features = np.asarray(
            [tuple(rf for rf in features)],
            dtype=np.dtype([(name, np.float32) for name in self.features.keys()]),
        )

        if len(self.constraints) > 0:
            constraints = []
            for name in sorted(self.constraints.keys()):
                constraints.append(np.min(self.constraints[name]))

            return {
                0: (
                    reduced_objectives,
                    reduced_features,
                    np.asarray(
                        constraints,
                        dtype=np.float32,
                    ),
                )
            }

        return {0: (reduced_objectives, reduced_features)}


def objective_names(c):
    target = config.import_object_by_path(
        c.config.dopt_params.obj_fun_init_args.target
    )()
    return target.objective_names()


def constraint_names(c):
    target = config.import_object_by_path(
        c.config.dopt_params.obj_fun_init_args.target
    )()
    return target.constraint_names()


def feature_dtypes(c):
    return [(f, np.float32) for f in objective_names(c)]


def get_model(model):
    return find_subclass_in_module(importlib.import_module(model), base_class=Model)()


def obj_fun_init(
    system,
    model,
    target,
    trials,
    subworld_size,
    worker=None,
):
    env = Env(
        system,
        model=get_model(model) if model is not None else None,
        io=io.MEA.from_json(os.path.join(system, "mea.json"), comm=False),
        comm=worker.merged_comm,
        subworld_size=subworld_size,
    )
    env.init()
    env.record_spikes()
    env.record_membrane_current()

    live_envs.append(env)

    target = config.import_object_by_path(target)()

    return partial(obj_fun, env=env, target=target, trials=trials)


def controller_init(system, model, target, subworld_size):
    env = Env(
        system,
        model=get_model(model) if model is not None else None,
        io=io.MEA.from_json(os.path.join(system, "mea.json"), comm=False),
        subworld_size=subworld_size,
    )

    live_envs.append(env)


def obj_fun(x, env, trials, target):
    results = {}
    constraints = {}

    for _ in range(trials):
        env.clear()
        env.set_params(target.transform_params(x))

        objectives_dict, constraints_dict = target(env)

        for name, val in objectives_dict.items():
            results.setdefault(name, [])
            results[name].append(val)

        for name, val in constraints_dict.items():
            constraints.setdefault(name, [])
            constraints[name].append(val)

    return results, constraints


def obj_reduce(payload):
    evaluation = Evaluation()

    objectives_dict, constraints_dict = payload[-1][0]

    for name, trials in objectives_dict.items():
        for objective, feature in trials:
            evaluation.push(name, objective, feature)

    for name, trials in constraints_dict.items():
        for constraint_value, feature in trials:
            evaluation.push_constraint(name, constraint_value, feature)

    return evaluation.result()


class Sopt(Dmosopt):
    class Config(Dmosopt.Config):
        dopt_params: dict = Field(
            default_factory=lambda: {
                "opt_id": "default",
                "obj_fun_init_name": "interface.sopt.obj_fun_init",
                "obj_fun_init_args": {
                    "system": "???",
                    "model": "???",
                    "target": "???",
                    "subworld_size": "${...nprocs_per_worker}",
                },
                # "objective_names": "${oc.dict.keys: .obj_fun_init_args.target_rates}",
                "objective_names": "interface.sopt.objective_names",
                "constraint_names": "interface.sopt.constraint_names",
                "feature_dtypes": "interface.sopt.feature_dtypes",
                "controller_init_fun_name": "interface.sopt.controller_init",
                "controller_init_fun_args": {
                    "subworld_size": "${...nprocs_per_worker}",
                    "system": "${..obj_fun_init_args.system}",
                    "model": "${..obj_fun_init_args.model}",
                    "target": "${..obj_fun_init_args.target}",
                },
                "reduce_fun_name": "interface.sopt.obj_reduce",
                "reduce_fun_args": (),
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
                "surrogate_method_name": None,
                "surrogate_method_kwargs": {},
                "surrogate_custom_training": "dmosopt.model_transformer.joint",
                "surrogate_custom_training_kwargs": {},
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
            env.close()

    def on_after_dispatch(self, success: bool):
        if success:
            # ensure shutdown
            MPI.COMM_WORLD.Abort()
