import hashlib
import os
from collections.abc import Mapping
from functools import partial

import numpy as np
from dmosopt import config
from machinable import Project
from machinable.config import Field as ConfigField
from machinable.config import to_dict
from mpi4py import MPI
from pydantic import Field

from livn.env import Env
from livn.utils import import_instance

# NEURON compatible rank order
os.environ.setdefault("DISTWQ_CONTROLLER_RANK", "-1")

_, Dmosopt = (
    Project(os.path.dirname(os.path.dirname(__file__)))
    .provider()
    .on_resolve_interface("interface.dmosopt")
)

live_envs = []


class Evaluation:
    def __init__(self, objectives=None, constraints=None, features=None):
        self.declared = {
            "objectives": None if objectives is None else list(objectives),
            "constraints": None if constraints is None else list(constraints),
            "features": None if features is None else list(features),
        }
        """What dmosopt will name each column, from `declared_names`."""
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

    def push_feature(self, name, feature):
        self.features.setdefault(name, [])
        self.features[name].append(feature)

    def _order(self, kind: str, collected: dict) -> list:
        declared = self.declared.get(kind)
        if declared is None:
            return list(collected)

        missing = [name for name in declared if name not in collected]
        unexpected = [name for name in collected if name not in declared]
        if missing or unexpected:
            raise ValueError(
                f"the target declares {kind} {declared} but this evaluation "
                f"produced {list(collected)}; dmosopt names the columns from "
                f"the declaration, so the two have to agree (missing "
                f"{missing}, unexpected {unexpected})"
            )
        return declared

    def result(self):
        objective_names = self._order("objectives", self.objectives)
        feature_names = self._order("features", self.features)

        objectives = [np.mean(self.objectives[name]) for name in objective_names]
        features = [np.mean(self.features[name]) for name in feature_names]

        reduced_objectives = np.array(objectives)
        reduced_features = np.asarray(
            [tuple(features)],
            dtype=np.dtype([(name, np.float32) for name in feature_names]),
        )

        if len(self.constraints) > 0:
            constraint_names = self._order("constraints", self.constraints)
            constraints = [np.min(self.constraints[name]) for name in constraint_names]

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


def declared_names(target) -> dict:
    objectives = list(target.objective_names())
    return {
        "objectives": objectives,
        "constraints": list(target.constraint_names()),
        "features": objectives
        + [n for n in target.observed_feature_names() if n not in objectives],
    }


def _declared(c) -> dict:
    return declared_names(
        import_instance(c.config.dopt_params.obj_fun_init_args.target)
    )


def objective_names(c):
    return _declared(c)["objectives"]


def constraint_names(c):
    return _declared(c)["constraints"]


def feature_dtypes(c):
    return [(f, np.float32) for f in _declared(c)["features"]]


def _build_env(target, system, model, comm, subworld_size, selection=None):
    model = import_instance(model)
    if hasattr(target, "build_env"):
        if selection is not None:
            raise ValueError(
                f"{type(target).__name__} builds its own env, so it owns which "
                "cells exist; set the selection through the target instead of "
                "overriding it here"
            )
        return target.build_env(system, model, comm=comm, subworld_size=subworld_size)
    env = Env(
        system,
        model=model,
        io=target.io() if hasattr(target, "io") else None,
        comm=comm,
        subworld_size=subworld_size,
    )
    if selection is not None:
        env.selection(selection)
    env.init()
    return target.init(env)


def obj_fun_init(
    system,
    model,
    target,
    trials,
    subworld_size,
    selection=None,
    worker=None,
):
    target = import_instance(target)
    env = _build_env(
        target, system, model, worker.merged_comm, subworld_size, selection=selection
    )
    live_envs.append(env)
    return partial(obj_fun, env=env, target=target, trials=trials)


def controller_init(system, model, target, subworld_size):
    target = import_instance(target)
    env = Env(
        system,
        model=import_instance(model),
        io=target.io() if hasattr(target, "io") else None,
        comm=MPI.COMM_SELF,
        subworld_size=subworld_size,
    )
    live_envs.append(env)


def obj_fun(x, env, trials, target):
    results = {}
    constraints = {}
    observed = {}

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

        for name, val in target.observed_features().items():
            observed.setdefault(name, [])
            observed[name].append(val)

    return results, constraints, observed, declared_names(target)


def obj_reduce(payload):
    result = payload[-1][0]
    objectives_dict, constraints_dict = result[0], result[1]
    observed_dict = result[2] if len(result) > 2 else {}
    evaluation = Evaluation(**(result[3] if len(result) > 3 else {}))

    for name, trials in objectives_dict.items():
        for objective, feature in trials:
            evaluation.push(name, objective, feature)

    for name, trials in constraints_dict.items():
        for constraint_value, feature in trials:
            evaluation.push_constraint(name, constraint_value, feature)

    for name, values in observed_dict.items():
        for value in values:
            evaluation.push_feature(name, value)

    return evaluation.result()


class Sopt(Dmosopt):
    class Config(Dmosopt.Config):
        system: str | int | dict[str, int] | None = ConfigField(None, identifying=False)
        dopt_params: dict = Field(
            default_factory=lambda: {
                "opt_id": "default",
                "obj_fun_init_name": "interface.sopt.obj_fun_init",
                "obj_fun_init_args": {
                    # system: injected at dispatch
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

    def on_compute_predicate(self):
        return {**self._system_predicate(), **self._problem_predicate()}

    def _system_predicate(self):
        system = self.config.system
        if system is None:
            return {}
        if isinstance(system, int):
            return {"system": f"{system} cell" + ("s" if system != 1 else "")}
        if isinstance(system, Mapping):
            counts = to_dict(system)
            return {
                "system": ", ".join(
                    f"{n} {population} cell" + ("s" if n != 1 else "")
                    for population, n in counts.items()
                )
            }
        return super().on_compute_predicate()

    def _problem_predicate(self):
        try:
            spec = self.config.dopt_params.obj_fun_init_args.target
            # "???" until dispatch injects the real one
            target = import_instance(spec) if spec and spec != "???" else None
        except Exception:
            target = None
        if target is None:
            return {}

        def names(attr):
            fn = getattr(target, attr, None)
            try:
                return list(fn()) if callable(fn) else []
            except Exception:
                return []

        objectives, constraints = names("objective_names"), names("constraint_names")
        try:
            space = sorted(target.search_space().items())
        except Exception:
            space = []
        if not objectives and not constraints and not space:
            return {}

        canonical = repr((objectives, constraints, space)).encode()
        return {
            "objectives": ", ".join(objectives),
            "problem": hashlib.sha256(canonical).hexdigest()[:8],
        }

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
