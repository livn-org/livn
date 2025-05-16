import glob
import importlib
import os
from functools import partial

import numpy as np
import pandas as pd
from dmosopt import config
from huggingface_hub import HfApi
from machinable import Interface, get
from machinable.utils import find_subclass_in_module, load_file, object_hash, save_file
from pydantic import BaseModel, ConfigDict

from livn import io
from livn.env import Env
from livn.types import Model


def _concat(a):
    if len(a) == 1:
        return a[0]

    if len(a) > 1:
        return np.concatenate(a)

    return []


def get_model(model):
    return find_subclass_in_module(importlib.import_module(model), base_class=Model)()


live_envs = []


def make_env(system, model, subworld_size, comm=None):
    env = Env(
        system,
        model=get_model(model),
        io=io.MEA.from_json(os.path.join(system, "mea.json"), comm=False),
        comm=comm,
        subworld_size=subworld_size,
    )

    env.init()

    env.record_spikes()

    return env


def obj_fun_init(system, model, encoding, noise, subworld_size, trials=1, worker=None):
    env = make_env(
        system,
        model,
        subworld_size,
        worker.merged_comm,
    )

    env.apply_model_defaults(noise=noise)

    live_envs.append(env)

    encoder = config.import_object_by_path(encoding)(env)

    return partial(obj_fun, env=env, encoder=encoder, trials=trials)


def controller_init(system, model, encoding, subworld_size):
    env = Env(
        system,
        model=get_model(model),
        io=io.MEA.from_json(os.path.join(system, "mea.json"), comm=False),
        subworld_size=subworld_size,
    )

    config.import_object_by_path(encoding)(env)

    live_envs.append(env)


def obj_fun(x, env, encoder, trials):
    env.clear()

    # sample from feature space and encode
    seed = int(x["seed"])

    encoder.feature_space.seed(seed)
    features = encoder.feature_space.sample()

    trial_inputs, trial_length = encoder(features)

    # construct trialed inputs
    warmup = 250
    t_end = warmup + trial_length * trials
    inputs = np.zeros([t_end, trial_inputs.shape[-1]])
    for trial in range(trials):
        pt = warmup + trial * trial_length
        inputs[pt : pt + trial_inputs.shape[0], :] = trial_inputs

    stimulus = env.cell_stimulus(inputs)

    it, t, iv, v = env.run(t_end, stimulus, root_only=False)

    # discard warmup
    it, t = it[t >= warmup], t[t >= warmup] - warmup
    t_end = t_end - warmup

    return {
        "seed": seed,
        "features": features,
        "it": it,
        "t": t,
        "t_end": t_end,
        "trials": trials,
        "trial_inputs": trial_inputs,
    }


def obj_reduce(payload, output_directory):
    it = []
    t = []
    t_end = 0
    trials = None
    trial_inputs = None
    seed = None
    features = None
    for result in payload:
        if features is None:
            features = result[0]["features"]
        else:
            np.testing.assert_equal(features, result[0]["features"])
        it.append(result[0]["it"])
        t.append(result[0]["t"])
        t_end = max(t_end, result[0]["t_end"])
        trials = result[0]["trials"]
        trial_inputs = result[0]["trial_inputs"]
        seed = result[0]["seed"]

    it, t = _concat(it), _concat(t)

    trial_length = t_end // trials
    trial_it = []
    trial_t = []
    for i in range(trials):
        trial_start = i * trial_length
        trial_end = (i + 1) * trial_length

        mask = (t >= trial_start) & (t < trial_end)

        trial_it.append(it[mask])
        trial_t.append(t[mask] - trial_start)

    payload = {"seed": seed, "trials": trials, "t_end": t_end}
    name = object_hash(payload)
    payload.update({"features": features, "trial_it": trial_it, "trial_t": trial_t})

    save_file([output_directory, f"{name[:8]}.p"], payload)

    return {0: np.array([0.0])}


def seed_generator(n, s, local_random, maxiter=0):
    n_initial = n // s
    return np.random.randint(0, 2**16, size=(n, 1))


class Sample(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        stimulate: int = 1
        trials: int = 1
        samples: int | tuple[int, int] = 100
        noise: bool = True

        system: str = "./systems/data/S1"
        model: str = "livn.models.rcsd"
        encoding: str = "livn.integrations.gym.PulseEncoding"
        output_directory: str = "???"

        nprocs_per_worker: int = 1

    def launch(self):
        get(
            "interface.dmosopt",
            {
                "dopt_params": {
                    "opt_id": "default",
                    "obj_fun_init_name": "systems.sample.obj_fun_init",
                    "obj_fun_init_args": {
                        "system": self.config.system,
                        "model": self.config.model,
                        "encoding": self.config.encoding,
                        "noise": self.config.noise,
                        "trials": self.config.trials,
                        "subworld_size": "${...nprocs_per_worker}",
                    },
                    "space": {"seed": [0, int(2**16)]},
                    "objective_names": ["dummy"],
                    "controller_init_fun_name": "systems.sample.controller_init",
                    "controller_init_fun_args": {
                        "system": "${..obj_fun_init_args.system}",
                        "model": "${..obj_fun_init_args.model}",
                        "encoding": "${..obj_fun_init_args.encoding}",
                        "subworld_size": "${...nprocs_per_worker}",
                    },
                    "reduce_fun_name": "systems.sample.obj_reduce",
                    "reduce_fun_args": (self.config.output_directory,),
                    "problem_parameters": {},
                    "initial_method": "systems.sample.seed_generator",
                    "n_initial": self.config.samples,
                    "initial_maxiter": 0,
                    "n_epochs": 0,
                    "surrogate_method_name": None,
                    "surrogate_method_kwargs": {},
                    "feasibility_method_name": None,
                    "feasibility_method_kwargs": {},
                    "save": True,
                },
                "nprocs_per_worker": self.config.nprocs_per_worker,
                "ranks": -1,
            },
        ).launch()

        return self

    def merge(self):
        samples = {"train": {}, "test": {}}
        for i, file_path in enumerate(
            glob.glob(os.path.join(self.config.output_directory, "*.p"))
        ):
            fn = os.path.basename(file_path).replace(".p", "")
            if len(fn) != 8:
                continue
            if isinstance(self.config.samples, int):
                split = "train" if i < self.config.samples else "test"
            else:
                train, test = self.config.samples
                split = "train" if i < train else "test"
                if i >= train + test:
                    continue
            samples[split][fn] = load_file(file_path)
            samples[split][fn]["features"] = samples[split][fn]["features"].tolist()

        for split in ["train", "test"]:
            samples_df = pd.DataFrame.from_dict(samples[split], orient="index")
            samples_fp = os.path.join(self.config.output_directory, split + ".parquet")
            samples_df.to_parquet(samples_fp, index=False)
            print(f"Written samples {len(samples[split])} to {samples_fp}")

    def count(self):
        print(len(list(glob.glob(os.path.join(self.config.output_directory, "*.p")))))

    def publish(self):
        api = HfApi(token=os.getenv("HF_TOKEN"))

        for split in ["train", "test"]:
            samples_fp = os.path.join(self.config.output_directory, split + ".parquet")
            if os.path.exists(samples_fp):
                print(f"Uploading {samples_fp} ...")
                api.upload_file(
                    path_or_fileobj=samples_fp,
                    path_in_repo=f"samples/S3/{split}.parquet",
                    repo_id="frthjf/test",
                    repo_type="dataset",
                )
                print(f"Successfully uploaded {split}")
            else:
                print(f"File {samples_fp} not found, skipping upload")
