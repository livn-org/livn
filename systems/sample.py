import glob
import os

from huggingface_hub import HfApi
from machinable import Component
from machinable.utils import load_file, save_file, random_str
from pydantic import BaseModel, ConfigDict
from livn.utils import import_object_by_path, P
from livn.integrations.distwq import DistributedEnv
from livn.decoding import GatherAndMerge, Slice
from livn.types import Encoding
from livn.env import Env
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
import numpy as np


class Raw(GatherAndMerge):
    def __call__(self, env, it, tt, iv, vv, im, mp):
        data = super().__call__(env, it, tt, iv, vv, im, mp)
        if data is None:
            return

        env.clear()

        it, tt, iv, vv, im, mp = data
        return {
            "duration": self.duration,
            "it": it,
            "tt": tt,
            "iv": iv,
            "vv": vv,
            "im": im,
            "mp": mp,
        }


class WithouInput(Encoding):
    def __call__(self, env, t_end, inputs):
        return None


class Sample(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        duration: int = 31000
        samples: int | tuple[int, int] = 100
        noise: bool = True

        system: str = "./systems/graphs/EI3"
        model: str | None = None
        encoding: str | None = "systems.sample.WithouInput"
        encoding_kwargs: dict = {}
        decoding: str = "systems.sample.Raw"
        decoding_kwargs: dict = {}

        output_directory: str = "???"
        nprocs_per_worker: int = 1

    def model(self):
        model = self.config.model
        if model is not None:
            model = import_object_by_path(model)()

        return model

    def __call__(self):
        env = DistributedEnv(
            self.config.system,
            model=self.model(),
            subworld_size=self.config.nprocs_per_worker,
        )

        env.init()
        env.apply_model_defaults(noise=self.config.noise)

        if env.is_root():
            encoding = self.config.encoding
            if encoding is not None:
                encoding = import_object_by_path(self.config.encoding)(
                    **self.config.encoding_kwargs
                )
            decoding = import_object_by_path(self.config.decoding)(
                duration=self.config.duration, **self.config.decoding_kwargs
            )

            batch_size = max(1, (P.size() - 1) // self.config.nprocs_per_worker)
            num_batches = (self.config.samples + batch_size - 1) // batch_size

            for batch_id in range(num_batches):
                batch_start = batch_id * batch_size
                batch_end = min(batch_start + batch_size, self.config.samples)

                for i in range(batch_start, batch_end):
                    env.submit_call(decoding, batch_start + i, encoding)

                for _ in range(batch_start, batch_end):
                    response = env.receive_response()
                    # we assume that the reduction happens
                    # through decoding on subworld root 0
                    payload = response[0]

                    save_file(
                        [self.config.output_directory, random_str(8) + ".p"], payload
                    )

        env.shutdown()

    def count(self):
        print(len(list(glob.glob(os.path.join(self.config.output_directory, "*.p")))))

    def merge(self, include_voltage: bool = False):
        dest = os.path.join(
            "./systems/datasets",
            os.path.basename(self.config.system),
            "ct-cp" + ("-cv" if include_voltage else ""),
        )
        os.makedirs(dest, exist_ok=True)

        train_dir = os.path.join(dest, "train")
        test_dir = os.path.join(dest, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        writers = {}
        split_counts = {"train": 0, "test": 0}

        env = Env(self.config.system, self.model())

        for i, file_path in enumerate(
            sorted(glob.glob(os.path.join(self.config.output_directory, "*.p")))
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

            data = load_file(file_path)

            # remove warmup
            it, tt, iv, vv, im, mp = Slice(start=1600, stop=data["duration"])(
                env,
                it=data["it"],
                tt=data["tt"],
                iv=data["iv"],
                vv=data["vv"],
                im=data["im"],
                mp=data["mp"],
            )

            cit, ctt = env.channel_recording(it, tt)
            p = env.potential_recording(mp)

            cit_array = np.array(
                [cit.get(k, np.array([])) for k in range(env.io.num_channels)],
                dtype=object,
            )
            ctt_array = np.array(
                [ctt.get(k, np.array([])) for k in range(env.io.num_channels)],
                dtype=object,
            )

            result = {
                "sample_id": fn,
                "cit": cit_array,
                "ctt": ctt_array,
                "cp": p,
            }

            if include_voltage:
                result["civ"], result["cvv"] = env.channel_recording(iv, vv)

            # write eagerly to free memory
            if split not in writers:
                split_dir = train_dir if split == "train" else test_dir
                writer = ArrowWriter(
                    path=os.path.join(split_dir, "data-00000-of-00001.arrow"),
                    writer_batch_size=1,
                )
                writers[split] = writer

            writers[split].write(result)
            split_counts[split] += 1

            del data, result, it, tt, iv, vv, im, mp, cit, ctt, p

        for split, writer in writers.items():
            writer.finalize()
            writer.close()

        import shutil
        import uuid

        for split in writers.keys():
            split_dir = train_dir if split == "train" else test_dir
            arrow_file = os.path.join(split_dir, "data-00000-of-00001.arrow")

            tmp_split_dir = os.path.join(dest, f"tmp_{split}_{uuid.uuid4().hex}")
            os.makedirs(tmp_split_dir, exist_ok=True)

            try:
                ds = Dataset.from_file(arrow_file)
                ds.save_to_disk(tmp_split_dir)

                os.remove(arrow_file)

                for filename in os.listdir(tmp_split_dir):
                    shutil.move(
                        os.path.join(tmp_split_dir, filename),
                        os.path.join(split_dir, filename),
                    )
            finally:
                if os.path.exists(tmp_split_dir):
                    shutil.rmtree(tmp_split_dir, ignore_errors=True)

        dataset_dict_json = {
            "splits": {split: {"name": split} for split in writers.keys()}
        }
        import json

        with open(os.path.join(dest, "dataset_dict.json"), "w") as f:
            json.dump(dataset_dict_json, f)

    def publish(self, repo_id: str = "livn-org/livn"):
        api = HfApi(token=os.getenv("HF_TOKEN"))

        dataset_base = os.path.join(
            "./systems/datasets",
            os.path.basename(self.config.system),
        )

        if os.path.exists(dataset_base):
            print(f"Uploading dataset directory {dataset_base} ...")
            api.upload_folder(
                folder_path=dataset_base,
                path_in_repo=dataset_base,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"Successfully uploaded {dataset_base}")
        else:
            print(f"Directory {dataset_base} not found, skipping upload")
