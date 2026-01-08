import glob
import os

import pandas as pd
from huggingface_hub import HfApi
from machinable import Component
from machinable.utils import load_file, save_file, random_str
from pydantic import BaseModel, ConfigDict
from livn.utils import import_object_by_path, P
from livn.integrations.distwq import DistributedEnv
from livn.decoding import GatherAndMerge
from livn.types import Encoding


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

    def __call__(self):
        model = self.config.model
        if model is not None:
            model = import_object_by_path(model)()

        env = DistributedEnv(
            self.config.system,
            model=model,
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
