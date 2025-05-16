import os

import numpy as np
import pandas as pd
from machinable import Interface
from miv_simulator.spikedata import read_spike_events
from pydantic import BaseModel, ConfigDict, Field

from livn.system import read_cells_meta_data


def neuroh5_to_parquet(neuroh5_file: str, output_filepath: str | None = None) -> str:
    if output_filepath is None:
        output_filepath = neuroh5_file.replace(".h5", "") + ".parquet"

    meta = read_cells_meta_data(neuroh5_file)

    spikes = read_spike_events(
        neuroh5_file,
        meta.population_names,
        namespace_id="Spike Events",
        spike_train_attr_name="t",
    )
    samples_df = pd.DataFrame.from_dict(
        {
            str(s): {
                "seed": 0,
                "trials": spikes["n_trials"],
                "t_end": 10_000,
                "features": [0.0, 0.0, 0.0],
                "trial_it": [
                    np.asarray(spikes["spkindlst"][0][s], dtype=np.int32).tolist()
                ],
                "trial_t": [
                    np.asarray(spikes["spktlst"][0][s], dtype=np.float32).tolist()
                ],
            }
            for s in range(1)
        },
        orient="index",
    )
    samples_fp = os.path.join(output_filepath)
    samples_df.to_parquet(samples_fp, index=False)

    return output_filepath


class Convert(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        filepath: str = Field("???")
        output_filepath: str | None = None
        mode: str = "auto"

    def neuroh5_to_parquet(self):
        print(
            "Written to ",
            neuroh5_to_parquet(self.config.filepath, self.config.output_filepath),
        )
