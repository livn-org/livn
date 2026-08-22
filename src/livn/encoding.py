from pydantic import PrivateAttr

from livn.types import Encoding
from livn.utils import ObjSpec, import_instance


class H5Inputs(Encoding):
    filepath: str | None = None
    namespace: str = ""
    attribute: str = "Spike Train"
    onset: int = 0
    io_size: int = 1
    microcircuit_inputs: bool = True
    n_trials: int = 1
    equilibration_duration: float = 250.0

    def __call__(self, env, t_end, inputs):
        filepath = self.filepath
        if filepath is None:
            filepath = inputs
        env.apply_stimulus_from_h5(
            filepath,
            self.namespace,
            self.attribute,
            self.onset,
            self.io_size,
            self.microcircuit_inputs,
            self.n_trials,
            self.equilibration_duration,
        )


class ElectrodeStimulus(Encoding):
    """Deliver an electrode policy, sized to the array the env actually has.

    A policy names channels of an array of known width, and the width is not
    known until an env exists -- which is what an encoding is for.

    `channel` is an index into `io.channel_ids`. `None` drives the one coupling
    most strongly into the tissue, since that is what decides whether a cell is
    driven; `input_radius` does not gate stimulation at all.
    """

    policy: ObjSpec = None
    channel: int | None = None

    _resolved: object = PrivateAttr(default=None)

    @property
    def resolved(self):
        """The policy actually delivered, once an env has sized it to its array"""
        return self._resolved

    def __call__(self, env, t_end, inputs):
        import numpy as np

        policy = inputs if self.policy is None else import_instance(self.policy)
        if policy is None:
            raise ValueError(
                "no policy to deliver; give this encoding one, or pass it as "
                "the run's inputs"
            )

        try:
            n_channels = len(env.io.channel_ids)
        except NotImplementedError:
            n_channels = 0
        if not n_channels:
            raise ValueError(
                "this run has no array to stimulate through. A graph does not "
                "bundle one -- the array belongs to the recording it was "
                "measured with -- so give the run an `io`"
            )

        channel = self.channel
        if channel is None:
            channel = int(np.asarray(env.channel_reach()).sum(axis=1).argmax())

        overrides = {}
        if "total_ms" in type(policy).model_fields:
            # fill the run, so the quiet stretch before the first pulse is part
            # of the same command rather than a separate call
            overrides["total_ms"] = float(t_end)

        self._resolved = policy.for_array(n_channels, [channel], **overrides)
        return self._resolved
