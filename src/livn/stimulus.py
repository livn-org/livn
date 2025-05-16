from livn.types import Array, Float


class Stimulus:
    def __init__(
        self,
        array: Float[Array, "timestep n_gids"] | None = None,
        dt: float = 1.0,
        **meta_data,
    ):
        self.array = array
        self.dt = dt
        self.meta_data = meta_data

    @classmethod
    def from_arg(cls, stimulus) -> "Stimulus":
        if isinstance(stimulus, cls):
            return stimulus

        if stimulus is None:
            return cls()

        if hasattr(stimulus, "shape"):
            return cls(stimulus)

        if isinstance(stimulus, (tuple, list)):
            return cls(*stimulus)

        if isinstance(stimulus, dict):
            return cls(**stimulus)

        raise ValueError("Invalid stimulus", stimulus)
