from livn.types import Array, Float, Int


class Stimulus:
    def __init__(
        self,
        array: Float[Array, "timestep n_gids"] | None = None,
        dt: float = 1.0,
        gids: Int[Array, "n_gids"] | None = None,
        **meta_data,
    ):
        self.array = array
        if dt <= 0:
            raise ValueError("Stimulus dt must be positive")
        self.dt = dt
        self.gids = gids
        self.meta_data = meta_data

    def __iter__(self):
        yield from zip(self.gids, self.array.T)

    def __len__(self):
        return self.array.shape[-1]

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
