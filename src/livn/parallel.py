from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from mpi4py import MPI


@dataclass(frozen=True)
class Layout:
    """How an MPI job's ranks are carved into independent envs.

    ``total = envs * ranks_per_env + controller`` (= distwq's ``nprocs_per_worker``)

    Arguments:
        ranks_per_env: MPI ranks one Env's solve spans.
        controller: Whether one rank sits out as the controller, as
            ``DistributedEnv`` arranges it.
    """

    ranks_per_env: int = 1
    controller: bool = True

    def __post_init__(self) -> None:
        if int(self.ranks_per_env) < 1:
            raise ValueError(
                f"ranks_per_env is a rank count, so it must be >= 1, "
                f"not {self.ranks_per_env}"
            )

    @property
    def reserved(self) -> int:
        return 1 if self.controller else 0

    def total(self, envs: int) -> int:
        return int(envs) * self.ranks_per_env + self.reserved

    def envs(self, total: int) -> int:
        return max(0, int(total) - self.reserved) // self.ranks_per_env

    def validate(self, total: int) -> Self:
        usable = int(total) - self.reserved
        if usable < self.ranks_per_env:
            raise ValueError(
                f"{total} ranks cannot run even one Env of "
                f"{self.ranks_per_env} rank(s)"
                + (" plus a controller" if self.controller else "")
            )
        if usable % self.ranks_per_env:
            raise ValueError(
                f"{total} ranks do not divide into Envs of "
                f"{self.ranks_per_env} rank(s)"
                + (" plus a controller" if self.controller else "")
                + f"; {usable} would leave {usable % self.ranks_per_env} over. "
                f"Use {self.total(self.envs(total))} ranks, or "
                f"{self.total(self.envs(total) + 1)}"
            )
        return self


@dataclass(frozen=True)
class Parallelism:
    ranks: int = 1
    """MPI ranks this Env's solve is spread over."""

    threads: int = 1
    """Cores this Env uses within one rank."""

    batch: int = 1
    """Replicates :meth:`~livn.types.Env.run_many` solves at once."""

    @property
    def total(self) -> int:
        return self.ranks * self.threads * self.batch

    def __str__(self) -> str:
        parts = [
            f"{n} {name}"
            for n, name in (
                (self.ranks, "rank(s)"),
                (self.threads, "thread(s)"),
                (self.batch, "batched run(s)"),
            )
            if n > 1
        ]
        return " x ".join(parts) if parts else "serial"


def partition(ranks_per_env: int = 1, comm: MPI.Intracomm | None = None, *, cls=None):
    from livn.env import Env

    return (cls or Env).partition(ranks_per_env, comm=comm)


def finalize(*, cls=None) -> None:
    from livn.env import Env

    (cls or Env).finalize()
