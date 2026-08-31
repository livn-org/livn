from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.types import Cell, Env, PopulationName


class CellRegistry(Mapping):
    def __init__(self, env: Env, comm: MPI.Intracomm | None = None):
        self._env = env
        self._comm = comm
        self._cells: dict[str, dict[int, Cell]] = {}
        self._gid_to_population: dict[int, str] | None = None
        self._gids: np.ndarray | None = None

    def add(self, population: PopulationName, cells: dict[int, Cell]) -> CellRegistry:
        self._cells[str(population)] = {int(gid): cell for gid, cell in cells.items()}
        self._gid_to_population = None
        self._gids = None
        return self

    def clear(self) -> CellRegistry:
        self._cells.clear()
        self._gid_to_population = None
        self._gids = None
        return self

    def __getitem__(self, key) -> Any:
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            return self._cell(int(key))
        return self._cells[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._cells))

    def __len__(self) -> int:
        return len(self._cells)

    def __repr__(self) -> str:
        counts = ", ".join(f"{p}: {len(c)}" for p, c in self._cells.items())
        return f"{type(self).__name__}({counts})"

    def _cell(self, gid: int) -> Cell:
        if self._gid_to_population is None:
            self._gid_to_population = {
                int(cell_gid): population
                for population, cells in self._cells.items()
                for cell_gid in cells
            }

        population = self._gid_to_population.get(gid)
        if population is None:
            raise KeyError(gid)

        return self._cells[population][gid]

    @property
    def local_gids(self) -> np.ndarray:
        """Ascending gids of the cells held on this rank"""
        gids = [int(gid) for cells in self._cells.values() for gid in cells]
        return np.array(sorted(gids), dtype=np.int64)

    @property
    def gids(self) -> np.ndarray:
        """Ascending gids of every cell, across all ranks"""
        if self._gids is None:
            gids: set[int] = set()
            for part in self._allgather(self.local_gids):
                gids.update(int(g) for g in part)
            self._gids = np.array(sorted(gids), dtype=np.int64)
        return self._gids

    def get_params(self) -> dict[str, np.ndarray]:
        """Per-cell parameters as ``{name: array}`` in :attr:`gids` order

        A parameter that only some of the cells expose is ``NaN`` for the rest.
        Collective over the communicator, like :attr:`gids`.
        """
        cells = {int(gid): self[int(gid)].get_params() for gid in self.local_gids}

        per_gid: dict[int, dict[str, float]] = {}
        for part in self._allgather(cells):
            per_gid.update(part)

        names = sorted({name for params in per_gid.values() for name in params})

        return {
            name: np.array(
                [
                    float(per_gid.get(int(gid), {}).get(name, np.nan))
                    for gid in self.gids
                ],
                dtype=np.float64,
            )
            for name in names
        }

    def set_params(self, params: dict[str, Any]) -> Env:
        """Set per-cell parameters

        Each value is either a scalar, applied to every cell, or a sequence
        holding one entry per cell in :attr:`gids` order. A rank applies only
        the entries of the cells it owns, so the same array can be passed on
        every rank.
        """
        order = [int(g) for g in self.gids]
        local = {int(g) for g in self.local_gids}
        population_of = self._population_of()

        scoped = []
        for name, value in params.items():
            population, key = self._split_population(name)
            scoped.append((population, key, _as_values(key, value, len(order))))

        env = self._env
        cells = self
        for position, gid in enumerate(order):
            if gid not in local:
                continue
            population = population_of.get(gid)
            applicable = {
                key: value[position]
                for scope, key, value in scoped
                if scope is None or scope == population
            }
            if not applicable:
                continue
            env = cells[gid].set_params(applicable)
            cells = env.cells

        return env

    def _population_of(self) -> dict[int, str]:
        if self._gid_to_population is None:
            self._gid_to_population = {
                int(cell_gid): population
                for population, cells in self._cells.items()
                for cell_gid in cells
            }
        return self._gid_to_population

    def _split_population(self, name: str) -> tuple[str | None, str]:
        """``"EXC:soma.g_pas"`` -> ``("EXC", "soma.g_pas")``."""
        population, sep, rest = str(name).partition(":")
        if not sep:
            return None, str(name)

        known = list(
            getattr(getattr(self._env, "system", None), "populations", None)
            or self._cells
        )
        if population not in known:
            raise KeyError(
                f"{name!r} is scoped to population {population!r}, which this "
                f"system does not have (it has {sorted(known)})"
            )
        return population, rest

    def _allgather(self, value):
        if self._comm is None or self._comm.Get_size() <= 1:
            return [value]
        return self._comm.allgather(value)


def _as_values(name: str, value: Any, n_cells: int) -> Any:
    """Normalize a parameter value into one entry per cell"""
    if np.ndim(value) == 0:
        return [value] * n_cells

    values = list(value)
    if len(values) != n_cells:
        raise ValueError(
            f"parameter {name!r} has {len(values)} values but there are "
            f"{n_cells} cells; pass a scalar to set all of them"
        )
    return values
