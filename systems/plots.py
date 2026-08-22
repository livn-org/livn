from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from livn.decoding import (
    peristimulus,
    population_gids,
    spike_waveforms,
    waveform_shape,
)

COLOUR = {"EXC": "#2f6fd0", "INH": "#cf3f57"}
BIN_MS = 20.0


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _spikes(run):
    it = run.spike_ids
    tt = run.spike_times
    return (
        np.asarray([] if it is None else it).astype(np.int64),
        np.asarray([] if tt is None else tt, dtype=float),
    )


def _population_ranges(env):
    return dict(getattr(getattr(env, "system", None), "population_ranges", {}) or {})


def _lanes(ax, ranges):
    """Label the gid axis by population, so a silent one is visible as a band."""
    if not ranges:
        return
    ax.set_ylim(0, max(start + count for start, count in ranges.values()))
    for name, (start, count) in sorted(ranges.items(), key=lambda kv: kv[1][0]):
        if start > 0:
            ax.axhline(start, color="C3", lw=0.8, alpha=0.6)
        ax.text(
            0.002,
            start + count / 2,
            name,
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color="C3",
        )


class Figure(BaseModel):
    """A drawing of one run."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""

    def __call__(self, run, path: str, *, env=None, encoding=None) -> str:
        raise NotImplementedError


class Raster(Figure):
    warmup: float = 0.0
    duration: float | None = None
    bin_ms: float = BIN_MS

    def __call__(self, run, path, *, env=None, encoding=None):
        plt = _plt()
        it, tt = _spikes(run)

        keep = tt >= self.warmup
        it, tt = it[keep], tt[keep] - self.warmup
        duration = self.duration
        if duration is None:
            duration = float(run.duration or (tt.max() if len(tt) else 1.0))
            duration -= self.warmup

        fig, (ax, bx) = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )
        ax.plot(tt / 1000.0, it, ",k", alpha=0.5)
        ax.set_ylabel("cell")
        ax.set_xlim(0, duration / 1000.0)
        _lanes(ax, _population_ranges(env))
        ax.set_title(self.title, fontsize=10)

        bins = np.arange(0, duration + self.bin_ms, self.bin_ms)
        counts, _ = np.histogram(tt, bins=bins)
        bx.fill_between(bins[:-1] / 1000.0, counts, step="post", alpha=0.7)
        bx.set_xlabel("time (s)")
        bx.set_ylabel(f"spikes / {self.bin_ms:g} ms")

        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path


class Traces(Figure):
    cells: int = 3
    pre_ms: float = 4.0
    post_ms: float = 10.0
    section: str = "soma"
    overlay: int = 200

    def __call__(self, run, path, *, env=None, encoding=None):
        plt = _plt()

        iv = np.asarray(run.voltage_ids if run.voltage_ids is not None else [])
        vv = np.asarray(run.voltage if run.voltage is not None else [])
        sv = run.voltage_sections
        if not len(iv):
            raise ValueError(
                "no voltage was recorded, so there is nothing to trace. Ask the "
                "decoding for it with `voltages=True`"
            )

        # by compartment name, not by which row swings furthest: a dendritic
        # spike is smaller and broader, and guessing reads as a cell type
        if sv is not None:
            keep = np.asarray(sv) == self.section
            iv, vv = iv[keep].astype(np.int64), vv[keep]
        iv = iv.astype(np.int64)

        it, tt = _spikes(run)
        dt = float(run.voltage_dt or 0.1)
        times = np.arange(vv.shape[1]) * dt
        where = population_gids(env)

        names = sorted({where.get(int(g), "all") for g in iv})
        fig, axes = plt.subplots(
            len(names),
            2,
            figsize=(13.5, 3.1 * len(names) + 1.4),
            gridspec_kw={"width_ratios": [2.5, 1]},
            squeeze=False,
        )

        table = []
        for row, name in enumerate(names):
            rows = [i for i, g in enumerate(iv) if where.get(int(g), "all") == name]
            colour = COLOUR.get(name, "#444444")

            counts = {i: int((it == iv[i]).sum()) for i in rows}
            shown = sorted(rows, key=lambda i: -counts[i])[: self.cells]

            ax = axes[row][0]
            offset = 0.0
            for i in shown:
                ax.plot(times / 1000.0, vv[i] + offset, lw=0.7, color=colour)
                ax.text(
                    times[0] / 1000.0,
                    offset - 55,
                    f"gid {int(iv[i])}",
                    fontsize=8,
                    color="#555",
                    va="center",
                )
                offset += 140.0
            ax.set_xlabel("time (s)")
            ax.set_ylabel("membrane V (mV, offset)")

            active = len({int(iv[i]) for i in rows if counts[i]})
            window = max(times[-1] / 1000.0, 1e-9) if len(times) else 1.0
            rate = sum(counts.values()) / max(len(rows), 1) / window
            ax.set_title(
                f"{name} -- {len(shown)} of {len(rows)} shown, {self.section}"
                f"     {rate:.3f} Hz/cell, {active}/{len(rows)} active",
                loc="left",
                fontsize=11,
            )

            cuts = [
                spike_waveforms(
                    vv[i],
                    times,
                    tt[it == iv[i]],
                    pre_ms=self.pre_ms,
                    post_ms=self.post_ms,
                    dt=dt,
                )
                for i in rows
                if counts[i]
            ]
            cuts = [c for c in cuts if len(c)]
            waves = np.vstack(cuts) if cuts else np.empty((0, 0))
            shape = waveform_shape(waves, pre_ms=self.pre_ms, dt=dt)

            bx = axes[row][1]
            if len(waves):
                t = np.arange(waves.shape[1]) * dt - self.pre_ms
                for one in waves[: self.overlay]:
                    bx.plot(t, one, lw=0.4, alpha=0.25, color=colour)
                bx.plot(t, waves.mean(axis=0), lw=2.0, color="black")
                bx.set_title(
                    f"{shape['n_spikes']} spikes   {shape['amplitude_mV']:.0f} mV   "
                    f"half-width {shape['half_width_ms']:.2f} ms   "
                    f"AHP {shape['ahp_mV']:.0f} mV",
                    loc="left",
                    fontsize=9.5,
                )
            else:
                bx.set_title("no spikes to align", loc="left", fontsize=9.5)
            bx.set_xlabel("ms from peak")
            bx.set_ylabel("mV")
            table.append((name, len(rows), rate, active, shape))

        fig.suptitle(self.title, fontsize=12, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(path, dpi=160)
        plt.close(fig)

        _print_shapes(table)
        return path


def _print_shapes(table):
    print(
        f"\n{'':6}{'cells':>7}{'Hz/cell':>10}{'active':>9}{'threshold':>11}"
        f"{'peak':>9}{'amplitude':>11}{'half-width':>12}{'AHP':>8}"
    )
    for name, n, rate, active, shape in table:
        row = f"{name:6}{n:>7}{rate:>10.3f}{active:>6}/{n:<2}"
        if shape:
            row += (
                f"{shape['threshold_mV']:>11.1f}{shape['peak_mV']:>9.1f}"
                f"{shape['amplitude_mV']:>11.1f}{shape['half_width_ms']:>12.2f}"
                f"{shape['ahp_mV']:>8.1f}"
            )
        else:
            row += f"{'no spikes':>11}"
        print(row)


class StimulusResponse(Figure):
    pre_ms: float = 50.0
    post_ms: float = 150.0
    bin_ms: float = 5.0

    def __call__(self, run, path, *, env=None, encoding=None):
        plt = _plt()
        policy = getattr(encoding, "resolved", None)
        if policy is None or not hasattr(policy, "schedule"):
            raise ValueError(
                "this figure reads the pulse times off the policy that was "
                "delivered, so it needs the `ElectrodeStimulus` encoding that "
                "produced them"
            )

        schedule = policy.schedule()
        by_amplitude: dict[float, list[float]] = {}
        for at, amplitude in schedule:
            by_amplitude.setdefault(float(amplitude), []).append(float(at))

        it, tt = _spikes(run)
        ranges = _population_ranges(env)
        where = population_gids(env)
        names = sorted(ranges) or ["all"]

        amplitudes = sorted(by_amplitude)
        fig, axes = plt.subplots(
            len(amplitudes) + 2,
            1,
            figsize=(11, 2.0 * len(amplitudes) + 6),
            gridspec_kw={"hspace": 0.45},
        )

        ax = axes[0]
        ax.plot(tt / 1000.0, it, ",k", alpha=0.5)
        for at, _amplitude in schedule:
            ax.axvline(at / 1000.0, color="C3", lw=0.5, alpha=0.5)
        _lanes(ax, ranges)
        ax.set_xlabel("time (s)")
        ax.set_title(f"{self.title}  --  whole recording", loc="left", fontsize=10)

        edges = np.arange(-self.pre_ms, self.post_ms + self.bin_ms, self.bin_ms)
        curve: dict[str, list[float]] = {name: [] for name in names}
        for row, amplitude in enumerate(amplitudes):
            onsets = by_amplitude[amplitude]
            relative, who = peristimulus(tt, it, onsets, self.pre_ms, self.post_ms)
            bx = axes[row + 1]
            for name in names:
                mask = (
                    np.ones(len(who), dtype=bool)
                    if name == "all"
                    else np.asarray(
                        [where.get(int(g)) == name for g in who], dtype=bool
                    )
                )
                counts = np.histogram(relative[mask], bins=edges)[0] / max(
                    len(onsets), 1
                )
                bx.step(
                    edges[:-1],
                    counts,
                    where="post",
                    label=name,
                    color=COLOUR.get(name, "#444444"),
                )
                # against the window before the pulse, in the same window: a
                # bare post-pulse count cannot separate a response from a
                # network that was already busy
                before = counts[edges[:-1] < 0].sum()
                curve[name].append(float(counts[edges[:-1] >= 0].sum() - before))
            bx.axvline(0, color="C3", lw=1.0)
            bx.set_ylabel("spikes / trial")
            bx.set_title(
                f"{amplitude:g}  --  {len(onsets)} trials", loc="left", fontsize=9
            )
            if row == 0:
                bx.legend(fontsize=7, ncol=len(names))
        axes[len(amplitudes)].set_xlabel("time from pulse (ms)")

        cx = axes[-1]
        for name in names:
            cx.plot(
                amplitudes,
                curve[name],
                "o-",
                label=name,
                color=COLOUR.get(name, "#444444"),
            )
        cx.axhline(0, color="#999999", lw=0.8)
        cx.set_xlabel("amplitude")
        cx.set_ylabel("evoked spikes / trial")
        cx.set_title(
            "response curve, against the pre-pulse window", loc="left", fontsize=9
        )
        cx.legend(fontsize=7, ncol=len(names))

        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path
