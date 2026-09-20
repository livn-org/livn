import json
import shutil

import pandas as pd

ANATOMY_REPORT = (
    "burst_width_ms",
    "spikes_per_unit_per_burst",
    "burst_onset_peak",
    "burst_interval_cv",
    "units_recruited_per_burst",
    "burst_rate",
    "fano_factor",
    "pop_autocorr_tau",
    "max_synchronous_peak",
    "mean_channel_correlation",
    "mfr",
    "isi_cv",
)


def terminal_width(default: int = 200) -> int:
    return shutil.get_terminal_size((default, 24)).columns


def wide(table: pd.DataFrame) -> str:
    with pd.option_context(
        "display.max_columns", None, "display.width", terminal_width()
    ):
        return table.to_string(index=False)


def as_json(payload) -> str:
    return json.dumps(payload, indent=4, sort_keys=True)


def anatomy_table(rows: list[dict], targets: dict, bands: dict, names=ANATOMY_REPORT):
    import numpy as np

    width = max((len(n) for n in names), default=0)
    head = f"{'feature':<{width}}" + "".join(f"{f'#{i}':>10}" for i in range(len(rows)))
    head += f"{'mean':>10}{'observed':>11}{'band':>19}   "
    lines = [head, "-" * len(head)]

    for name in names:
        values = np.asarray(
            [float(row.get(name, float("nan"))) for row in rows], dtype=float
        )
        finite = values[np.isfinite(values)]
        mean = float(finite.mean()) if finite.size else float("nan")
        line = f"{name:<{width}}" + "".join(f"{v:>10.3f}" for v in values)
        line += f"{mean:>10.3f}"

        observed = targets.get(name)
        line += f"{observed:>11.3f}" if observed is not None else f"{'':>11}"

        band = bands.get(name)
        if band is None:
            line += f"{'':>19}   "
        else:
            lo, hi = float(band[0]), float(band[1])
            line += f"{f'{lo:.3f} - {hi:.3f}':>19}   "
            if finite.size:
                line += "in" if lo <= mean <= hi else ("low" if mean < lo else "high")
            else:
                line += "unmeasured"
        lines.append(line.rstrip())

    return "\n".join(lines)


def front_table(features, bands: dict, decoded: dict | None = None) -> pd.DataFrame:
    rows = []
    for i in range(len(features)):
        row = {"loc": i}
        n_in = 0
        for name in features.columns:
            value = float(features[name].iloc[i])
            lo_hi = bands.get(name)
            inside = lo_hi is not None and lo_hi[0] <= value <= lo_hi[1]
            n_in += bool(inside)
            row[name] = f"{value:.4g}{'*' if inside else ''}"
        if bands:
            row["in_band"] = f"{n_in}/{len(bands)}"
            row["_n"] = n_in
        for name, value in (decoded or {}).get(i, {}).items():
            row[name] = "--" if value is None else f"{float(value):.4g}"
        rows.append(row)
    return pd.DataFrame(rows)


def gist(best, bands: dict, n_evals: int) -> str:
    features = best.get("f")
    if features is None or len(features) == 0:
        return "  no solutions yet"

    row = features.iloc[0]

    def inside(name) -> bool:
        band = bands.get(name)
        return band is not None and band[0] <= float(row[name]) <= band[1]

    head = f"  {n_evals} evaluations, {len(features)} on the front"
    if bands:
        scored = [name for name in features.columns if name in bands]
        head += f", best {sum(inside(n) for n in scored)}/{len(bands)} in band"
    constraints = best.get("c")
    if constraints is not None and len(constraints):
        head += ", feasible" if (constraints.iloc[0] > 0).all() else ", infeasible"

    values = "  ".join(
        f"{name} {float(row[name]):.4g}{'*' if inside(name) else ''}"
        for name in features.columns
    )
    return f"{head}\n    {values}"


def band_legend(bands: dict) -> str:
    if not bands:
        return "\n(the target states no feature bands, so nothing is marked)"
    lines = ["\n* = inside the measured band:"]
    lines += [
        f"    {name:<24} {lo:>10.4g} - {hi:<10.4g}" for name, (lo, hi) in bands.items()
    ]
    return "\n".join(lines)


def in_band_counts(features, bands: dict) -> list[tuple[int, int]]:
    """`(n_in_band, loc)` per solution, best first."""
    counts = [
        (
            sum(
                lo <= float(features[name].iloc[i]) <= hi
                for name, (lo, hi) in bands.items()
                if name in features.columns
            ),
            i,
        )
        for i in range(len(features))
    ]
    counts.sort(reverse=True)
    return counts


def _floor_note(plan: dict) -> str:
    """Where the ranks-per-worker floor came from, when it is not the default."""
    floor, asked = plan["floor"], plan.get("floor_asked", plan["floor"])
    if floor == asked:
        return f"  (target asks for at least {floor})" if floor > 1 else ""
    return f"  (floor {floor}, raised from the target's {asked})"


def layout_report(plan: dict, config, stated_memory: bool) -> str:
    """What `--sizing` prints: the layout, and how to launch it."""
    lines = [
        f"\n  system        {plan['system']}",
        f"  selection     {plan['selection'] or 'none'}",
        f"  node          {plan['node_gib']:.1f} GiB x {plan['cores_per_node']} "
        f"cores, planned to {plan['headroom']:.0%}"
        + (
            ""
            if stated_memory
            else "  (this machine -- state LIVN_WORKER_MEMORY_MAX for a cluster)"
        ),
        f"  worker        {plan['worker_gib']:.1f} GiB over "
        f"{plan['ranks_per_worker']} rank(s)" + _floor_note(plan),
        f"  layout        {plan['nodes']} node(s) x {plan['ranks_per_node']} "
        f"rank(s) = {plan['total_ranks']} ranks "
        f"({plan['node_used_gib']:.1f} GiB used per node)",
        f"  workers       {plan['workers']} for {plan['workers_wanted']} samples "
        "per epoch"
        + (
            ""
            if plan["workers"] >= plan["workers_wanted"]
            else "  -- epochs will queue"
        ),
        f"  evaluations   {plan['initial_evals']} in the first epoch "
        f"({plan['space']} dims x n_initial={plan['n_initial']}), then "
        f"{plan['workers_wanted']} per epoch x {plan['n_epochs'] - 1} "
        f"= {plan['total_evals']} in all",
    ]
    if plan["nodes"] == 1:
        lines.append(
            "\n  launch with\n"
            f'    livn systems mpi tune ... **resources=\'{{"-n": '
            f"{plan['total_ranks']}}}' --launch\n"
        )
    else:
        lines.append(
            "\n  launch with\n"
            "    livn systems slurm tune ... --launch\n"
            "  which takes --nodes and --ntasks-per-node from this plan. A "
            "local mpi run\n  has one node; pass max_nodes=1 to size for it.\n"
        )
    return "\n".join(lines)
