from __future__ import annotations
import numpy as np
import MEAutility as MEA_util
from pathlib import Path
from livn.io import MEA

_CONVERT_PLANE_TO_AXIS: dict[str, tupe[int, int]] = {
    "xy": (0,1),
    "xz": (0,2),
    "yz": (1,2),
}
def _extract_livn_coordinates(probe: MEA_util.core.MEA, plane: str, z: float,)->np.ndarray: #helper
    plane = plane.lower().strip()  #removes whitespace, convert to lowercase
    if plane not in _CONVERT_PLANE_TO_AXIS:
        raise ValueError(f"unknown plane")
    axis0, axis1 = _CONVERT_PLANE_TO_AXIS[plane]
    positions = probe.positions #[N, 3] array
    n = len(positions)  #no of electrodes
    coors = np.column_stack([  #4 columns: electrode ID, first axis, second axis, depth(175.0)
        np.arange(n, dtype = float),
        positions[:, axis0],
        positions[:, axis1],
        np.full(n, z, dtype = float),

    ])
    return coors

def MEA_from_MIVos_yaml(yaml_path: str | Path, z: float = 175.0, input_rad: float = 250.0, output_rad: float = 250.0)->MEA:
    yaml_path = Path(yaml_path) #convert to Path object
    if not yaml_path.exists():
        raise FileNotFoundError(f"yaml file not found")
    MEA_util.add_mea(str(yaml_path))  #registers yaml to MEAutility
    import yaml
    with open(yaml_path) as f:
        info = yaml.safe_load(f)  #parse yaml contents into dictionary
    plane = info.get("plane", "xy") #read plane from dict, default xy
    electrode_name = info["electrode_name"]
    probe = MEA_util.return_mea(electrode_name) #load named electrode
    coors = _extract_livn_coordinates(probe, plane, z) #[N, 3]->[N, 4]
    return MEA(
        electrode_coordinates = coors,
        input_radius = input_rad,
        output_radius = output_rad,
    )

