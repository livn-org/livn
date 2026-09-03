// Install the PyEmscripten wheel into Pyodide and run a simulation, so a
// broken wasm wheel fails the release rather than the first user.
import { loadPyodide } from "pyodide";
import fs from "fs";

const wheel = process.argv[2];
if (!wheel) {
    console.error("usage: pyodide_smoke.mjs <wheel>");
    process.exit(2);
}

function brief(e) {
    return String(e.message || e)
        .split("\n")
        .filter((l) => !l.includes("pyodide.asm"))
        .slice(0, 20)
        .join("\n");
}

const py = await loadPyodide();
const name = wheel.split("/").pop();
py.FS.writeFile("/" + name, new Uint8Array(fs.readFileSync(wheel)));
await py.loadPackage("micropip");

try {
    await py.runPythonAsync(`
import micropip
await micropip.install("emfs:/${name}")
`);
    await py.runPythonAsync(`
import numpy as np
from livn.backend import backend
assert backend() == "native", backend()

from livn.env import Env
from livn.stimulus import Stimulus

env = Env({"EXC": 1}).init()
env.record_spikes()
current = np.zeros((2400, 1))
current[80:, 0] = 1.0
run = env.run(60.0, stimulus=Stimulus.from_current(current, dt=0.025, gids=np.array([0])))

# the spike times the NEURON backend produces for this protocol, which the
# native backend reproduces on every platform it is built for
expected = [3.6, 9.1, 14.225, 19.275, 24.325, 29.375, 34.425, 39.525, 44.6, 49.675, 54.775, 59.875]
got = [round(float(t), 6) for t in run.spike_times]
assert got == expected, f"{got} != {expected}"
print("pyodide smoke ok:", len(got), "spikes, identical to the native build")
env.close()
`);
} catch (e) {
    console.error("PYODIDE SMOKE FAILED:\n" + brief(e));
    process.exit(1);
}
