# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "livn",
#   "datasets",
# ]
# ///
from datasets import load_dataset

from livn.io import MEA
from livn.system import make

system_name = "S1"

dataset = load_dataset("livn-org/livn", name=system_name)

sample = dataset["train_with_noise"][0]
it = sample["trial_it"][0]
t = sample["trial_t"][0]

# use a multi-electrode array to 'observe' the data
system = make(system_name)
mea = MEA.from_directory(system.uri)

cit, ct = mea.channel_recording(system.neuron_coordinates, it, t)

print("Spikes in channel 0:")
print(ct[0])
