# livn visualization

A Jupyter widget for visualizing neural simulation data using React Three Fiber. This widget provides an interactive 3D visualization of neural populations and their spatial distribution.

## Installation

```bash
pip install -e .

npm install
npm run build
```

## Development

To work on the widget's frontend:

```bash
npm run dev
```

## Usage

```python
from livn_visualization import Widget
import ipywidgets as widgets

# Create widget with data and morphologies
widget = Widget(
    data={
        "populations": {
            "PYR": {
                "neurons": [
                    {"i": 0, "x": 1000, "y": 100, "z": 200},
                    # ... more neurons
                ]
            },
            "PVBC": {
                "neurons": [
                    {"i": 0, "x": 1500, "y": 120, "z": 220},
                    # ... more neurons
                ]
            },
            "STIM": {  # Required for MEA3D visualization
                "neurons": [
                    {"i": 0, "x": 2000, "y": 100, "z": 200},
                    # ... more neurons
                ]
            }
        }
    },
    morphologies={  # Optional: SWC morphology data
        "PYR": "# SWC format\n1 1 0 0 0 1 -1\n...",
        "PVBC": "# SWC format\n1 1 0 0 0 1 -1\n..."
    },
    show_mea=False,
    show_dish=False,
    show_morphologies=False
)

mea_checkbox = widgets.Checkbox(value=False, description='Show MEA3D')
dish_checkbox = widgets.Checkbox(value=False, description='Show Dish')
morph_checkbox = widgets.Checkbox(value=False, description='Show Morphologies')

widgets.jslink((mea_checkbox, 'value'), (widget, 'show_mea'))
widgets.jslink((dish_checkbox, 'value'), (widget, 'show_dish'))
widgets.jslink((morph_checkbox, 'value'), (widget, 'show_morphologies'))

widgets.VBox([
    widgets.HBox([mea_checkbox, dish_checkbox, morph_checkbox]),
    widget
])
```

You can also load data and morphologies from files:

```python
# Load data from JSON file
widget = NeuralWidget(
    data='path/to/your/data.json',
    morphologies='path/to/swc/directory',  # Directory containing .swc files
    show_morphologies=True
)

# Or load morphologies from a dictionary of file paths
widget = NeuralWidget(
    data='path/to/your/data.json',
    morphologies={
        'PYR': 'path/to/PYR.swc',
        'PVBC': 'path/to/PVBC.swc'
    },
    show_morphologies=True
)
```

## Data Format

The widget expects data in the following format:

```json
{
    "populations": {
        "population_name": {
            "neurons": [
                {
                    "i": 0,  // neuron index
                    "x": 1000,  // x coordinate
                    "y": 100,   // y coordinate
                    "z": 200    // z coordinate
                },
                // ... more neurons
            ]
        },
        // ... more populations
    }
}
```
