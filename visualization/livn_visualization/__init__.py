import base64
import json

import anywidget
import traitlets


class Widget(anywidget.AnyWidget):
    _esm = "src/index.js"
    data = traitlets.Dict().tag(sync=True)
    show_mea = traitlets.Bool(False).tag(sync=True)
    show_dish = traitlets.Bool(False).tag(sync=True)
    show_morphologies = traitlets.Bool(False).tag(sync=True)
    morphologies = traitlets.Dict().tag(sync=True)

    def __init__(
        self,
        data=None,
        show_mea=False,
        show_dish=False,
        show_morphologies=False,
        morphologies=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.show_mea = show_mea
        self.show_dish = show_dish
        self.show_morphologies = show_morphologies

        if data is not None:
            if isinstance(data, str):
                with open(data, "r") as f:
                    self.data = json.load(f)
            else:
                self.data = data

        if morphologies is not None:
            if isinstance(morphologies, str):
                # Load morphologies from a directory
                import glob
                import os

                self.morphologies = {}
                for swc_file in glob.glob(os.path.join(morphologies, "*.swc")):
                    with open(swc_file, "r") as f:
                        name = os.path.splitext(os.path.basename(swc_file))[0]
                        self.morphologies[name] = f.read()
            elif isinstance(morphologies, dict):
                # Load morphologies from a dictionary of file paths
                self.morphologies = {}
                for name, path in morphologies.items():
                    with open(path, "r") as f:
                        self.morphologies[name] = f.read()
            else:
                self.morphologies = morphologies
