from machinable.project import Project


class Systems(Project):
    # def on_resolve_element(self, module):
    #     if module == "interface.dmosopt":
    #         m, c = super().on_resolve_element("interface.execution.slurm")
    #         return [
    #             m,
    #             {"mpi": "ibrun"},
    #         ], c

    def on_resolve_remotes(self):
        version = "443884463fc2bec3b072d21492d7524f36bccc8d"
        src = f"url+https://raw.githubusercontent.com/machinable-org/machinable/{version}/docs/examples"
        return {
            "slurm": src + "/slurm-execution/slurm.py",
            "mpi": src + "/mpi-execution/mpi.py",
            "interface.dmosopt": src + "/dmosopt-component/dmosopt.py",
        }
