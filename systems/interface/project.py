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
        version = "85fc17f32babf2858c18104c64e1aac6ad991363"
        src = f"url+https://raw.githubusercontent.com/machinable-org/machinable/{version}/integrations"
        return {
            "slurm": src + "/slurm/slurm.py",
            "mpi": src + "/mpi/mpi.py",
            "interface.dmosopt": src + "/dmosopt/dmosopt.py",
        }
