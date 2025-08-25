from typing import Literal, Optional, Union

import os
import subprocess
import sys
import time

import arrow
import yaml
from machinable import Execution, Project, Index
from machinable.errors import ExecutionFailed
from machinable.utils import chmodx, run_and_stream
from pydantic import BaseModel, ConfigDict
from pathlib import Path
import distwq


def make_relative_if_subpath(path, root):
    path_abs = Path(path).resolve()
    root_abs = Path(root).resolve()

    try:
        rel_path = path_abs.relative_to(root_abs)
        return str(rel_path)
    except ValueError:
        return str(path)


class TACC(Execution):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        preamble: Optional[str] = ""
        mpi: Optional[str] = "ibrun"
        mpi_args: str = ""
        python: Optional[str] = None
        throttle: float = 0.5
        confirm: bool = True
        copy_project_source: bool = False
        resume_failed: Union[bool, Literal["new", "skip"]] = False
        dry: bool = False

    def on_before_dispatch(self):
        if self.config.confirm and not self.config.dry:
            return confirm(self)

    def on_compute_default_resources(self, executable):
        resources = {}
        resources["-p"] = "development"
        resources["-t"] = "2:00:00"
        if (nodes := executable.config.get("nodes", False)) not in [
            None,
            False,
        ]:
            resources["--nodes"] = nodes
        if (ranks := executable.config.get("ranks", 56)) not in [
            None,
            False,
        ]:
            resources["--ntasks-per-node"] = ranks

        return resources

    def __call__(self):
        jobs = {}
        for executable in self.pending_executables:
            # check if job is already launched
            if job := Job.find_by_name(executable.id):
                if job.status in ["PENDING", "RUNNING"]:
                    print(
                        f"{executable.id} is already launched with job_id={job.job_id}, skipping ..."
                    )
                    continue

            if self.config.resume_failed is not True:
                if (
                    executable.executions.filter(
                        lambda x: x.is_incomplete(executable)
                    ).count()
                    > 0
                ):
                    if self.config.resume_failed == "new":
                        executable = executable.new().commit()
                    elif self.config.resume_failed == "skip":
                        continue
                    else:
                        err = f"{executable.module} <{executable.id})> has previously been executed unsuccessfully. Set `resume_failed` to True, 'new' or 'skip' to handle resubmission."
                        if self.config.dry:
                            print(err)
                        else:
                            raise ExecutionFailed(err)

            index = Index.get()
            index_directory = os.path.abspath(index.config.directory)
            source_code = os.getcwd()
            index_exclude = make_relative_if_subpath(index_directory, source_code)

            if self.config.copy_project_source:
                print("Copy project source code ...")
                source_code = self.local_directory(executable.id, "source_code")
                cmd = [
                    "rsync",
                    "-rLptgoD",
                    "--exclude",
                    ".venv",
                    "--exclude",
                    ".git",
                    "--exclude",
                    index_exclude,
                    # "--filter",
                    # "dir-merge,- .gitignore",
                    Project.get().path(""),
                    source_code,
                ]
                print(" ".join(cmd))
                run_and_stream(cmd, check=True)

            script = "#!/usr/bin/env bash\n"

            resources = self.computed_resources(executable)
            mpi = executable.config.get("mpi", self.config.mpi)
            mpi_args = self.config.mpi_args
            ranks = executable.config.get("ranks", None)
            if ranks is not None:
                if mpi_args:
                    mpi_args = mpi_args.replace("{ranks}", str(ranks))
            python = self.config.python or sys.executable

            # usage dependencies
            if "--dependency" not in resources and (dependencies := executable.uses):
                ds = []
                for dependency in dependencies:
                    if dependency.id in jobs:
                        ds.append(str(jobs[dependency.id]))
                    else:
                        if job := Job.find_by_name(dependency.id):
                            ds.append(str(job.job_id))
                if ds:
                    resources["--dependency"] = "afterok:" + (":".join(ds))

            if "--job-name" not in resources:
                resources["--job-name"] = f"{executable.id}"
            if "--output" not in resources:
                resources["--output"] = os.path.abspath(
                    self.local_directory(executable.id, "output.log")
                )
            if "--open-mode" not in resources:
                resources["--open-mode"] = "append"

            sbatch_arguments = []
            for k, v in resources.items():
                if not k.startswith("--"):
                    continue
                line = "#SBATCH " + k
                if v not in [None, True]:
                    line += f"={v}"
                sbatch_arguments.append(line)

            script += "\n".join(sbatch_arguments) + "\n\n"

            local_directory = os.path.abspath(executable.local_directory())

            # copy local
            if resources.get("--partition", "development") != "development":
                script += "sleep $((RANDOM % 300))\n"  # sync-hit prevention

            script += "export I_MPI_EXTRA_FILESYSTEM_FORCE=ufs\n"
            script += "module load cdtools\n"

            script += f"distribute.bash {local_directory}\n"
            # script += f"distribute.bash {os.path.abspath(source_code)}\n"

            # script += "distribute.bash ${SCRATCH}/venvs/.venv\n"
            script += "distribute.bash ${SCRATCH}/venvs/venv.tar\n"
            env_script = chmodx(
                self.save_file(
                    [executable.id, "env_setup.sh"],
                    "#!/usr/bin/env bash\ncd /tmp\n"
                    + "tar -xf venv.tar\n"
                    + (
                        " ".join(
                            [
                                "rsync",
                                "-rLptgoD",
                                "--exclude",
                                ".git",
                                "--exclude",
                                ".venv",
                                "--exclude",
                                index_exclude,
                                os.path.join(source_code, ""),
                                "/tmp/source_code",
                            ]
                        )
                    )
                    + "\n",
                )
            )
            script += (
                f"TACC_TASKS_PER_NODE=1 ibrun -n $SLURM_TACC_NODES {env_script}\n\n"
            )
            script += "cd /tmp/source_code\n"

            script += ". /tmp/.venv/bin/activate\n"

            python = "/tmp/.venv/bin/python3"

            if self.config.preamble:
                script += self.config.preamble

            if mpi:
                if mpi[-1] != " ":
                    mpi += " "
                mpi = mpi + mpi_args
                if mpi[-1] != " ":
                    mpi += " "
                python = mpi + python

            with Index("/tmp"):
                script += executable.dispatch_code(python=python)

            # copy back
            controller_rank = getattr(distwq, "controller_rank")
            if os.getenv("DISTWQ_CONTROLLER_RANK", 0) == "-1":
                controller_rank = int(resources.get("--nodes", 0)) - 1 
            script += "\n\nrm -rf ${SCRATCH}/tmp/" + f"{executable.uuid}_{controller_rank}\n"
            script += f"\ncollect.bash /tmp/{executable.uuid}" + " ${SCRATCH}/tmp\n"
            script += (
                "rsync -a ${SCRATCH}/tmp/"
                + executable.uuid
                + f"_{controller_rank}/ "
                + local_directory + '/'
                + "\n"
            )
            script += "rm -rf ${SCRATCH}/tmp/" + executable.uuid + "\n"

            print(f"Submitting job {executable} with resources: ")
            print(yaml.dump(resources))

            # add debug information
            script += "\n\n"
            script += f"# generated at: {arrow.now()}\n"
            script += f"# {executable.module} <{executable.id}>\n"
            script += f"# {executable.local_directory()}\n\n"
            script += "# " + yaml.dump(executable.version()).replace("\n", "\n# ")
            script += "\n"

            # submit to slurm
            script_file = chmodx(self.save_file([executable.id, "slurm.sh"], script))

            cmd = ["sbatch", script_file]
            print(" ".join(cmd))

            self.save_file(
                [executable.id, "slurm.json"],
                data={
                    "job_id": None,
                    "cmd": sbatch_arguments,
                    "script": script,
                },
            )

            if self.config.dry:
                print("Dry run ... ", executable)
                continue

            try:
                output = subprocess.run(
                    cmd,
                    text=True,
                    check=True,
                    env=os.environ,
                    capture_output=True,
                )
                print(output.stdout)
            except subprocess.CalledProcessError as _ex:
                print(_ex.output)
                raise _ex

            try:
                job_id = int(output.stdout.rsplit(" ", maxsplit=1)[-1])
            except ValueError:
                job_id = False
            print(
                f"{job_id}  named `{resources['--job-name']}` for {executable.local_directory()} (output at {resources['--output']})"
            )

            # update job information
            jobs[executable.id] = job_id
            self.save_file(
                [executable.id, "slurm.json"],
                data={
                    "job_id": job_id,
                    "cmd": sbatch_arguments,
                    "script": script,
                },
            )

            if self.config.throttle > 0 and len(self.pending_executables) > 1:
                time.sleep(self.config.throttle)

    def canonicalize_resources(self, resources):
        if resources is None:
            return {}

        shorthands = {
            "A": "account",
            "B": "extra-node-info",
            "C": "constraint",
            "c": "cpus-per-task",
            "d": "dependency",
            "D": "workdir",
            "e": "error",
            "F": "nodefile",
            "H": "hold",
            "h": "help",
            "I": "immediate",
            "i": "input",
            "J": "job-name",
            "k": "no-kill",
            "L": "licenses",
            "M": "clusters",
            "m": "distribution",
            "N": "nodes",
            "n": "ntasks",
            "O": "overcommit",
            "o": "output",
            "p": "partition",
            "Q": "quiet",
            "s": "share",
            "t": "time",
            "u": "usage",
            "V": "version",
            "v": "verbose",
            "w": "nodelist",
            "x": "exclude",
            "g": "geometry",
            "R": "no-rotate",
        }

        canonicalized = {}
        for k, v in resources.items():
            prefix = ""
            if k.startswith("#"):
                prefix = "#"
                k = k[1:]

            if k.startswith("--"):
                # already correct
                canonicalized[prefix + k] = str(v)
                continue
            if k.startswith("-"):
                # -p => --partition
                try:
                    if len(k) != 2:
                        raise KeyError("Invalid length")
                    canonicalized[prefix + "--" + shorthands[k[1]]] = str(v)
                    continue
                except KeyError as _ex:
                    raise ValueError(f"Invalid short option: {k}") from _ex
            if len(k) == 1:
                # p => --partition
                try:
                    canonicalized[prefix + "--" + shorthands[k]] = str(v)
                    continue
                except KeyError as _ex:
                    raise ValueError(f"Invalid short option: -{k}") from _ex
            else:
                # option => --option
                canonicalized[prefix + "--" + k] = str(v)

        return canonicalized


def yes_or_no() -> bool:
    choice = input().lower()
    return {"": True, "yes": True, "y": True, "no": False, "n": False}[choice]


def confirm(execution: Execution) -> bool:
    sys.stdout.write("\n".join(execution.pending_executables.map(lambda x: x.module)))
    sys.stdout.write(
        f"\nSubmitting {len(execution.pending_executables)} jobs ({len(execution.executables)} total). Proceed? [Y/n]: "
    )
    if yes_or_no():
        sys.stdout.write("yes\n")
        return True
    else:
        sys.stdout.write("no\n")
        return False


class Job:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.details = self._fetch_details()

    def _fetch_details(self) -> dict:
        cmd = ["scontrol", "show", "job", str(self.job_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None

        details = {}
        raw_info = result.stdout.split()
        for item in raw_info:
            if "=" in item:
                key, value = item.split("=", 1)
                details[key] = value
        return details

    @classmethod
    def find_by_name(cls, job_name: str) -> Optional["Job"]:
        cmd = ["squeue", "--name", job_name, "--noheader", "--format=%i"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return cls(result.stdout.strip())

        return None

    @property
    def status(
        self,
    ) -> Literal[
        "",
        "PENDING",
        "RUNNING",
        "SUSPENDED",
        "CANCELLED",
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "PREEMPTED",
    ]:
        return self.details.get("JobState", "")

    @property
    def info(self) -> dict:
        return self.details

    def cancel(self) -> bool:
        cmd = ["scancel", str(self.job_id)]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
