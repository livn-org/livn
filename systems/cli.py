import os
import sys


def main(args: list | None = None):
    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        print("Usage:")
        print("livn systems [module] ...")
        return 0

    from machinable.cli import main as machinable_cli
    from machinable.project import Project

    with Project(os.path.dirname(__file__)):
        return machinable_cli(["get"] + args)
