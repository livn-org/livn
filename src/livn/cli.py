import sys


def main(args: list | None = None):
    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        print("Usage:")
        print("livn [action] ...")
        print("      version -- display the package version")
        print("      systems -- systems cli (if installed)")
        return 0

    action, args = args[0], args[1:]

    if action == "version":
        import livn

        version = livn.get_version()
        print(version)
        return 0

    if action == "systems":
        sys.path.insert(0, "")
        try:
            from systems.cli import main as systems_cli
        except ModuleNotFoundError:
            print("Module not found. Is systems installed? (uv sync --package systems)")
            return 0

        return systems_cli(args)

    print("Invalid argument")
    return 128
