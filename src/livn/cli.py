import os
import sys
import tomllib


def _find_packages():
    root = os.getcwd()
    root_toml = os.path.join(root, "pyproject.toml")
    if not os.path.isfile(root_toml):
        return []

    with open(root_toml, "rb") as f:
        root_data = tomllib.load(f)

    members = (
        root_data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    )
    packages = []
    for member in sorted(members):
        member_toml = os.path.join(root, member, "pyproject.toml")
        if not os.path.isfile(member_toml):
            continue
        with open(member_toml, "rb") as f:
            member_data = tomllib.load(f)
        desc = member_data.get("project", {}).get("description", "")
        packages.append((member, desc))

    return packages


def _print_help():
    packages = _find_packages()
    width = max((len(name) for name, _ in packages), default=0)
    width = max(width, len("version")) + 1

    print("Usage:")
    print("livn [action] ...")
    print(f"      {'version':<{width}} - Display the package version")
    for name, desc in packages:
        if desc:
            print(f"      {name:<{width}} - {desc}")
        else:
            print(f"      {name}")


def main(args: list | None = None):
    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        _print_help()
        return 0

    action, args = args[0], args[1:]

    if action == "version":
        import livn

        version = livn.get_version()
        print(version)
        return 0

    subproject_dir = os.path.join(os.getcwd(), action)
    if os.path.isfile(os.path.join(subproject_dir, "pyproject.toml")):
        sys.path.insert(0, "")
        try:
            from machinable.cli import main as machinable_cli
            from machinable.project import Project
        except ModuleNotFoundError:
            print(
                f"Module not found. Is {action} installed? (uv sync --package {action})"
            )
            return 0

        if len(args) == 0:
            print("Usage:")
            print(f"livn {action} [module] ...")
            return 0

        with Project(subproject_dir):
            return machinable_cli(["get"] + args)

    _print_help()
    return 128
