import ast
import importlib
import pathlib
import py_compile

import pytest

EXAMPLES = sorted(
    p
    for p in (pathlib.Path(__file__).parents[2] / "examples").glob("*.py")
    if not p.name.startswith("_")
)


def _example_ids():
    return [p.name for p in EXAMPLES]


@pytest.mark.skipif(not EXAMPLES, reason="no examples/ directory")
@pytest.mark.parametrize("example", EXAMPLES, ids=_example_ids())
def test_an_example_compiles(example, tmp_path):
    py_compile.compile(
        str(example), cfile=str(tmp_path / f"{example.stem}.pyc"), doraise=True
    )


def _livn_imports(source: str):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "livn" or node.module.startswith("livn.")
            ):
                for alias in node.names:
                    yield node.module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "livn" or alias.name.startswith("livn."):
                    yield alias.name, None


@pytest.mark.skipif(not EXAMPLES, reason="no examples/ directory")
@pytest.mark.parametrize("example", EXAMPLES, ids=_example_ids())
def test_an_example_only_names_api_that_exists(example):
    imports = list(_livn_imports(example.read_text()))
    if not imports:
        pytest.skip(f"{example.name} imports nothing from livn")

    for module_name, attribute in imports:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            pytest.skip(f"{module_name} needs an extra that is not installed: {e}")

        if attribute is None:
            continue

        assert hasattr(module, attribute), (
            f"{example.name} imports {attribute!r} from {module_name}, "
            "which no longer has it"
        )
