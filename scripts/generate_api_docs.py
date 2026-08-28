import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "forecast_realtime"
API_DOC = PROJECT_ROOT / "docs" / "api.md"


def read_exports(path: Path) -> list[str]:
    """Read a module's literal ``__all__`` declaration without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in targets
        ):
            exports = ast.literal_eval(node.value)
            if not isinstance(exports, list) or not all(
                isinstance(name, str) for name in exports
            ):
                raise ValueError(f"{path}: __all__ must be a literal list of strings")
            return exports
    raise ValueError(f"{path}: no literal __all__ declaration found")


def read_literal(path: Path, variable_name: str) -> object:
    """Read a literal module-level variable without importing its module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path}: no literal {variable_name} declaration found")


def public_objects() -> list[tuple[str, list[str]]]:
    """Return object paths derived only from package ``__all__`` declarations."""
    root_exports = read_exports(PACKAGE_ROOT / "__init__.py")
    root_objects = [
        f"forecast_realtime.{name}" for name in root_exports if name != "models"
    ]

    models_init = PACKAGE_ROOT / "models" / "__init__.py"
    model_exports = read_exports(models_init)
    model_specs = read_literal(models_init, "_MODEL_SPECS")
    if not isinstance(model_specs, dict) or set(model_specs) != set(model_exports):
        raise ValueError("models._MODEL_SPECS must map every name in models.__all__")

    model_objects = []
    for name in model_exports:
        spec = model_specs[name]
        if not isinstance(spec, tuple) or not spec or not isinstance(spec[0], str):
            raise ValueError(f"models._MODEL_SPECS[{name!r}] has an invalid module")
        model_objects.append(f"forecast_realtime.models.{spec[0]}.{name}")

    return [("Core API", root_objects), ("Built-in models", model_objects)]


def render_api_page(sections: list[tuple[str, list[str]]]) -> str:
    """Render the Markdown manifest consumed by mkdocstrings."""
    lines = [
        "# API Reference",
        "",
        "`scripts/generate_api_docs.py` builds this page from the package's",
        "`__all__` declarations. Zensical renders each public object from the",
        "current source.",
        "",
    ]
    for section, objects in sections:
        lines.extend((f"## {section}", ""))
        for object_path in objects:
            lines.extend(
                (
                    f"::: {object_path}",
                    "    options:",
                    "      show_source: false",
                    "      show_root_heading: true",
                    "",
                )
            )
    return "\n".join(lines)


def main() -> None:
    """Update the generated API documentation manifest."""
    API_DOC.write_text(render_api_page(public_objects()), encoding="utf-8")


if __name__ == "__main__":
    main()
