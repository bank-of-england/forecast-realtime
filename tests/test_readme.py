import re
from pathlib import Path

from tests.realtime_fixtures import generate_synthetic_data

README_PATH = Path(__file__).parents[1] / "README.md"
README = README_PATH.read_text(encoding="utf-8")
PYTHON_EXAMPLES = re.findall(r"```python\n(.*?)```", README, flags=re.DOTALL)


def test_readme_has_expected_python_examples():
    assert len(PYTHON_EXAMPLES) == 4


def test_readme_python_walkthrough_runs(monkeypatch):
    monkeypatch.setattr(
        "forecast_realtime.generate_synthetic_data", generate_synthetic_data
    )
    namespace = {"__name__": "__readme_example__"}
    for index, source in enumerate(PYTHON_EXAMPLES, start=1):
        assert "load_fer" not in source
        filename = f"{README_PATH} block {index}"
        exec(compile(source, filename, "exec"), namespace)

    match = re.search(
        r"The first five generated outturn rows are:\n\n```text\n(.*?)\n```",
        README,
        flags=re.DOTALL,
    )
    assert match is not None

    sample_data = namespace["sample_data"]
    displayed_rows = [line.split() for line in match.group(1).splitlines()]
    expected_rows = [
        line.split() for line in sample_data.head().to_string(index=False).splitlines()
    ]
    assert displayed_rows == expected_rows
