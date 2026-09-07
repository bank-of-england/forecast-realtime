import re
from pathlib import Path

import pytest

import forecast_realtime as rt

README_PATH = Path(__file__).parents[1] / "README.md"
README = README_PATH.read_text(encoding="utf-8")
PYTHON_EXAMPLES = re.findall(r"```python\n(.*?)```", README, flags=re.DOTALL)


def test_readme_has_expected_python_examples():
    assert len(PYTHON_EXAMPLES) == 4


@pytest.mark.parametrize(
    "block_count",
    range(1, 5),
    ids=["import-data", "existing-model", "custom-model", "forecast"],
)
def test_readme_python_walkthrough_runs(block_count):
    namespace = {"__name__": "__readme_example__"}
    for index, source in enumerate(PYTHON_EXAMPLES[:block_count], start=1):
        assert "load_fer" not in source
        filename = f"{README_PATH} block {index}"
        exec(compile(source, filename, "exec"), namespace)


def test_readme_synthetic_data_preview_is_current():
    match = re.search(
        r"The first five generated outturn rows are:\n\n```text\n(.*?)\n```",
        README,
        flags=re.DOTALL,
    )
    assert match is not None

    sample_data = rt.generate_synthetic_data(
        N=2,
        first_period="2015-01-31",
        endpoint="2024-12-31",
    )
    displayed_rows = [line.split() for line in match.group(1).splitlines()]
    expected_rows = [
        line.split() for line in sample_data.head().to_string(index=False).splitlines()
    ]
    assert displayed_rows == expected_rows
