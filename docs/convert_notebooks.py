"""Convert marimo apps in notebooks/ to Markdown for Zensical.

Run this script whenever a marimo app changes:
    python docs/convert_notebooks.py

Check that the committed Markdown exports are current:
    python docs/convert_notebooks.py --check

The script uses marimo to write one .md file for each .py app in notebooks/.
"""

import argparse
import subprocess
import sys
from pathlib import Path

APPS_DIR = Path(__file__).parent.parent / "notebooks"
OUTPUT_DIR = Path(__file__).parent / "notebooks"
REPO_ROOT = APPS_DIR.parent


def convert(app: Path) -> None:
    """Export one marimo app to Markdown."""
    print(f"Converting {app.name} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "md",
            str(app),
            "--output",
            str(OUTPUT_DIR / f"{app.stem}.md"),
            "--force",
        ],
        check=True,
    )


def main() -> int:
    """Convert apps and optionally check that committed exports are current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when conversion changes a committed Markdown export",
    )
    args = parser.parse_args()

    apps = sorted(APPS_DIR.glob("*.py"))
    if not apps:
        print("No marimo apps found in", APPS_DIR)
        return 0

    for app in apps:
        convert(app)

    if args.check:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--exit-code",
                "--",
                str(OUTPUT_DIR.relative_to(REPO_ROOT)),
            ],
            check=False,
            cwd=REPO_ROOT,
        )
        if result.returncode:
            print(
                "Notebook Markdown exports are stale. "
                "Run `python docs/convert_notebooks.py` and commit the changes."
            )
            return result.returncode

    print("\nDone. Re-run this script whenever a marimo app changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
