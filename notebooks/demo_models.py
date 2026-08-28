"""Run the compact forecast model demonstration in marimo."""

import marimo

__generated_with = "0.24.0"

app = marimo.App(width="full")


@app.cell
def __():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from news_decomp import NewsData

    from examples.demo_models import run_demo

    return NewsData, run_demo


@app.cell
def _(NewsData, run_demo):
    demo = run_demo(
        N_vintages=6,
        decomp=True,
        reconstruct_levels=False,
    )

    news_data = NewsData(demo.decompositions)
    news_data.report(
        variable="quarterly_1",
        source="Ridge",
    )
    return demo, news_data


if __name__ == "__main__":
    app.run()
