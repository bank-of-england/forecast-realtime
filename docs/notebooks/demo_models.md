---
title: Demo Models
marimo-version: 0.24.0
width: full
header: |-
  """Run the compact forecast model demonstration in marimo."""
---

```python {.marimo}
from news_decomp import NewsData

from forecast_realtime.examples.demo_models import run_demo
```

```python {.marimo}
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
```