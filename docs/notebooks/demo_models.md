---
title: Demo Models
marimo-version: 0.24.0
width: full
header: |-
  """Run the compact forecast model demonstration in marimo."""
---

```python {.marimo}
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news_decomp import NewsData

from examples.demo_models import run_demo
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