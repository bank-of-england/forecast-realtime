# `ForecastTree`

`ForecastTree` combines several forecast models into a single model. You describe
the combination as a **tree**: leaves are individual `ForecastModel` instances,
and each internal node transforms its children with a `transform`. The top-most
(root) node's output is the tree's final forecast.

A `ForecastTree` *is itself* a `ForecastModel`, so it fits the usual contract:

```python
tree.fit(y, X=X, **kwargs)
tree.forecast(steps=4, X=X, y=y)
```

---

## The shape of a tree

```
                      gdp_tree (WeightedSum)
                 /                            \
     expenditure (WeightedSum)          income (OLS)
        /            \                     /            \
 consumption (OLS)  investment (OLS)    wage (OLS)   profit (OLS)
   data                data                  data           data
```

- **Leaves** are bare `ForecastModel` instances (e.g. `ForecastOLS`, `Ridge`,`MIDAS`). A leaf is identified by its `label`.
- **Internal nodes** are `TreeNode` instances. A node is identified by its
  `name` and holds a `transform` plus a `list` of `children`.
- The root `TreeNode` produces the value returned by `ForecastTree.forecast()`.

Combination happens **bottom-up**: each node transforms *only its direct children*,
and a node's result then feeds its parent.

---

## The two building blocks

### `TreeNode(transform, children, name=None, target=None)`

One node of the tree.

| Parameter   | Meaning                                                                                     |
| ----------- | ------------------------------------------------------------------------------------------- |
| `transform` | How this node transforms its direct children — a callable **or** a `ForecastModel` (see below). |
| `children`  | A non-empty `list` of children, each a `ForecastModel` (leaf) or a nested `TreeNode`.        |
| `name`      | Identifier for the node. Defaults to `"node"` when `None`.                                   |
| `target`    | The `y` column a **callable** `transform` reduces its children to; auto-resolved when `y` has one column. Must not be set on a `ForecastModel` transform. |

### `ForecastTree(spec, label=None)`

The model that drives the whole tree.

| Parameter | Meaning                                                                                                     |
| --------- | ---------------------------------------------------------------------------------------------------------- |
| `spec`    | The root `TreeNode`.                                                                                        |
| `label`   | Label for this model instance (passed through to `ForecastModel`).                                          |
| `data_transformation` | Optional model-owned input transformation fallback for leaves and nested trees. |

Both are importable from the package root:

```python
from forecast_realtime import ForecastTree, TreeNode
```

---

## Two kinds of `transform`

A node's `transform` is either a plain function or a `ForecastModel`. A tree can
contain both kinds of transform.

### 1. Callable transform — a fixed rule

A function `dict[str, pd.DataFrame] -> pd.DataFrame`. It receives one entry per
child (keyed by leaf `label` / node `name`), each a **single-column** DataFrame
holding that child's forecast of the resolved `target`. It returns the combined
single-column forecast. Use this for fixed rules such as a weighted sum (weights
need not sum to one):

```python
class WeightedSum:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, components):
        return sum(components[name] * w for name, w in self.weights.items())


expenditure = TreeNode(
    name="expenditure",
    children=[consumption_model, investment_model],
    transform=WeightedSum({"consumption_model": 1.1, "investment_model": 0.95}),
    target="gdp",
)
```

Because callable transforms expect a single target-named column per child, each
callable node's `target` must be resolvable — either set on the `TreeNode` or
inferred when `y` has exactly one column.

### 2. `ForecastModel` transform — a learned combination

Any model instance. The children's **raw** forecasts are stacked into a design
matrix (one column per child) and the transform is `fit`/`forecast`ed on them,
exactly like a leaf. Its own `formula`/target selection decides what it targets:

```python
income_stacker = ForecastOLS(
    label="income_stacker", formula="gdp ~ wage_model + profit_model"
)

income = TreeNode(
    name="income",
    children=[wage_model, profit_model],
    transform=income_stacker,
)
```

Because a `ForecastModel` transform picks its own target, it takes no `target`
(setting one on such a node raises); with an all-`ForecastModel` tree no target
needs resolving anywhere, even with multi-column `y`.

---

## Full example

A synthetic economy where GDP is combined from an **expenditure** side (a fixed
weighted sum of consumption and investment) and an **income** side (an OLS
stacking of wage and profit models), the two then averaged at the root. See
[examples/forecast_tree.py](../examples/forecast_tree.py) for the
complete runnable script.

```python
from forecast_realtime import ForecastTree, TreeNode
from forecast_realtime.models.ols import ForecastOLS

# --- Leaves: one OLS per component ---------------------------------------
consumption_model = ForecastOLS(
    label="consumption_model", formula="consumption ~ consumption_indicator"
)
investment_model = ForecastOLS(
    label="investment_model", formula="investment ~ investment_indicator"
)
wage_model = ForecastOLS(label="wage_model", formula="wages ~ wage_indicator")
profit_model = ForecastOLS(label="profit_model", formula="profits ~ profit_indicator")

# --- Expenditure side: a fixed weighted sum (callable transform) ---------
expenditure = TreeNode(
    name="expenditure",
    children=[consumption_model, investment_model],
    transform=WeightedSum({"consumption_model": 1.1, "investment_model": 0.95}),
    target="gdp",
)

# --- Income side: a learned OLS stacking (ForecastModel transform) -------
income = TreeNode(
    name="income",
    children=[wage_model, profit_model],
    transform=ForecastOLS(
        label="income_stacker", formula="gdp ~ wage_model + profit_model"
    ),
)

# --- Root: average the two GDP measures ----------------------------------
gdp_tree = TreeNode(
    name="gdp_tree",
    children=[expenditure, income],
    transform=WeightedSum({"expenditure": 0.5, "income": 0.5}),
    target="gdp",
)

tree = ForecastTree(spec=gdp_tree, label="gdp_expenditure_income_tree")
tree.fit(y_train, X=X)
forecast = tree.forecast(steps=4, X=X)
```

### Root output metric

`ForecastTree` treats a callable root as returning levels, because the
callable's result is a new combined series. When the root transform is a
`ForecastModel`, the tree inherits that model's fitted target metric. A model
root therefore declares its metric through its `data_transformation` mapping,
not through a separate output setting.

---

## Nowcast, then condition: MIDAS → BVAR

A common shape is a two-stage tree: nowcast every variable from high-frequency
indicators, then let a BVAR forecast the whole system *conditional* on those
nowcasts.

A plain `ForecastBVAR` cannot be used as the transform here. A `ForecastModel`
transform receives its children's forecasts as `X`, but a BVAR takes its
conditioning paths through `y`. A small `ForecastBVAR` subclass bridges the two:
it maps the children's forecasts onto the fitted variables **by name** and
passes them on as constraint paths, so **each leaf must be labelled after the
variable it nowcasts**.

`ConditionalBVAR` is not part of the package — it lives in
[examples/midas_bvar_tree.py](../examples/midas_bvar_tree.py) as a worked example
of writing a custom transform, alongside the runnable tree below.

```python
from examples.midas_bvar_tree import ConditionalBVAR  # example code

from forecast_realtime import ForecastTree, TreeNode
from forecast_realtime.models.midas import ForecastMIDAS

# --- Leaves: one MIDAS nowcast per variable in y -------------------------
nowcasts = [
    ForecastMIDAS(label="gdp", formula="gdp ~ gdp_indicator", horizons=[0, 1, 2, 3]),
    ForecastMIDAS(label="cpi", formula="cpi ~ cpi_indicator", horizons=[0, 1, 2, 3]),
]

# --- Root: a BVAR conditioned on those nowcasts --------------------------
spec = TreeNode(
    name="bvar",
    children=nowcasts,
    transform=ConditionalBVAR(n_lags=2, conditioning_steps=1),
)

tree = ForecastTree(spec=spec, label="midas_bvar")
tree.fit(y, X=X)  # y: quarterly variables; X: monthly indicators
forecast = tree.forecast(steps=4, X=X)
```

Alignment is by **date**, so MIDAS's model-anchored nowcast dates line up with
the BVAR's forecast dates on their own. Anything the nowcasts do not cover —
an unmatched variable, or a horizon beyond them — stays `NaN` and is left
unconstrained. `conditioning_steps` narrows this further, constraining only the
first *n* horizons (above, the nowcast quarter only).

The paths the BVAR actually used are kept for inspection:

```python
tree.spec.transform.conditioning_  # (steps × n_variables), NaN = unconstrained
```

The example needs both the `bvar` and `nowcast_midas` extras.

---

## Inspecting intermediate results

After `forecast()`, the per-source outputs are kept on the instance:

```python
tree.node_forecasts_["expenditure"]  # the expenditure measure
tree.node_forecasts_["income"]  # the income measure
tree.leaf_forecasts_["wage_model"]  # an individual leaf's forecast
```

- `leaf_forecasts_` — keyed by leaf `label`.
- `node_forecasts_` — keyed by node `name`.

Both hold each source's raw/natural forecast: a callable node's entry is its
single target-column output, while a `ForecastModel` leaf or node keeps its own
natural (possibly multi-column) forecast.

---

## What happens on fit / forecast

1. Every **unique** leaf is fitted once on the shared `y`/`X`; `**kwargs`
   (e.g. `y_lags`, `X_lags`, `dummies`) are forwarded to each leaf's `fit()`.
2. Nodes are evaluated **bottom-up** (post-order). Callable transforms are
   applied to their children's single target-column forecasts; `ForecastModel`
  transforms are fitted on their children's stacked raw forecasts.
3. The root node's combined forecast is returned.

### Runtime input rules for node transforms

- `y_lags`, `X_lags`, and `dummies` are **leaf structural controls**: they are
  forwarded to leaf `fit()` calls, where lagged/dummy design matrices are built.
- `ForecastModel` node transforms are fitted on an already-stacked child design
  matrix, so those structural controls are not forwarded to node-transform
  `fit()`.
- Other fit-time custom kwargs are forwarded to `ForecastModel` node-transform
  `fit()`.
- At forecast time, `ForecastModel` node transforms receive the forecast-time
  conditioning `y` and forecast custom kwargs, alongside the stacked child
  forecasts passed as `X`; leaf structural controls (`y_lags`, `X_lags`,
  `dummies`) are excluded.

---

## Rules the tree enforces

- `children` must be a **non-empty `list`** (not a tuple or other sequence).
- Direct children of a single node must have **unique names**.
- Two *distinct* objects anywhere in the tree may not **share a name**.
- The tree may not contain a **cycle**.
- `transform` must be **callable or a `ForecastModel`**; each child must be a
  `ForecastModel` or `TreeNode`.

The same object may be reused across branches (DAG-style); traversal
deduplicates by object identity, so a shared leaf is fitted and forecast only
once.
