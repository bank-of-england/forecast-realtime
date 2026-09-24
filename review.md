# Review: `feature/density_forecasts` → `dev`

## Fix before merge

1. **A failure in one model aborts the whole quantile run.** In point mode, a model that fails is caught and the other models still produce forecasts. In quantile mode that protection is switched off (`real_time_model.py:628`). I suspect this was done because a failed task carries no quantile table, so combining the results would fail with "No objects to concatenate". The fix is to skip empty results when combining, not to drop the protection.

2. **Quantiles are passed back in the wrong slot.** `real_time_model.py:45-51` puts the quantile table into the slot meant for point forecasts, reads it back by position (`result[0]`, `result[2]`), and hard-codes `all_vintages_skipped=False` when combining. Nothing depends on the point slot, because quantile rows already carry a `quantile` column. It would be simpler to keep one table and split it only when saving to `self.quantiles`, which would also remove both special cases.

## Should fix (clarity and correctness)

3. **Linear regression quantiles are simulated when they can be calculated exactly.** `linear_regression.py:311` draws 10,000 random samples from a normal distribution with a known mean and spread. The same answer comes directly from `mean + se * stats.t.ppf(q, df)`, or `norm.ppf(q)` if a normal distribution is intended. That would remove the `n_samples` and `random_state` options, which aren't documented anyway, and the check that `n_samples` is even. The test at `test_quantile_contract.py:253-263` repeats the same simulation, so it checks the code against itself rather than against the statistics.

4. **Restrict regression density forecasts to unpenalised regressions for now.** Use the residual standard error to model forecast noise; this omits coefficient uncertainty. Revisit resampling and parameter uncertainty in later work.

7. **Simplify the tree's frequency resolution.** `RealTimeModel` supplies the frequency at fit time, so a missing value on that path is a bug. Standalone `ForecastTree.fit()` permits an omitted frequency: resolve it once from input dates or metadata, then raise a clear error if it is ambiguous. Remove the extra fallback chain in `forecast_tree.py:631-649`.

9. **`forecast()` passes `quantiles` in a roundabout way.** `forecast_model.py:1039` adds it to `kwargs` only when it isn't `False`. Passing `quantiles=quantiles` straight to `_predict_data` is simpler, since that function already defaults it to `False`.

## Suggested squash-merge message

```
feat!: add quantile forecasts and standardise long forecast results

Add native-metric predictive quantiles for OLS-family and BVAR models
through `quantiles=` on ForecastModel.forecast() and RealTimeModel.forecast().
Unify the fitted configuration and design specification so that fitting and
forecasting share a single frozen DesignSpec, and move result validation into
ForecastResult.

BREAKING CHANGE: ForecastModel.forecast() returns a long table with date,
variable and value columns instead of a wide frame. ForecastModel.predict()
is removed; use forecast(context=...). The forecast_realtime.data_transformation
module is now private. Fitted mirror attributes such as X_names, dummies,
y_estimation and _forecast_frequency are removed.
```