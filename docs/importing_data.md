# Importing data

`RealTimeModel` accepts `forecast_evaluation.ForecastData` and its subclasses.
Choose the container according to how often forecasts are issued:

- Use `NowcastData` when forecasts may be issued several times within one
  target period, such as weekly or monthly vintages for quarterly GDP.
- Use `ForecastData` when each target period has one forecast date, such as one quarterly forecast for each quarter.

The generated real-time data contains monthly vintages, so this example uses
`NowcastData`:

```python
import forecast_evaluation as fe
import forecast_realtime as rt

sample_data = rt.generate_synthetic_data(
    N=2,
    first_period="2015-01-31",
    endpoint="2024-12-31",
)
print(sample_data.head().to_string(index=False))

forecast_data = fe.NowcastData(outturns_data=sample_data)
rt_model = rt.RealTimeModel(data=forecast_data, models=models)
```

The generated data begins:

```text
	date frequency  variable      value vintage_date metric
2015-01-31         M monthly_1 101.577869   2024-01-31 levels
2015-01-31         M monthly_2 101.256092   2024-01-31 levels
2015-02-28         M monthly_1 101.293703   2024-01-31 levels
2015-02-28         M monthly_1  -0.002798   2024-01-31    pop
2015-02-28         M monthly_2  98.456195   2024-01-31 levels
```

## Outturn schema

The outturn DataFrame uses long-form rows with these columns:

| Column | Meaning |
| --- | --- |
| `date` | End of the period measured |
| `frequency` | `M` for monthly or `Q` for quarterly |
| `variable` | Series name |
| `value` | Observed value |
| `vintage_date` | Date when the value became available |
| `metric` | `levels`, `pop`, or `yoy` |

`date` identifies the period measured; `vintage_date` identifies when the
value became available. Use another row with a later `vintage_date` to record a
revision. At each forecast vintage, `RealTimeModel` uses the latest value
available by that date. Complete snapshots are not required.

Use `levels` for raw values, `pop` for period-on-period percentage growth, and
`yoy` for year-on-year percentage growth. See [Usage](usage.md) for input
transformations and forecast execution. `ForecastData` validates the outturn
rows when it is created.