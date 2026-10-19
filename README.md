# Risk Pillar Predictors

A web app for testing how well machine-learning models predict country risk-pillar scores.

**Live app:** https://risk-pillar-predictors.netlify.app

## What it does

1. **Upload** a CSV of monthly risk-pillar scores by country.
2. **Pick a pillar** and a test period (last 6, 12 or 24 months).
3. **Click any of 10 models** to train it and see its performance on 5 metrics, plus a chart of actual vs predicted values (all-country average or a single country).
4. **Compare** the selected models side by side, with the best model highlighted for each metric. The comparison can be downloaded as a CSV.

Everything runs in the browser using [Pyodide](https://pyodide.org) (Python compiled to WebAssembly) and scikit-learn. Uploaded data is never sent to a server.

## Data format

One row per country per month. Columns: `Country`, `Period` (e.g. `2025-01`), then one numeric column per risk pillar.

```
Country,Period,InflationRisk,FXExternalRisk,DebtFiscalRisk,...
Algeria,2015-01,5.09,0.87,7.95,...
Algeria,2015-02,5.22,1.66,8.91,...
```

Each country needs at least 13 months of history before the test period.

## Method

Each month is predicted from that country's own recent history:

- values 1, 2, 3, 6 and 12 months back
- 3, 6 and 12-month averages and 6-month volatility
- calendar month and country

Models train on every month before the test period and are scored on the test period (one-step-ahead predictions).

### Models

Linear Regression, Ridge, Lasso, Elastic Net, Huber, K-Nearest Neighbours, Decision Tree, Random Forest, Extra Trees, Gradient Boosting.

### Metrics

| Metric | Meaning | Better |
|---|---|---|
| MAE | Average absolute error, in pillar units | Lower |
| RMSE | Root mean squared error, so large misses count more | Lower |
| R² | Share of variation explained | Higher |
| sMAPE | Symmetric mean absolute percentage error (works when values are zero) | Lower |
| MASE | MAE divided by the MAE of a "same as last month" forecast; below 1 beats it | Lower |

## Files

- `index.html`: the interface
- `worker.js`: loads Python in a background thread
- `ml.py`: feature engineering, models and metrics (also runs in regular Python)

## Run locally

Any static file server works, for example `python3 -m http.server`, then open http://localhost:8000.
