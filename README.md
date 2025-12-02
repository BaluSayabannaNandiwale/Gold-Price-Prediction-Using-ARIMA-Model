# Gold Price Prediction

> **Gold Price Analysis & Prediction** — Time series forecasting, strategy backtesting, and sentiment analysis on historical gold prices (Jan 19, 2014 → Jan 22, 2024).

---

## Project Overview

This repository contains a complete data science pipeline for exploring, modeling, and deriving trading insights from historical gold prices. It combines classical time-series analysis (ARIMA), simple trading strategies (SMA), and market-sentiment analysis derived from news headlines to help researchers, analysts, and data enthusiasts better understand gold market behavior.

---

## Table of Contents

1. [Dataset(s)](#datasets)
2. [Project Objectives](#objectives)
3. [Repository Structure](#structure)
4. [Quick Start / Requirements](#quick-start)
5. [Data Preparation & EDA](#eda)
6. [Modeling](#modeling)
7. [Trading Strategy & Backtesting](#trading)
8. [Market Sentiment Analysis](#sentiment)
9. [Results & Metrics](#results)
10. [How to reproduce](#reproduce)
11. [Limitations & Considerations](#limitations)
12. [Contributing & License](#contributing)

---

## <a name="datasets"></a>Datasets

**Dataset 1 — Gold Price Data**

* Source: Nasdaq (as used in the notebook)
* Columns: `Date`, `Close`, `Volume`, `Open`, `High`, `Low`
* Period covered: 2014-01-19 → 2024-01-22

**Dataset 2 — Market Sentiment Data**

* Columns: `Dates`, `URL`, `News`, `Price Direction Up`, `Price Direction Constant`, `Price Direction Down`, `Asset Comparison`, `Past Information`, `Future Information`, `Price Sentiment`
* Contains labeled news headlines & derived sentiment fields used to study correlations with price movements.

> NOTE: Keep raw data files in `/data/raw/` and processed files in `/data/processed/`.

---

## <a name="objectives"></a>Project Objectives

* Perform time-series analysis to identify trend, seasonality and residual behavior.
* Build forecasting models (ARIMA baseline) and evaluate performance.
* Design and backtest a simple SMA-based trading strategy and compute performance metrics (total return, Sharpe ratio).
* Merge sentiment features with price data to evaluate how headlines correlate with short/long-term price moves.
* Document findings and provide reproducible notebooks/scripts.

---

## <a name="structure"></a>Repository Structure

```
README.md
data/
  raw/
  processed/
notebooks/
  01_EDA.ipynb
  02_Modeling_ARIMA.ipynb
  03_Trading_Backtest.ipynb
  04_Sentiment_Analysis.ipynb
scripts/
  preprocess.py
  train_arima.py
  backtest_sma.py
requirements.txt
LICENSE
```

---

## <a name="quick-start"></a>Quick Start / Requirements

**Install dependencies (recommended: create a venv or conda env)**

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

**requirements.txt (example)**

```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
python-dateutil
jupyterlab
```

---

## <a name="eda"></a>Data Preparation & EDA

Main steps implemented in the notebooks:

1. Load CSV files and drop unnecessary columns (for example, `Unnamed: 0`).
2. Convert `Date` / `Dates` column to `datetime` and set as index.
3. Basic checks: `df.info()`, `df.describe()`, null checks.
4. Visualizations: closing price time series plot, seasonal decomposition (period=365), moving averages.

**Example snippet used in notebooks**

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('data/raw/goldstock.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# seasonal decomposition
decomp = seasonal_decompose(df['Close'], model='multiplicative', period=365)
decomp.plot();
```

---

## <a name="modeling"></a>Modeling (ARIMA baseline)

* Train / test split: 80% train, 20% test.
* Baseline model: `ARIMA(order=(5,1,0))` on training set.
* Forecast on test set and evaluate using RMSE.

**Notes & tips**

* Use AIC/BIC and residual diagnostics to tune ARIMA orders.
* Consider seasonal ARIMA (SARIMA) or modern alternatives (Prophet, ETS, LSTM, Transformer-based time series models) for better performance.

---

## <a name="trading"></a>Trading Strategy & Backtesting

**Simple Moving Average (SMA) Strategy implemented**

* Compute `SMA_50` and `SMA_200`.
* Signal: go long when SMA_50 > SMA_200, exit (or go flat) otherwise.
* Compute position changes, daily returns, strategy returns and cumulative returns.

**Performance metrics**

* Total Return
* Sharpe Ratio (annualized using 252 trading days)

**Caveats**

* No transaction costs, slippage or position sizing applied in baseline backtest.
* Consider adding stop-loss, take-profit, and realistic execution assumptions for production backtests.

---

## <a name="sentiment"></a>Market Sentiment Analysis

* Parse news `Dates` into datetime and set as index.
* Merge sentiment headlines with price data on dates to analyze contemporaneous relationships.
* Visualize price vs. sentiment (e.g., overlay daily price changes with sentiment labels or aggregated sentiment scores).

**Ideas for extension**

* Use NLP models or transformers to compute continuous sentiment scores instead of discrete labels.
* Lagged analysis to check whether sentiment leads price moves (Granger causality tests).

---

## <a name="results"></a>Results & Example Metrics

* Example ARIMA RMSE (from notebook): `~135.88` (units = price points)
* Example SMA strategy results (from notebook):

  * Total Return: `-0.0387` (about `-3.87%` over test period)
  * Sharpe Ratio: `-0.466`

> These are baseline results and should be used as starting points rather than final investment guidance.

---

## <a name="reproduce"></a>How to run (reproducible steps)

1. Place raw CSV files in `data/raw/`.
2. Create & activate virtual environment and install requirements.
3. Run preprocessing:

```bash
python scripts/preprocess.py --input data/raw/goldstock.csv --sentiment data/raw/gold-dataset-sinha-khandait.csv --output data/processed/
```

4. Run notebooks interactively or execute modeling scripts:

```bash
python scripts/train_arima.py --data data/processed/gold_close.csv
python scripts/backtest_sma.py --data data/processed/gold_with_ma.csv
```

---

## <a name="limitations"></a>Limitations & Considerations

* Historical performance is not a guarantee of future returns.
* Model performance may change when retrained with newer data.
* Sentiment labeling in the provided sentiment dataset may be noisy or coarse-grained.
* Trading simulations exclude real-world factors like transaction costs, market impact, taxes.

---

## <a name="contributing"></a>Contributing & License

Contributions are welcome. Please open an issue or a pull request for:

* Improved models (Prophet, LSTM, transformers)
* Extended backtests with transaction costs and position sizing
* Better sentiment pipelines (fine-tuned NLP models)

**License**: MIT

---

## Acknowledgements

* Dataset providers (Nasdaq, original sentiment dataset authors)
* Python open-source community and libraries used in the analysis

---
