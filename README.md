# Stock Behavior Prediction

Every trading day, after the close, a LightGBM model estimates for each stock in the S&P 500 **on that day** the probability that it goes up the next day. A long-only portfolio buys, at the next open, the stocks whose probability is above a confidence threshold, weighted by that probability, and sells them at the following open; if none qualifies, it holds cash at the 13-week T-bill rate. The question it answers: can next-day predictions from price history alone beat the S&P 500 after costs?

## How it works

### 1. One decision, step by step

```
close of t                open of t+1                     open of t+2
    │                          │                               │
    ├─ compute features        ├─ buy the selected stocks      ├─ sell them and buy the
    │  (data dated ≤ t)        │                               │  next day's selection
    ├─ predict, select         └──────── hold one session ─────┘
```

There is one decision per trading day, from 2020-01-02 to the last available day.

### 2. What the model predicts: "will this stock go up tomorrow?"

- **Label:** for each eligible stock on day *t*, take its return from the open of *t*+1 to the open of *t*+2 (with the dividend if it goes ex at *t*+2). The label is **1** if that return is above zero and **0** otherwise. It is only known after the open of *t*+2.
- **Output:** the model outputs P(up). A probability of 0.62 means "62% chance this stock rises tomorrow".
- **Selection:** the **threshold** (app slider, default 0.55) is how confident the model must be before the portfolio buys.
- **Market effect:** on a typical day 52% of the stocks rise, but that share swings by about ±22 points from day to day because stocks move together. So this label rewards guessing the market's direction as much as picking stocks. The deciles in "Previsto vs realizado" rank stocks within each day, so they show only the stock-picking part. The market call shows up as how many stocks pass the threshold each day ("Nº de ações na carteira").
- **Earlier experiment:** round 3 tried a relative label (beats the day's median) and multi-day horizons; that version is in the git history (commit 191308b).

### 3. How the model is trained

- **One model for all stocks.** Each training example is one (stock, day) row with **that stock's own** features and its label. The model learns one rule, "a stock in this state tends to rise / fall", and applies it to every stock. One model per stock would have only a few dozen examples each, far too few for noisy returns.
- **Only outcomes already known.** At decision *t* the model trains on the rows dated *t*−*X*−1 … *t*−2 (the label of *t*−1 needs the open of *t*+1, which is not known yet). With X = 21 that is about 21 × 480 ≈ 10,000 examples.
- **Retraining.** The model is refit every *k* sessions (k = 1: every day). Between refits, the last model keeps predicting; it never sees later data. A larger *k* is the main way to make a run faster: k = 5 is about 4.5× faster.
- **Degenerate windows.** If every training example has the same label (possible with X = 1 on a day when everything fell), the prediction is that label's frequency.

### 4. Features (34, all computed from data dated ≤ *t*)

- **Stock (19):**
  - returns over 1, 5, 21 and 63 days, and 12−1-month momentum;
  - overnight gap and intraday move;
  - volatility, its ratio and the average range;
  - distance to the 50- and 200-day averages and to the 52-week high;
  - RSI(14);
  - volume ratio, dollar volume, dividend yield and beta.
- **Cross-section (5):** the stock's percentile among the index members that day for 1/5/21-day return, momentum and volatility.
- **Correlation peers (4):** the 10 members most correlated with the stock over the last 63 days, their average 1/5/21-day return, and the stock's 5-day return minus theirs.
- **Market (6):** S&P 500 returns over 1/5/21 days, its volatility, its distance to the 200-day average, and the share of members above their 50-day average.

No normalization: trees don't need it, and statistics computed over the whole period would leak the future.

### 5. Portfolio and costs

- Every day, buy at the open of *t*+1 the stocks whose probability is **above the threshold**, each weighted in proportion to its probability, and sell them at the open of *t*+2.
- If no stock passes the threshold, stay in cash earning the 13-week T-bill rate (`^IRX`).
- **Costs:** *cost_bps* (default 5) per side on the value bought plus the value sold to move from yesterday's holdings (after their price moves) to today's.

### 6. How results are judged

- **Portfolio vs S&P 500 Total Return** (`^SP500TR`, open to open, same days), gross and net of costs: cumulative return, volatility, Sharpe (over the T-bill), maximum drawdown and hit ratio.
- **Model:**
  - AUC: the chance that a stock that rose got a higher probability than one that did not (0.5 = coin flip).
  - Accuracy: share of correct calls at a 0.5 cut.
  - IC: rank correlation between probability and the realized return on each day, averaged over the days.
- **Predicted vs realized** (app):
  - each day, the stocks are split into 10 groups by probability;
  - with real signal, the mean next-day return rises from group 1 to 10;
  - in a well-calibrated model, the share that rose matches the predicted probability;
  - the "top 10% minus bottom 10%" spread accumulates over time.

### Run parameters

| Parameter | Meaning | Default |
|---|---|---|
| *X* (`--window`) | sessions of training examples | 21 |
| *k* (`--retrain`) | refit the model every *k* sessions | 1 |
| threshold | minimum probability of going up to buy (app slider) | 0.55 |
| cost | bps per side on traded value (app input) | 5 |

## Data and guarantees

- **Prices:** from Yahoo Finance.
  - Not adjusted for dividends: `auto_adjust=False`, and Yahoo's `Adj Close` is never used. The dividend is added to the return on the day it goes ex.
  - Adjusted for splits, as Yahoo always delivers them; otherwise a 4:1 split would look like a −75% day.
- **Constituents:** point-in-time S&P 500 constituents ([fja05680/sp500](https://github.com/fja05680/sp500)). Price history only, no macro data.
- **Universe at *t*:** the index members on that date that have prices and at least 63 days of history.
- **Coverage:** Yahoo has prices for about 86% of the members in 2018, rising to 99% in 2026. Most of the missing ones left the market (acquired or delisted), which leaves some survivorship bias, mostly in the early years. Three are renames whose old ticker has no data on Yahoo: BK→BNY, MMC→MRSH and FI→FISV. Their membership before the rename is lost because they are not in the alias list. The app lists all of them.
- **No lookahead:** a decision at *t* uses only data dated on or before *t*, and the model trains only on outcomes already known at *t*. Tests perturb every price after a date and require identical predictions up to it.
- **Portfolio check:** an independent event-by-event simulation must reproduce the portfolio curve to 10 decimal places. It tracks cash, share counts, fees on traded value, carry on cash and dividends.
- **Out of scope:** live trading and hyperparameter tuning.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python main.py download                       # latest constituents and prices
venv/bin/python main.py features                       # features, once per download
venv/bin/python main.py run --window 21 --retrain 1    # one walk-forward run
venv/bin/streamlit run app/dashboard.py                # results; pick X and k in the sidebar and click "Rodar"
venv/bin/python -m pytest
```

All defaults live in `config/config.py`.
