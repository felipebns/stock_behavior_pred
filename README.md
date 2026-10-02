# Stock Behavior Prediction

Every trading day, after the close, a model estimates for each stock in the S&P 500 **on that day** the probability that it rises from the next day's open to that day's close. A long-only portfolio buys, at the open, the stocks whose probability is above a confidence threshold, weighted by that probability, and sells them at the close; if none qualifies, it holds cash at the 13-week T-bill rate. The question it answers: can intraday predictions from price history alone beat the S&P 500 after costs?

## How it works

### 1. One decision, step by step

```
close of t                       open of t+1                         close of t+1
    │                                 │                                    │
    ├─ compute features               ├─ buy the selected stocks           ├─ sell them all
    │  (data dated ≤ t)               │                                    │
    ├─ predict, select                └────────── hold during the day ─────┘
```

There is one decision per trading day, from 2020-01-02 to the last available day. The portfolio never holds overnight.

### 2. What the model predicts: "will this stock go up during tomorrow's session?"

- **Label:** for each eligible stock on day *t*, take its return from the open of *t*+1 to the close of *t*+1. The label is **1** if that return is above zero. It is known at the close of *t*+1. No dividend enters: whoever buys at the open of the ex-date already pays the price without it.
- **Output:** the model outputs P(up). A probability of 0.62 means "62% chance this stock closes above its open tomorrow".
- **Threshold:** the threshold (app slider, default 0.55) is how confident the model must be before the portfolio buys.
- **Market effect:** most stocks move with the market on any given day, so this label rewards guessing the market's direction as much as picking stocks. The deciles in "Previsto vs realizado" rank stocks within each day, so they show only the stock-picking part. The market call shows up as how many stocks pass the threshold each day ("Nº de ações na carteira").
- **Where the market's return happens:** from 2020-01 to 2026-09 the S&P 500 TR returned +36.6% summing only open → close sessions and +92.2% summing only close → open nights. An intraday strategy gives up the overnight part by design, so its fair benchmark is the S&P intraday.

### 3. How the model is trained

- **Only outcomes already known.** At decision *t* the model trains on the rows dated *t*−*X* … *t*−1. The label of *t*−1 is known at the close of *t*.
- **Scope**, chosen in the app:
  - **one model for all stocks** (`pooled`): each example is one (stock, day) row with that stock's own features; the model learns one rule and applies it to every stock. With X = 21 that is about 21 × 470 ≈ 10,000 examples.
  - **one model per stock** (`per_stock`): each stock trains only on its own rows. If it has fewer than 63 known examples, or a single class, it predicts its up-frequency in the window. Default X = 252 and a refit every 21 sessions.
- **Model**, chosen in the app; default parameters live in `config/config.py` (`MODEL_PARAMS`), ready for tuning:
  - LightGBM;
  - regularized logistic regression;
  - random forest;
  - extra trees;
  - Gaussian naive Bayes.
  The scikit-learn models fill missing features with the training rows' median, and the logistic also standardizes them with the training rows' statistics.
- **Retraining:** the model is refit every *k* sessions. Between refits the last model keeps predicting; it never sees later data.

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

No normalization over the whole period: it would leak the future.

### 5. Portfolio and costs

- Every day, buy at the open the stocks whose probability is above the threshold, each weighted in proportion to its probability, and sell them at the close.
- If no stock passes the threshold, stay in cash earning the 13-week T-bill rate (`^IRX`).
- **Costs:** *cost_bps* (default 5) per side on the value bought and the value sold. Every invested day pays a full round trip: net = (1 + gross)(1 − fee)² − 1. At 5 bps that is about 22% a year if invested every day.

### 6. Choosing parameters without fooling yourself

- **Sweep:** `main.py sweep` runs a grid of model × scope × X × *k* and prints AUC and IC for each new combination.
- **Walk-forward of parameters** (app tab "Walk-forward"):
  - every candidate is a (run, threshold) pair;
  - every 63 sessions, it picks the pair with the best net Sharpe over the previous 252 sessions, using only days already realized, and follows it until the next pick;
  - nothing is refit, because every run's predictions are already out of sample.

  The tab shows the stitched curve against the S&P intraday, which run and threshold were picked when, and how long each stayed.
- Picking the best run by looking at the whole period is data snooping; the walk-forward curve is the honest estimate.

### 7. How results are judged

- **Portfolio vs S&P 500 Total Return, open to close on the same days**, gross and net of costs: cumulative return, volatility, Sharpe (over the T-bill), maximum drawdown and hit ratio.
- **Model:**
  - **AUC:** the chance that a stock that rose got a higher probability than one that did not (0.5 = coin flip).
  - **Accuracy:** share of correct calls at a 0.5 cut.
  - **IC:** the rank correlation between probability and the realized return on each day, averaged over the days.
- **Predicted vs realized** (app):
  - each day, the stocks are split into 10 groups by probability;
  - with real signal, the mean return rises from group 1 to 10;
  - in a well-calibrated model, the share that rose matches the predicted probability;
  - the "top 10% minus bottom 10%" spread accumulates over time.

### Run parameters

| Parameter | Meaning | Default |
|---|---|---|
| model (`--model`) | lightgbm, logistic, random_forest, extra_trees, naive_bayes | lightgbm |
| scope (`--scope`) | pooled (one model for all stocks) or per_stock | pooled |
| *X* (`--window`) | sessions of training examples | 21 pooled, 252 per stock |
| *k* (`--retrain`) | refit every *k* sessions | 1 pooled, 21 per stock |
| threshold | minimum probability of going up to buy (app slider) | 0.55 |
| cost | bps per side on traded value (app input) | 5 |
| lookback / step | walk-forward window and how often it re-picks (app) | 252 / 63 |

## Data and guarantees

- **Prices:** from Yahoo Finance.
  - Not adjusted for dividends: `auto_adjust=False`, and Yahoo's `Adj Close` is never used.
  - Adjusted for splits, as Yahoo always delivers them; otherwise a 4:1 split would look like a −75% day.
- **Constituents:** point-in-time S&P 500 constituents from [fja05680/sp500](https://github.com/fja05680/sp500). Price history only, no macro data.
- **Universe at *t*:** the index members on that date with a close on **each of the last 252 sessions**. A gap or a recent listing keeps a stock out until its window is complete. The rule uses only the past.
- **Coverage:**
  - Yahoo has prices for about 86% of the members in 2018, rising to 99% in 2026.
  - Most of the missing ones left the market (acquired or delisted), which leaves some survivorship bias, mostly in the early years.
  - Three are renames whose old ticker has no data on Yahoo: BK→BNY, MMC→MRSH and FI→FISV. Their membership before the rename is lost because they are not in the alias list.
  - The app lists all of them.
- **No lookahead:** a decision at *t* uses only data dated on or before *t*, and the model trains only on outcomes already known at *t*.
  - The tests perturb every price after a date and require bit-identical predictions up to it, for both scopes.
  - Every model's probability for a row is independent of the other rows predicted with it.
- **Portfolio check:** an independent event-by-event simulation must reproduce the portfolio curve to 10 decimal places. It tracks cash, share counts, fees on each buy and sell, and carry on cash.
- **Out of scope:** live trading.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python main.py download                                            # latest constituents and prices
venv/bin/python main.py features                                            # features, once per download
venv/bin/python main.py run --model lightgbm --scope pooled --window 21     # one walk-forward run
venv/bin/python main.py sweep --models lightgbm logistic naive_bayes --scopes pooled --windows 21 63 --retrain 5
venv/bin/python main.py sweep --models logistic naive_bayes --scopes per_stock --windows 252 --retrain 21
venv/bin/streamlit run app/dashboard.py                                     # results, new runs, walk-forward tab
venv/bin/python -m pytest
```

All defaults live in `config/config.py`.
