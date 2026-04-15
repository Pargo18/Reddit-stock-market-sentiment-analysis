# Reddit Stock Market Sentiment Analysis

An empirical study investigating whether retail investor sentiment expressed on **r/WallStreetBets** influences **implied volatility (IV)** of S&P 500 options. The project combines NLP-based sentiment analysis of Reddit submissions, options market data extraction, volatility modeling, and statistical hypothesis testing (linear regression and Granger causality) to test whether the "meme stock era" of 2021 left measurable footprints on derivatives pricing.

---

## Table of Contents

- [Research Question](#research-question)
- [Background & Motivation](#background--motivation)
- [Methodology](#methodology)
  - [Stock Universe Selection](#1-stock-universe-selection)
  - [Reddit Data Extraction](#2-reddit-data-extraction)
  - [Sentiment Analysis](#3-sentiment-analysis)
  - [Options Data & Implied Volatility](#4-options-data--implied-volatility)
  - [Feature Engineering](#5-feature-engineering)
  - [Statistical Testing](#6-statistical-testing)
  - [Trading Strategy Backtest](#7-trading-strategy-backtest)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Data Notice](#data-notice)
- [References](#references)

---

## Research Question

> **Does the sentiment expressed in r/WallStreetBets discussions exert a statistically significant effect on the implied volatility of the underlying stocks' options?**

If sentiment-driven retail flow influences derivatives markets, this would suggest the existence of **mispricing opportunities** that could be systematically exploited via long/short volatility strategies.

---

## Background & Motivation

The 2021 retail trading boom — catalyzed by zero-commission brokerages, COVID-era stimulus, and viral coordination on social platforms — fundamentally altered the microstructure of US equity and options markets. The **GameStop short squeeze** of January 2021 demonstrated that retail-driven sentiment, organized through r/WallStreetBets, could move billion-dollar institutional positions and trigger broker-dealer liquidity crises.

While extensive academic literature exists on sentiment derived from **Twitter** (Bollen et al., 2011), **Google Trends** (Da et al., 2011), and **financial news** (Tetlock, 2007), the role of **Reddit-based community sentiment** in derivatives pricing remains underexplored. This project addresses that gap by:

1. Constructing a custom dataset of the 50 most-discussed S&P 500 tickers on r/WallStreetBets throughout 2021
2. Quantifying daily sentiment per ticker using NLP techniques
3. Aligning sentiment time series with implied volatility surfaces extracted from options data
4. Applying rigorous econometric tests to identify causal relationships
5. Implementing a long/short volatility strategy to assess economic significance

---

## Methodology

### 1. Stock Universe Selection

**Notebook:** `Main code/Tickers for stock selection.ipynb`

- Scraped r/WallStreetBets submissions for the entire year 2021
- Identified the 50 most frequently mentioned S&P 500 tickers
- Filtered for liquidity criteria: minimum daily options volume, bid-ask spread, open interest

### 2. Reddit Data Extraction

**Notebook:** `Main code/Feature extraction (mentions, sentiment etc.).ipynb`

- Extracted all posts and comments containing ticker mentions (cashtags `$XYZ` and naked tickers)
- Aggregated post metadata: upvote ratio, score, awards, comment count, author karma
- Constructed daily mention counts per ticker as a proxy for **attention intensity**
- Output: `Data/wallstreetbets_2021.csv`

### 3. Sentiment Analysis

**Approach:** Lexicon-based and rule-based NLP scoring tailored to financial slang.

- Standard sentiment analyzers (VADER, TextBlob) underperform on r/WallStreetBets due to:
  - Domain-specific slang ("tendies", "diamond hands", "to the moon", "bagholder")
  - Sarcasm and irony as primary modes of expression
  - Heavy emoji usage with directional implications (🚀 = bullish, 💎🙌 = hold conviction)
- A custom lexicon was developed and applied to compute per-post sentiment scores
- Daily sentiment per ticker = volume-weighted average of post-level scores

### 4. Options Data & Implied Volatility

**Scripts:**
- `Main code/Download_Option_Data.py` — Downloads option chains for the 50 selected tickers
- `Main code/Volatility_Calculator.py` — Computes implied volatility from option prices using the **Black-Scholes** model (Newton-Raphson root finding)
- `Main code/Fix_implied_volatility.py` — Cleans IV time series, handles missing data, removes outliers from illiquid contracts

**Outputs:**
- `Data/Implied_volatility.csv` — Daily IV per ticker
- `Data/OrganizedImpliedVolatility.csv` — Cleaned, regression-ready format
- `Data/Close_data.csv`, `Return_data.csv`, `Volume_data.csv` — Underlying market data

### 5. Feature Engineering

**Notebook:** `Main code/Feature engineering.ipynb`
**Script:** `Main code/generate_features.py`

Engineered features for the regression analysis:

| Feature | Description |
|---|---|
| `Sentiment` | Daily volume-weighted sentiment score per ticker |
| `Mentions` | Daily mention count (attention proxy) |
| `IV_t` | Implied volatility at time t |
| `IV_t-1` | Lagged IV (autoregressive control) |
| `Returns` | Daily underlying log returns |
| `Volume` | Trading volume (liquidity control) |
| `Realized_Vol` | Historical realized volatility (control) |

### 6. Statistical Testing

**Scripts:**
- `Main code/Linear_Regression.py` — OLS regression of IV on sentiment features with controls
- `Main code/Visualize_Regression.py` — Coefficient plots, residual diagnostics, Q-Q plots
- `Extra material/granger analysis.pdf` — Granger causality testing (does sentiment lead IV?)

**Hypotheses tested:**
- **H₀:** Sentiment has no effect on implied volatility (β_sentiment = 0)
- **H₁:** Sentiment is a statistically significant predictor of IV (β_sentiment ≠ 0)

**Robustness checks:**
- Heteroskedasticity-consistent (HC3) standard errors
- Granger causality at multiple lag specifications (1–10 days)
- Per-stock and pooled-panel regressions
- Subsample analysis (high-attention vs. low-attention periods)

### 7. Trading Strategy Backtest

A **long/short volatility strategy** was implemented to test economic significance:
- **Long** straddles on tickers with high positive sentiment shocks
- **Short** straddles on tickers with high negative sentiment shocks
- Performance metrics: Sharpe ratio, maximum drawdown, hit rate, alpha vs. VIX

---

## Key Findings

> **The empirical analysis does NOT find statistically significant evidence that r/WallStreetBets sentiment causally influences implied volatility** at conventional significance levels (p < 0.05) once controlling for autoregressive IV dynamics, realized volatility, and trading volume.

However, the project yields several notable contributions:

1. **Economic significance of attention:** While sentiment direction is not significant, **mention volume** (attention intensity) shows weak predictive power for IV expansion — consistent with the "attention-induced volatility" hypothesis.
2. **L/S strategy returns:** The implemented long/short volatility strategy demonstrates non-trivial returns, suggesting that even in the absence of statistical significance, sentiment-based filters carry tradable information.
3. **Methodological framework:** Provides a reusable NLP + options analytics pipeline for future research on social-media-driven market dynamics (e.g., extending to TikTok FinTok, Discord servers, Twitter cashtags).

---

## Project Structure

```
Reddit-stock-market-sentiment-analysis/
├── Main code/
│   ├── Tickers for stock selection.ipynb           # S&P 500 ticker filtering
│   ├── Feature extraction (mentions, sentiment etc.).ipynb  # Reddit NLP pipeline
│   ├── Download_Option_Data.py                     # Options chain downloader
│   ├── Volatility_Calculator.py                    # Black-Scholes IV computation
│   ├── Fix_implied_volatility.py                   # IV cleaning
│   ├── Data preparation.ipynb                      # Data alignment & merging
│   ├── Feature engineering.ipynb                   # Feature construction
│   ├── generate_features.py                        # Feature generation script
│   ├── Linear_Regression.py                        # OLS regression analysis
│   ├── Visualize_Regression.py                     # Regression diagnostics
│   └── Relevant data visualizations.ipynb          # EDA plots
├── Data/
│   ├── wallstreetbets_2021.csv                     # Reddit submissions dataset
│   ├── FinalTickers.csv                            # Final 50 selected tickers
│   ├── Implied_volatility.csv                      # Raw IV data
│   ├── OrganizedImpliedVolatility.csv              # Processed IV
│   ├── Close_data.csv, Return_data.csv, Volume_data.csv  # Market data
│   ├── volatility_data.csv                         # Realized vol
│   ├── implied_vol.xlsx                            # Excel-formatted IV
│   ├── Output/                                     # Regression outputs
│   └── submissions/                                # Per-ticker Reddit data
└── Extra material/
    ├── FM2021 Lab 5 - Implied Volatility.pdf      # IV theory reference
    ├── Implied Volatility.xlsm                     # IV calculation spreadsheet
    ├── Sentiment-Volatility regression hypothesis testing.pptx
    ├── granger analysis.pdf                        # Granger causality results
    ├── README.md
    └── LICENSE
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **Data Acquisition** | PRAW (Reddit API), yfinance, pandas-datareader |
| **NLP / Sentiment** | NLTK, custom WSB lexicon, regex tokenization |
| **Quantitative Finance** | Black-Scholes IV (Newton-Raphson), realized volatility estimators |
| **Statistical Analysis** | statsmodels (OLS, Granger causality), scipy.stats |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Core** | Python 3.x, Jupyter Notebooks |

---

## Data Notice

> The Reddit submissions dataset (`Data/wallstreetbets_2021.csv`) and option chain data were collected via the Reddit API and commercial options data providers throughout 2021. While portions are included for reproducibility, users intending to extend this work should refresh the data via the provided scripts (`Download_Option_Data.py`, `Volatility_Calculator.py`) and respect the terms of service of the underlying data sources.

---

## References

- **Bollen, J., Mao, H., & Zeng, X. (2011).** Twitter mood predicts the stock market. *Journal of Computational Science*, 2(1), 1–8.
- **Mao, H., Counts, S., & Bollen, J. (2011).** Predicting financial markets: Comparing survey, news, Twitter and search engine data. *arXiv preprint*.
- **Da, Z., Engelberg, J., & Gao, P. (2011).** In search of attention. *The Journal of Finance*, 66(5), 1461–1499.
- **Tetlock, P. C. (2007).** Giving content to investor sentiment: The role of media in the stock market. *The Journal of Finance*, 62(3), 1139–1168.
- **Black, F., & Scholes, M. (1973).** The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
