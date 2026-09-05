# BetAI

**"Fair odds" models for NFL and NBA games built from public ESPN stats and Sleeper injury data.** For sports-betting research — estimating each team's true win probability and comparing model lines against the market.

## Overview

BetAI pulls completed games from ESPN's public endpoints, extracts team (and, for NBA, player) box-score stats, builds rolling pre-game features and home-minus-away differentials, then trains a calibrated classifier to produce a "fair" win probability / fair odds for each matchup. For upcoming games it layers on an injury-based log-odds adjustment using Sleeper's injury tags (more reliable than ESPN's), prints who's hurt, and the NBA pipeline additionally models player props (points, rebounds, assists, threes) and emits over/under recommendations.

The repo is a set of Jupyter notebooks — successive versions and model variants of the same pipeline — rather than a single packaged app.

## Tech stack

- **Python** + **Jupyter**
- **pandas / NumPy** — data wrangling and feature engineering
- **scikit-learn** — Logistic Regression, Elastic-Net Logistic, ExtraTrees, boosted Decision Trees, MLP neural net; probability calibration
- **requests** — ESPN and Sleeper public APIs

## Prerequisites

- Python 3.10+
- Jupyter

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# the model notebooks also need:
pip install requests scikit-learn
```

> `requirements.txt` pins only the core data stack (numpy, pandas, dateutil). The model notebooks additionally require `requests` and `scikit-learn`, as noted in each notebook's header.

## Usage

```bash
jupyter notebook
```

Open a notebook and run all cells; each is self-contained and pulls live data from ESPN/Sleeper, trains, and prints performance (accuracy, log-loss), latest completed-week fair odds, and upcoming-game odds (base vs. injury-adjusted).

**NFL pipeline (versioned iterations + model variants):**

| Notebook | Model |
|----------|-------|
| `BetAI_V1` – `BetAI_V4` | End-to-end pipeline iterations (Ridge → calibrated LogReg + Sleeper injuries) |
| `BetAI_ElasticNetLogistic` | Elastic-Net Logistic Regression |
| `BetAI_DT` | Boosted decision trees |
| `BetAI_ExtraTrees` | ExtraTrees ensemble |
| `BetAI_NN` | MLP neural net |

**NBA pipeline:**

| Notebook | Description |
|----------|-------------|
| `BetAI_NBA_V1`, `BetAI_NBA_V2` | NBA win model + player props (points/rebounds/assists/threes) |

**Data exploration:**

| Notebook | Description |
|----------|-------------|
| `ESPNDATA`, `ESPNJSONFILE` | Scratch notebooks for ESPN endpoints and JSON parsing |

## Project structure

```
bet-ai/
├── BetAI_V1..V4.ipynb              # NFL pipeline iterations
├── BetAI_DT / ExtraTrees / NN / ElasticNetLogistic.ipynb  # NFL model variants
├── BetAI_NBA_V1..V2.ipynb          # NBA win model + props
├── ESPNDATA.ipynb / ESPNJSONFILE.ipynb  # API/data exploration
├── nfl_players_*.csv, nfl_teams_*.csv    # Cached reference data
├── sleeper_players_nfl_cache.json        # Cached Sleeper player data
├── espn_summary_*.json                   # Sample ESPN summary payload
├── ou_recommendations_upcoming.csv       # Generated over/under recommendations
└── requirements.txt
```
