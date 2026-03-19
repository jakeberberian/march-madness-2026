# March Madness 2026 Projection System

Custom statistical modeling and simulation toolkit for NCAA Tournament bracket optimization across 7 pool configurations.

## Quick Start

```bash
pip install -r requirements.txt
jupyter lab
```

Run notebooks in order: `01` → `02` → `03` → `04` → `05` → `06`

Delete `./data/`, `./models/`, `./results/`, `./brackets/`, `./plots/` before a clean re-run.

## Architecture

```
01_data_scraping.ipynb    → Scrape & engineer features        → data/
02_modeling.ipynb          → Train 8 models + ensemble         → models/
03_simulation.ipynb        → 1M bracket simulations            → results/
04_evaluation.ipynb        → Diagnostics & plots               → plots/
05_bracket_generator.ipynb → Pool-specific optimal brackets    → brackets/
06_round1_analysis.ipynb   → Spread & total (O/U) projections  → results/
```

## Data Pipeline (Notebook 01)

**Sources:**
- BartTorvik: AdjOE, AdjDE, Barthag, AdjTempo, WAB (2008-2026)
- Sports Reference: SRS, SOS, eFG%, TOV%, ORtg, Pace, ORB% (2002-2026)
- Historical tournament results scraped from Sports Reference (2002-2025)

**Features (31 per matchup):**
- 11 seed-derived features (seed_diff, seed_sum, seed_diff_sq, etc.)
- 4 SRef basic diffs (SRS, SOS, eFG%, TOV%)
- 3 SRef advanced diffs (ORtg, Pace, ORB%)
- 5 BartTorvik diffs (AdjOE, AdjDE, Barthag, AdjTempo, WAB)
- 8 level features for total prediction (Pace_avg, ORtg_avg, SRS_sum, AdjOE_avg, AdjDE_avg, AdjTempo_avg, OE_vs_DE_1, OE_vs_DE_2)

**Key fix:** BartTorvik `team_results.csv` for 2008-2022 has a column alignment bug (headers shifted left by one). Auto-detected and corrected during scraping.

## Models (Notebook 02)

| Model | Type | Role |
|-------|------|------|
| Logistic Regression | Frequentist | Interpretable baseline, strong generalizer |
| XGBoost | Frequentist | Non-linear interactions, feature importance |
| Random Forest | Frequentist | Robust ensemble, OOB validation |
| Neural Network | Deep Learning | Residual architecture, MPS-accelerated |
| Bayesian Logistic Regression | Bayesian | Full posterior, credible intervals |
| Bayesian Hierarchical | Bayesian | Non-centered parameterization, partial pooling across seed tiers |
| Bayesian Neural Network | Bayesian | Weight uncertainty via ADVI (excluded if AUC < 0.5) |
| Beta-Binomial | Bayesian | Conjugate seed matchup model (used in simulation only) |

**Validation methodology:**
- Rolling temporal CV: train on 2008-N, validate on N+1, for N in {2014, ..., 2024} (9 folds)
- Held-out test set: 2025 tournament (never seen during training or model selection)
- Ensemble weights: CV AUC-squared (penalizes weak models quadratically)
- Final deployment models retrained on 2008-2024 before generating 2026 predictions

## Simulation (Notebook 03)

- 1,000,000 tournament simulations with batch processing (25k per batch)
- Round-aware probability cache: matchup probabilities computed per round (round_x_seeddiff feature varies by stage)
- Bayesian posterior sampling with noise injection for simulation uncertainty
- Output: per-team advancement probabilities for R64 through Champion

## Bracket Generator (Notebook 05)

**7 pools with tailored strategies:**

| Pool | Size | Scoring | Strategy |
|------|------|---------|----------|
| Family | ~10 | ESPN | Chalk, portfolio pick #1 |
| Fantasy Prem | ~4 | ESPN | Chalk, portfolio pick #2 |
| College | ~8 | Yahoo (1-2-4-8-16-32) | Chalk, portfolio pick #3 |
| Derek's | ~14 | ESPN | Chalk, portfolio pick #4 |
| DK Company | ~2500 | ESPN | Contrarian (cf=0.40), prob floors |
| DK ATLAS Org | ~35 | Custom (1pt / # pickers) | Inverse-ownership optimization |
| DK Analytics Org | ~88 | ESPN | Moderate contrarian (cf=0.20) |

**Key design decisions:**
- Contrarian factor capped at 0.40 (1-seeds win 70% of championships; overcorrecting is -EV)
- Probability floors: F4 candidates need at least 10% F4 probability, champions need at least 2%
- Small pools use portfolio diversification: each pool gets a different champion pick to cover more outcomes
- Yahoo scoring uses straight 1-2-4-8-16-32 (no seed bonus)

## Spread and Total Projections (Notebook 06)

**Models:** Ridge Regression, Random Forest, XGBoost, Bayesian Ridge

**Separate pipelines:**
- Spread models: 14 differential features (predict margin of victory)
- Total models: 22 features (14 diffs + 8 level features to predict combined scoring)

Level features (Pace_avg, AdjOE_avg, AdjTempo_avg, etc.) are critical for total prediction. Without them, all totals regress to the historical mean (~140).

**Output per matchup:** spread, total, implied team points, favorite win probability, and model agreement range. Update `TARGET_ROUND` and matchup list as the tournament progresses.

## Reproducibility

```python
RANDOM_SEED = 51
SALT = 2026_03
SEED = RANDOM_SEED + SALT  # 202654
```

All models, simulations, and random processes use this seed. Change the salt for different reproducible runs.

## Hardware (Mac Mini M4, 16GB RAM)

- PyTorch uses MPS backend for GPU-accelerated neural network training
- PyMC NUTS sampling runs 4 chains across CPU cores
- 1M simulations complete in ~7 minutes with 25k batch size
- Rolling CV adds ~2-3 minutes to model training (9 folds x 5 models)
