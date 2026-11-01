# Robust food-indexed cash transfers under partial price information

Code and derived data for the paper *A conformally calibrated linear-fractional robust model for food-indexed cash transfers and price monitoring under partial information* (in preparation for Socio-Economic Planning Sciences).

A fixed transfer budget is allocated across Nigeria's 36 states and FCT so that every recipient keeps the same share of a food entitlement. Local prices are seen only through published zonal averages, state extremes and delayed state data. The robust allocation pays each state in proportion to its worst-case need share, and its guarantee is calibrated by conformal prediction.

## Layout

| Path | Contents |
|---|---|
| `src/robust.py` | need, protection, efficiency; Charnes-Cooper LP for worst-case need shares; Theorem 1 allocation and guarantee |
| `src/fastlf.py` | grouped price polytopes: per-group continuous knapsacks and batched Dinkelbach iteration (Proposition 5) |
| `src/setup.py` | `Panel`: loads the cleaned state panel, basket weights and recipients; builds information polytopes |
| `src/polytope.py` | generic polytope builder (U1 to U6) |
| `tests/test_theory.py` | Proposition 1, Theorem 1 (brute force), monotonicity, full-information G = 1 |
| `experiments/s1_clean_and_audit.py` | cleaning, outlier log, aggregation audit (needs the NBS research dataset folder next to this repo) |
| `experiments/s2_basket_and_recipients.py` | NLSS 2018/19 basket weights and poor-population recipients (needs NLSS microdata, not included) |
| `experiments/e2_e10_gate1_nlss.py`, `e2b_absolute_need.py` | Gate 1: efficiency of flat, zonal and delayed-state rules |
| `experiments/e3_e4_backtest.py` | price-polytope robust rule, information sets I1 and I2 |
| `experiments/e3b_need_space.py` | need-box robust rule (Proposition 6), split and adaptive conformal |
| `experiments/e3c_frontier.py` | guarantee-constrained water-filling rule (Proposition 7) |
| `experiments/e5_monitoring.py` | greedy monitoring design with delays 0, 1, 3; accuracy-focused rival; exhaustive check |
| `experiments/e8_live.py` | live application Sep 2023 to May 2026 with horizon-calibrated bands |
| `experiments/make_tables_figures.py` | summary tables and figures |
| `data/derived/` | cleaned state panel (genuine state data only, to Aug 2023), published national/zonal averages and state extremes, basket weights, recipients |
| `results/` | tables (CSV) and figures |
| `paper/` | LaTeX manuscript |

## Reproduce

```
pip install -r requirements.txt
python -m pytest -q tests
cd experiments
python e3b_need_space.py && python e3c_frontier.py && python e5_monitoring.py && python e8_live.py && python make_tables_figures.py
```

## Data notes

- Food prices: NBS Selected Food Price Watch. Genuine state-level food prices end in August 2023 (June 2023 is missing). Some compiled files fill June 2023 and September 2023 to April 2024 by copying zonal averages to every state; those rows are excluded here.
- NLSS 2018/19 microdata are available from the World Bank Microdata Library under its terms of use and are not redistributed. Only aggregated weights and recipient counts derived from them are included.

## Main results (poor-household weights)

- Jan 2021 to Aug 2023: flat transfer realised 0.784 of attainable protection, zonal indexation 0.689, plug-in 0.922, guarantee-constrained rule (kappa = 0.9) 0.910 with certified worst case 0.642 (flat 0.403). Certificates held in every month; coverage 87% at an 80% target.
- Monitoring: guarantee 0.721 with no monitored states, 0.792 with five, 0.837 with ten; with a one-month delay the gain mostly disappears.
- Sep 2023 to May 2026: guarantee falls from 0.88 to 0.67; flat rule certificate from 0.70 to 0.39.
