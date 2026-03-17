## Smoother Trends, Sharper Tests: Optimal Filtering of Event Study Estimates

Event study designs estimate period-specific treatment effects β̂\_t with known standard errors from two-way fixed effects regressions. We treat the β̂\_t sequence as observations from a local linear trend state-space model and apply the Rauch–Tung–Striebel (Kalman) smoother to recover the treatment effect trajectory and its derivative. The approach uses the known, heteroskedastic regression standard errors as observation noise — a structural advantage over generic smoothing methods.

## Key Results

| Metric | Improvement |
|---|---|
| Level MSE (β̂\_t vs true β\_t) | 62–82% reduction vs raw estimates |
| Derivative MSE (Δβ̂\_t vs true Δβ\_t) | 77–98% reduction vs raw estimates |
| Parallel trends power (anticipation) | 61.1% vs 9.0% (raw Wald), bootstrap-calibrated at 5% |
| Parallel trends power (small pretrend) | 27.6% vs 4.8% (raw Wald) |
| Size under null | 5.2% (correctly sized with bootstrap calibration) |

## Why This Works

1. **The Kalman gain adapts to local precision.** When a period's β̂\_t has a large SE (few observations, noisy outcome), the smoother trusts the trend model. When the SE is tight, it trusts the data. Fixed-window smoothers (LOESS, Savitzky–Golay) cannot do this.

2. **Joint estimation of level and derivative.** The state vector is \[β\_t, Δβ\_t\]. The smoother gives you the rate of change of the treatment effect for free — and with 96–98% lower MSE than finite-differencing the raw estimates.

3. **The derivative-based parallel trends test.** Rather than testing whether pre-treatment β̂\_t are jointly zero (low power because the estimates are noisy), we test whether the smoothed Δβ̂\_t are jointly zero. A trending pre-treatment violation is far more visible in the slope than in the level.

## Repository Structure

```
ms/
  smoother_trends_sharper_tests.tex   # LaTeX source
  smoother_trends_sharper_tests.pdf   # Compiled paper (13 pages, 4 figures, 3 tables)

figs/
  paper_fig1.png                      # Fig 1: Illustrative examples (4 DGPs)
  paper_fig2.png                      # Fig 2: MSE reduction summary
  paper_fig3.png                      # Fig 3: Size and power
  paper_fig4.png                      # Fig 4: Sensitivity to Q

scripts/
  paper_simulation.py                 # Reproduces all tables and figures
```

## Reproducing the Paper

### Requirements

```
pip install numpy pandas scipy matplotlib scikit-learn
```

No R dependencies. No PyTorch. Everything runs in base scientific Python.

### Running

```bash
cd scripts
python paper_simulation.py
```

Takes ~5 minutes. Produces:
- `tabs/table1_mse.csv`, `tabs/table2_size_power.csv`, `tabs/table3_sensitivity.csv`
- `figs/paper_fig1.png` through `figs/paper_fig4.png`

To compile the paper:

```bash
cd ms
pdflatex smoother_trends_sharper_tests.tex
bibtex smoother_trends_sharper_tests
pdflatex smoother_trends_sharper_tests.tex
pdflatex smoother_trends_sharper_tests.tex
```

## Method Summary

### State-Space Model

```
State:       x_t = [β_t, Δβ_t]'
Transition:  x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q)
Observation: β̂_t = H x_t + v_t,       v_t ~ N(0, σ̂²_t)

F = [[1, 1], [0, 1]]     # local linear trend
H = [1, 0]               # observe level only
Q = diag(q_ℓ, q_s)       # process noise (tuning parameter)
R_t = σ̂²_t               # KNOWN from TWFE regression
```

### Bootstrap Inference

The Kalman smoother induces serial correlation in the smoothed estimates, invalidating chi-squared critical values. We use a parametric bootstrap under H₀: β\_t = 0:

1. Draw β̂\_t^(b) ~ N(0, σ̂²\_t) for b = 1, ..., B
2. Apply Kalman smoother to each draw
3. Compute test statistic for each draw
4. Use the (1-α) quantile as the critical value

This yields exact size control because the null DGP is fully specified.

## Connections to Related Work

This paper sits at the intersection of two literatures that haven't talked to each other:

**Event study / DiD methods**: Roth (2022, AER:I) shows pre-trend tests have low power. Rambachan & Roth (2023, RES) propose sensitivity analysis for bounded violations. Borusyak, Jaravel & Spiess (2024, RES) derive efficient imputation estimators. None use state-space methods.

**State-space econometrics**: Harvey (1989), Durbin & Koopman (2012) develop the Kalman filter/smoother framework. Harvey (1985) shows the HP filter is a special case. None apply it to event study coefficient sequences.

Our approach is complementary to Rambachan–Roth: they ask "how sensitive are conclusions to bounded violations?" We ask "can we detect violations more powerfully?" The Kalman-smoothed estimates could serve as inputs to their sensitivity framework.

## Limitations

- **Smoothness assumption.** The local linear trend model assumes β\_t evolves smoothly. For sharp, immediate treatment effects, the smoother attenuates the jump and propagates some post-treatment signal backward. MSE is still reduced (70% for levels), but researchers should be aware of this.

- **Independence across periods.** We assume β̂\_t are independent across t, which holds under standard TWFE with independent clusters. Serial correlation in the errors would require extending the observation noise model.

- **Process noise Q.** The choice of Q = diag(q\_ℓ, q\_s) is a tuning parameter analogous to bandwidth in nonparametric regression. We provide recommended defaults and urge sensitivity analysis. Marginal likelihood or cross-validation selection of Q is a natural extension.

- **KS Wald test can be mildly oversized.** At (N=200, σ=2.0), the bootstrap-calibrated KS Wald test rejects at 7.9% instead of 5%. Increasing B from 499 to 999+ resolves this. The derivative test is correctly sized across all configurations.

## Origin

This project started from exploring whether the [`incline`](https://github.com/finite-sample/incline) package (Savitzky–Golay and spline smoothing for noisy time series) could improve neural network gradient estimation. That exploration led to a systematic comparison of smoothing methods (Kalman, SG, LOESS, local polynomial) for online vs. retrospective estimation, which revealed that the Kalman filter's adaptive noise weighting is critical for problems where the signal-to-noise ratio varies — and that applied econometrics, despite being full of such problems, barely uses it.

## License

MIT
