"""
Monte Carlo Simulation for:
"Kalman Smoothing for Event Study Estimation"

Comprehensive simulation covering:
  1. Level and derivative MSE across DGPs
  2. Bootstrap-calibrated pre-trend tests (fixes size distortion)
  3. Derivative-based parallel trends test
  4. Power curves
  5. Sensitivity to Q specification
  6. Comparison: Raw, LOESS, Kalman filter, Kalman smoother

Outputs: tables and figures for the paper.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import chi2, norm
import warnings, time, json
warnings.filterwarnings("ignore")

MASTER_SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIGS_DIR = os.path.join(PROJECT_ROOT, "figs")
TABS_DIR = os.path.join(PROJECT_ROOT, "tabs")
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(TABS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "true": "#000000",
    "raw": "#888888",
    "loess": "#E69F00",
    "kalman": "#0072B2",
    "kalman_wald": "#0072B2",
    "kalman_deriv": "#D55E00",
    "raw_wald": "#888888",
}

# ══════════════════════════════════════════════════════════════════════════════
# CORE: State-Space Model & Kalman Smoother
# ══════════════════════════════════════════════════════════════════════════════

class KalmanEventStudySmoother:
    """
    Local linear trend state-space model for event study coefficients.

    State vector:  x_t = [β_t, Δβ_t]'
    Transition:    x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q)
                   F = [[1, 1], [0, 1]]
    Observation:   β̂_t = H x_t + v_t,      v_t ~ N(0, R_t)
                   H = [1, 0],  R_t = σ̂²_t (known from TWFE)

    Process noise Q = diag(q_level, q_slope) governs smoothness.
    """
    def __init__(self, q_level, q_slope):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.diag([q_level, q_slope])

    def smooth(self, beta_hat, se_t):
        """
        Rauch-Tung-Striebel smoother.

        Args:
            beta_hat: (T,) array of estimated coefficients
            se_t: (T,) array of standard errors

        Returns:
            levels: (T,) smoothed β_t
            slopes: (T,) smoothed Δβ_t (derivative)
            level_se: (T,) posterior std of β_t
            slope_se: (T,) posterior std of Δβ_t
            P_smooth: list of (2,2) posterior covariance matrices
        """
        T = len(beta_hat)
        F, H, Q = self.F, self.H, self.Q

        # --- Forward pass (Kalman filter) ---
        # Initialize at first observation
        x = np.array([beta_hat[0], 0.0])
        P = np.diag([se_t[0]**2, self.Q[1, 1] * 10])

        xs_filt, Ps_filt = [], []
        xs_pred, Ps_pred = [], []

        for t in range(T):
            # Predict
            if t > 0:
                x_p = F @ x
                P_p = F @ P @ F.T + Q
            else:
                x_p = x.copy()
                P_p = P.copy()

            xs_pred.append(x_p.copy())
            Ps_pred.append(P_p.copy())

            # Update
            R_t = np.array([[se_t[t]**2]])
            S = H @ P_p @ H.T + R_t
            K = P_p @ H.T / S[0, 0]
            inn = beta_hat[t] - (H @ x_p)[0]
            x = x_p + K.flatten() * inn
            P = (np.eye(2) - K @ H) @ P_p

            xs_filt.append(x.copy())
            Ps_filt.append(P.copy())

        # --- Backward pass (RTS smoother) ---
        xs_smooth = [xs_filt[-1].copy()]
        Ps_smooth = [Ps_filt[-1].copy()]

        for t in range(T - 2, -1, -1):
            P_pred_next = Ps_pred[t + 1]
            # Avoid singular inversion
            try:
                C = Ps_filt[t] @ F.T @ np.linalg.inv(P_pred_next)
            except np.linalg.LinAlgError:
                C = Ps_filt[t] @ F.T @ np.linalg.pinv(P_pred_next)

            x_s = xs_filt[t] + C @ (xs_smooth[0] - xs_pred[t + 1])
            P_s = Ps_filt[t] + C @ (Ps_smooth[0] - P_pred_next) @ C.T
            xs_smooth.insert(0, x_s)
            Ps_smooth.insert(0, P_s)

        levels = np.array([x[0] for x in xs_smooth])
        slopes = np.array([x[1] for x in xs_smooth])
        level_se = np.array([np.sqrt(max(P[0, 0], 1e-12)) for P in Ps_smooth])
        slope_se = np.array([np.sqrt(max(P[1, 1], 1e-12)) for P in Ps_smooth])

        return levels, slopes, level_se, slope_se, Ps_smooth


# ══════════════════════════════════════════════════════════════════════════════
# DGP
# ══════════════════════════════════════════════════════════════════════════════

def true_effect(T_pre, T_post, pattern):
    T = T_pre + T_post
    beta = np.zeros(T)
    t0 = T_pre
    if pattern == "gradual":
        for i in range(t0, T): beta[i] = 0.5 * (1 - np.exp(-0.3 * (i - t0 + 1)))
    elif pattern == "immediate":
        beta[t0:] = 0.4
    elif pattern == "fadeout":
        for i in range(t0, T): beta[i] = 0.5 * np.exp(-0.1 * (i - t0))
    elif pattern == "anticipation":
        for i in range(t0 - 3, t0): beta[i] = 0.05 * (i - (t0 - 4))
        for i in range(t0, T): beta[i] = 0.15 + 0.3 * (1 - np.exp(-0.3 * (i - t0 + 1)))
    elif pattern == "zero":
        pass
    elif pattern == "small_pretrend":
        for i in range(t0 - 4, t0): beta[i] = 0.015 * (i - (t0 - 5))
        for i in range(t0, T): beta[i] = 0.06 + 0.3 * (1 - np.exp(-0.3 * (i - t0 + 1)))
    return beta


def simulate_beta_hat(beta_true, n_units, frac_treated, sigma_eps, rng,
                      heteroskedastic=True):
    """Simulate β̂_t ~ N(β_t, σ²_t)."""
    T = len(beta_true)
    n_treat = int(n_units * frac_treated)
    n_ctrl = n_units - n_treat
    base_se = sigma_eps * np.sqrt(1.0 / n_treat + 1.0 / n_ctrl)

    if heteroskedastic:
        se_scale = np.ones(T)
        se_scale[:3] *= 1.5
        se_scale[-3:] *= 1.3
        # Add mild random variation in SE
        se_scale *= (1 + 0.1 * rng.randn(T)).clip(0.5, 2.0)
    else:
        se_scale = np.ones(T)

    se_t = base_se * se_scale
    beta_hat = beta_true + rng.randn(T) * se_t
    return beta_hat, se_t


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def wald_test_raw(beta_hat, se_t, T_pre):
    """Standard Wald test on pre-treatment coefficients."""
    pre_b = beta_hat[:T_pre]
    pre_se = se_t[:T_pre]
    return np.sum((pre_b / pre_se)**2)


def wald_test_kalman(smoother, beta_hat, se_t, T_pre):
    """Wald test using Kalman-smoothed pre-treatment coefficients."""
    levels, slopes, level_se, slope_se, Ps = smoother.smooth(beta_hat, se_t)
    pre_lvl = levels[:T_pre]
    pre_se = level_se[:T_pre]
    return np.sum((pre_lvl / pre_se)**2)


def derivative_test_kalman(smoother, beta_hat, se_t, T_pre):
    """
    Derivative-based parallel trends test.
    Under H0: Δβ_t = 0 for t < T_pre.
    Test statistic: sum of (Δβ̂_t / se(Δβ̂_t))^2 for pre-treatment periods.
    """
    levels, slopes, level_se, slope_se, Ps = smoother.smooth(beta_hat, se_t)
    pre_slopes = slopes[:T_pre]
    pre_slope_se = slope_se[:T_pre]
    return np.sum((pre_slopes / pre_slope_se)**2)


def bootstrap_critical_value(test_fn, T_pre, T_post, n_units, frac_treated,
                              sigma_eps, smoother, n_bootstrap=999, alpha=0.05,
                              rng=None):
    """
    Parametric bootstrap under H0: β_t = 0 for all t.
    Simulates β̂_t under the null, computes test statistics,
    returns the (1-alpha) quantile as the critical value.
    """
    if rng is None:
        rng = np.random.RandomState(12345)

    T = T_pre + T_post
    beta_null = np.zeros(T)
    stats = []

    for b in range(n_bootstrap):
        b_hat, b_se = simulate_beta_hat(beta_null, n_units, frac_treated,
                                         sigma_eps, rng)
        stat = test_fn(smoother, b_hat, b_se, T_pre) if smoother is not None \
               else test_fn(b_hat, b_se, T_pre)
        stats.append(stat)

    return np.percentile(stats, 100 * (1 - alpha))


# ══════════════════════════════════════════════════════════════════════════════
# LOESS baseline
# ══════════════════════════════════════════════════════════════════════════════

def loess_smooth(beta_hat, span=0.3):
    T = len(beta_hat)
    w = max(5, int(T * span))
    if w % 2 == 0: w += 1
    w = min(w, T - 2)
    if w < 5: return beta_hat.copy(), np.gradient(beta_hat)
    smoothed = savgol_filter(beta_hat, w, 3)
    deriv = savgol_filter(beta_hat, w, 3, deriv=1)
    return smoothed, deriv


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

T_PRE, T_POST = 12, 12
N_SIMS = 1000
N_BOOT = 499  # for bootstrap critical values

master_rng = np.random.RandomState(MASTER_SEED)

# --- Table 1: Level and Derivative MSE ---
print("=" * 70)
print("TABLE 1: MSE Comparison (1000 simulations)")
print("=" * 70)

smoother = KalmanEventStudySmoother(q_level=0.002, q_slope=0.001)
patterns = ["zero", "gradual", "immediate", "fadeout", "anticipation"]
n_configs = [(200, 1.0), (100, 1.0), (200, 2.0)]  # (n_units, sigma)

table1_seeds = master_rng.randint(0, 2**31, size=N_SIMS)

table1_rows = []
for n_units, sigma in n_configs:
    for pattern in patterns:
        raw_l, loess_l, kf_l, ks_l = [], [], [], []
        raw_d, loess_d, kf_d, ks_d = [], [], [], []

        for sim in range(N_SIMS):
            rng = np.random.RandomState(table1_seeds[sim])
            bt = true_effect(T_PRE, T_POST, pattern)
            bh, se = simulate_beta_hat(bt, n_units, 0.5, sigma, rng)
            td = np.gradient(bt)

            # Raw
            raw_l.append(np.mean((bh - bt)**2))
            raw_d.append(np.mean((np.gradient(bh) - td)**2))

            # LOESS
            ll, ld = loess_smooth(bh)
            loess_l.append(np.mean((ll - bt)**2))
            loess_d.append(np.mean((ld - td)**2))

            # Kalman smoother
            lvl, slp, lse, sse, _ = smoother.smooth(bh, se)
            ks_l.append(np.mean((lvl - bt)**2))
            ks_d.append(np.mean((slp - td)**2))

        row = {
            "n": n_units, "sigma": sigma, "pattern": pattern,
            "raw_level": np.mean(raw_l), "loess_level": np.mean(loess_l),
            "ks_level": np.mean(ks_l),
            "raw_deriv": np.mean(raw_d), "loess_deriv": np.mean(loess_d),
            "ks_deriv": np.mean(ks_d),
            "ks_level_pct": (1 - np.mean(ks_l)/np.mean(raw_l))*100,
            "ks_deriv_pct": (1 - np.mean(ks_d)/np.mean(raw_d))*100,
        }
        table1_rows.append(row)
        if n_units == 200 and sigma == 1.0:
            print(f"  {pattern:15s}: Level reduction={row['ks_level_pct']:.1f}%  "
                  f"Deriv reduction={row['ks_deriv_pct']:.1f}%")

t1 = pd.DataFrame(table1_rows)


# --- Table 2: Bootstrap-calibrated pre-trend tests ---
print("\n" + "=" * 70)
print("TABLE 2: Size and Power with Bootstrap Critical Values")
print("=" * 70)

# Compute bootstrap critical values under H0
print("\n  Computing bootstrap critical values (this takes a minute)...")
boot_rng = np.random.RandomState(master_rng.randint(0, 2**31))

# We need CVs for each (n, sigma) config
cv_cache = {}
for n_units, sigma in n_configs:
    key = (n_units, sigma)
    print(f"    Config n={n_units}, σ={sigma}...")

    # Raw Wald
    cv_raw = bootstrap_critical_value(
        lambda bh, se, tp: wald_test_raw(bh, se, tp),
        T_PRE, T_POST, n_units, 0.5, sigma, None, N_BOOT, 0.05, boot_rng)

    # Kalman Wald
    cv_kw = bootstrap_critical_value(
        lambda sm, bh, se, tp: wald_test_kalman(sm, bh, se, tp),
        T_PRE, T_POST, n_units, 0.5, sigma, smoother, N_BOOT, 0.05, boot_rng)

    # Kalman derivative test
    cv_kd = bootstrap_critical_value(
        lambda sm, bh, se, tp: derivative_test_kalman(sm, bh, se, tp),
        T_PRE, T_POST, n_units, 0.5, sigma, smoother, N_BOOT, 0.05, boot_rng)

    cv_cache[key] = {"raw": cv_raw, "kalman_wald": cv_kw, "kalman_deriv": cv_kd}
    print(f"      CVs: raw={cv_raw:.2f}, kalman_wald={cv_kw:.2f}, kalman_deriv={cv_kd:.2f}")

# Now run size and power
print("\n  Running size/power simulations...")

table2_seeds = master_rng.randint(0, 2**31, size=N_SIMS)

table2_rows = []
test_patterns = ["zero", "small_pretrend", "anticipation", "gradual", "immediate"]

for n_units, sigma in n_configs:
    cvs = cv_cache[(n_units, sigma)]
    for pattern in test_patterns:
        rej_raw, rej_kw, rej_kd = 0, 0, 0

        for sim in range(N_SIMS):
            rng = np.random.RandomState(table2_seeds[sim])
            bt = true_effect(T_PRE, T_POST, pattern)
            bh, se = simulate_beta_hat(bt, n_units, 0.5, sigma, rng)

            # Raw Wald
            stat_raw = wald_test_raw(bh, se, T_PRE)
            if stat_raw > cvs["raw"]: rej_raw += 1

            # Kalman Wald
            stat_kw = wald_test_kalman(smoother, bh, se, T_PRE)
            if stat_kw > cvs["kalman_wald"]: rej_kw += 1

            # Kalman derivative
            stat_kd = derivative_test_kalman(smoother, bh, se, T_PRE)
            if stat_kd > cvs["kalman_deriv"]: rej_kd += 1

        row = {
            "n": n_units, "sigma": sigma, "pattern": pattern,
            "rej_raw": rej_raw / N_SIMS,
            "rej_kalman_wald": rej_kw / N_SIMS,
            "rej_kalman_deriv": rej_kd / N_SIMS,
        }
        table2_rows.append(row)

t2 = pd.DataFrame(table2_rows)

# Print Table 2
for n_units, sigma in n_configs:
    print(f"\n  n={n_units}, σ={sigma}:")
    print(f"  {'Pattern':20s} {'Raw Wald':>10s} {'KS Wald':>10s} {'KS Deriv':>10s}")
    print("  " + "-" * 55)
    sub = t2[(t2["n"] == n_units) & (t2["sigma"] == sigma)]
    for _, r in sub.iterrows():
        label = r["pattern"]
        if label == "zero": label += " (size)"
        print(f"  {label:20s} {r['rej_raw']:10.3f} {r['rej_kalman_wald']:10.3f} "
              f"{r['rej_kalman_deriv']:10.3f}")


# --- Table 3: Sensitivity to Q ---
print("\n" + "=" * 70)
print("TABLE 3: Sensitivity to Process Noise Q")
print("=" * 70)

q_configs = [
    (0.0005, 0.0002),
    (0.001, 0.0005),
    (0.002, 0.001),
    (0.005, 0.002),
    (0.01, 0.005),
    (0.02, 0.01),
]

table3_seeds = master_rng.randint(0, 2**31, size=N_SIMS)

table3_rows = []
for ql, qs in q_configs:
    sm = KalmanEventStudySmoother(ql, qs)
    for pattern in ["zero", "gradual", "anticipation"]:
        mse_l, mse_d = [], []
        for sim in range(N_SIMS):
            rng = np.random.RandomState(table3_seeds[sim])
            bt = true_effect(T_PRE, T_POST, pattern)
            bh, se = simulate_beta_hat(bt, 200, 0.5, 1.0, rng)
            lvl, slp, _, _, _ = sm.smooth(bh, se)
            mse_l.append(np.mean((lvl - bt)**2))
            mse_d.append(np.mean((slp - np.gradient(bt))**2))
        table3_rows.append({
            "q_level": ql, "q_slope": qs, "pattern": pattern,
            "level_mse": np.mean(mse_l), "deriv_mse": np.mean(mse_d)
        })

t3 = pd.DataFrame(table3_rows)
print(f"\n  {'q_level':>8s} {'q_slope':>8s} │ {'null(L)':>8s} {'grad(L)':>8s} {'antic(L)':>8s} │ "
      f"{'null(D)':>8s} {'grad(D)':>8s} {'antic(D)':>8s}")
print("  " + "─" * 80)
for ql, qs in q_configs:
    sub = t3[(t3["q_level"] == ql) & (t3["q_slope"] == qs)]
    vals_l = [sub[sub["pattern"]==p]["level_mse"].values[0] for p in ["zero","gradual","anticipation"]]
    vals_d = [sub[sub["pattern"]==p]["deriv_mse"].values[0] for p in ["zero","gradual","anticipation"]]
    print(f"  {ql:8.4f} {qs:8.4f} │ {vals_l[0]:8.5f} {vals_l[1]:8.5f} {vals_l[2]:8.5f} │ "
          f"{vals_d[0]:8.5f} {vals_d[1]:8.5f} {vals_d[2]:8.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# --- Figure 1: Illustrative example ---
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
smoother = KalmanEventStudySmoother(q_level=0.002, q_slope=0.001)
fig1_rng = np.random.RandomState(MASTER_SEED)

for i, pattern in enumerate(["gradual", "immediate", "fadeout", "anticipation"]):
    rng = np.random.RandomState(fig1_rng.randint(0, 2**31))
    bt = true_effect(T_PRE, T_POST, pattern)
    bh, se = simulate_beta_hat(bt, 200, 0.5, 1.0, rng)
    lvl, slp, lse, sse, _ = smoother.smooth(bh, se)
    ll, ld = loess_smooth(bh)
    t = np.arange(len(bh)) - T_PRE
    td = np.gradient(bt)

    # Level
    ax = axes[0, i]
    ax.fill_between(t, bh-1.96*se, bh+1.96*se, alpha=0.12, color=COLORS["raw"])
    ax.scatter(t, bh, s=18, color=COLORS["raw"], alpha=0.5, zorder=3, edgecolors="none")
    ax.plot(t, bt, color=COLORS["true"], ls="--", lw=2, label="True $\\beta_t$")
    ax.plot(t, ll, color=COLORS["loess"], lw=1.5, alpha=0.9, label="LOESS")
    ax.plot(t, lvl, color=COLORS["kalman"], lw=2.5, zorder=4, label="Kalman (RTS)")
    ax.fill_between(t, lvl-1.96*lse, lvl+1.96*lse, alpha=0.2, color=COLORS["kalman"])
    ax.axvline(0, color="#CC79A7", ls=":", lw=1.5, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(pattern.capitalize(), fontweight="bold")
    if i == 0:
        ax.set_ylabel("Treatment Effect $\\beta_t$")
        ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Derivative
    ax = axes[1, i]
    ax.scatter(t, np.gradient(bh), s=12, color=COLORS["raw"], alpha=0.4, edgecolors="none")
    ax.plot(t, td, color=COLORS["true"], ls="--", lw=2, label="True $\\Delta\\beta_t$")
    ax.plot(t, ld, color=COLORS["loess"], lw=1.5, alpha=0.9, label="LOESS")
    ax.plot(t, slp, color=COLORS["kalman"], lw=2.5, zorder=4, label="Kalman slope")
    ax.fill_between(t, slp-1.96*sse, slp+1.96*sse, alpha=0.2, color=COLORS["kalman"])
    ax.axvline(0, color="#CC79A7", ls=":", lw=1.5, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Periods relative to treatment")
    if i == 0:
        ax.set_ylabel("$\\Delta\\beta_t$")
        ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

plt.tight_layout()
fig1_path = os.path.join(FIGS_DIR, "paper_fig1.png")
plt.savefig(fig1_path, bbox_inches="tight")
plt.close()
print(f"\nSaved {fig1_path}")


# --- Figure 2: MSE reduction summary ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sub = t1[(t1["n"] == 200) & (t1["sigma"] == 1.0)]
x = np.arange(len(patterns))
w = 0.35

ax = axes[0]
ax.bar(x - w/2, sub["ks_level_pct"], w, label="Level", color=COLORS["kalman"], alpha=0.85)
ax.bar(x + w/2, sub["ks_deriv_pct"], w, label="Derivative", color=COLORS["kalman_deriv"], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([p.capitalize() for p in patterns])
ax.set_ylabel("% MSE Reduction (vs Raw)")
ax.set_title("(a) Baseline: $n=200$, $\\sigma=1.0$")
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.3, axis="y", linestyle="-", linewidth=0.5)
for xi, v1, v2 in zip(x, sub["ks_level_pct"], sub["ks_deriv_pct"]):
    ax.text(xi-w/2, v1+1.5, f"{v1:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(xi+w/2, v2+1.5, f"{v2:.0f}", ha="center", fontsize=9, fontweight="bold")

# Across configs
ax = axes[1]
configs_labels = [f"$n$={n}, $\\sigma$={s}" for n,s in n_configs]
bar_colors = [COLORS["kalman"], COLORS["kalman_deriv"]]
for j, pat in enumerate(["gradual", "anticipation"]):
    vals = [t1[(t1["n"]==n) & (t1["sigma"]==s) & (t1["pattern"]==pat)]["ks_level_pct"].values[0]
            for n,s in n_configs]
    ax.bar(np.arange(len(n_configs)) + j*0.3 - 0.15, vals, 0.28,
           label=pat.capitalize(), color=bar_colors[j], alpha=0.85)
ax.set_xticks(range(len(n_configs)))
ax.set_xticklabels(configs_labels)
ax.set_ylabel("% Level MSE Reduction")
ax.set_title("(b) Across Sample Configurations")
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.3, axis="y", linestyle="-", linewidth=0.5)

plt.tight_layout()
fig2_path = os.path.join(FIGS_DIR, "paper_fig2.png")
plt.savefig(fig2_path, bbox_inches="tight")
plt.close()
print(f"Saved {fig2_path}")


# --- Figure 3: Size and Power ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sub = t2[(t2["n"] == 200) & (t2["sigma"] == 1.0)]
test_pats_plot = ["zero", "small_pretrend", "anticipation"]
x = np.arange(len(test_pats_plot))
w = 0.25

ax = axes[0]
test_colors = [COLORS["raw_wald"], COLORS["kalman_wald"], COLORS["kalman_deriv"]]
for j, (col, label) in enumerate([
    ("rej_raw", "Raw Wald"),
    ("rej_kalman_wald", "Kalman Wald"),
    ("rej_kalman_deriv", "Kalman Deriv"),
]):
    vals = [sub[sub["pattern"]==p][col].values[0] for p in test_pats_plot]
    ax.bar(x + (j-1)*w, vals, w, label=label, color=test_colors[j], alpha=0.85)

ax.axhline(0.05, color="black", ls="--", lw=1.2, label="Nominal 5%")
ax.set_xticks(x)
ax.set_xticklabels(["Null\n(size)", "Small\npretrend", "Anticipation\n(power)"])
ax.set_ylabel("Rejection Rate")
ax.set_title("(a) Bootstrap-Calibrated: $n=200$, $\\sigma=1.0$")
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.3, axis="y", linestyle="-", linewidth=0.5)

# Power across noise levels
ax = axes[1]
for j, (col, label) in enumerate([
    ("rej_raw", "Raw Wald"),
    ("rej_kalman_wald", "Kalman Wald"),
    ("rej_kalman_deriv", "Kalman Deriv"),
]):
    vals = []
    for n,s in n_configs:
        v = t2[(t2["n"]==n) & (t2["sigma"]==s) & (t2["pattern"]=="anticipation")][col].values[0]
        vals.append(v)
    ax.plot(range(len(n_configs)), vals, "o-", label=label, color=test_colors[j], lw=2.5, markersize=9)

ax.set_xticks(range(len(n_configs)))
ax.set_xticklabels(configs_labels)
ax.set_ylabel("Power (anticipation pattern)")
ax.set_title("(b) Power Across Configurations")
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

plt.tight_layout()
fig3_path = os.path.join(FIGS_DIR, "paper_fig3.png")
plt.savefig(fig3_path, bbox_inches="tight")
plt.close()
print(f"Saved {fig3_path}")


# --- Figure 4: Sensitivity to Q ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pattern_colors = {"zero": COLORS["raw"], "gradual": COLORS["kalman"], "anticipation": COLORS["kalman_deriv"]}
for k, metric in enumerate(["level_mse", "deriv_mse"]):
    ax = axes[k]
    for pat in ["zero", "gradual", "anticipation"]:
        sub = t3[t3["pattern"] == pat]
        ax.plot(sub["q_level"], sub[metric], "o-", label=pat.capitalize(),
                color=pattern_colors[pat], lw=2.5, markersize=8)
    ax.set_xlabel("$q_{\\mathrm{level}}$")
    ax.set_ylabel("MSE")
    ax.set_title(f"({'a' if k==0 else 'b'}) {'Level' if k==0 else 'Derivative'} MSE vs $q_{{\\mathrm{{level}}}}$")
    ax.set_xscale("log")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

plt.tight_layout()
fig4_path = os.path.join(FIGS_DIR, "paper_fig4.png")
plt.savefig(fig4_path, bbox_inches="tight")
plt.close()
print(f"Saved {fig4_path}")


# --- Save tables as CSV ---
t1.to_csv(os.path.join(TABS_DIR, "table1_mse.csv"), index=False)
t2.to_csv(os.path.join(TABS_DIR, "table2_size_power.csv"), index=False)
t3.to_csv(os.path.join(TABS_DIR, "table3_sensitivity.csv"), index=False)
print(f"\nSaved CSV tables to {TABS_DIR}")

print("\n✓ All simulations complete.")