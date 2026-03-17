"""
Monte Carlo Simulations for:
"Smoother Trends, Sharper Tests: Optimal Filtering of Event Study Estimates"

Produces all tables and figures for the paper:
  Table 1: Level and Derivative MSE — Kalman vs competitive baselines
  Table 2: Bootstrap-calibrated size and power — all tests
  Table 3: Sensitivity to process noise Q
  Figure 1: Illustrative examples (4 DGPs × level + derivative)
  Figure 2: MSE reduction summary
  Figure 3: Size and power across methods
  Figure 4: Sensitivity to Q
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import warnings, os
warnings.filterwarnings("ignore")

OUT_FIG = "../figs"
OUT_TABS = "../tabs"
os.makedirs(OUT_FIG, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# KALMAN SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class KalmanSmoother:
    """
    Local linear trend state-space model + RTS smoother.

    State: [beta_t, Delta_beta_t]
    Transition: F = [[1,1],[0,1]], process noise Q = diag(q_level, q_slope)
    Observation: beta_hat_t = [1,0] x_t + v_t, v_t ~ N(0, sigma_t^2)
    """
    def __init__(self, q_level=0.002, q_slope=0.001):
        self.F = np.array([[1., 1.], [0., 1.]])
        self.H = np.array([[1., 0.]])
        self.Q = np.diag([q_level, q_slope])

    def smooth(self, bhat, se):
        T = len(bhat)
        F, H, Q = self.F, self.H, self.Q
        x = np.array([bhat[0], 0.0])
        P = np.diag([se[0]**2, Q[1, 1] * 10])

        xf, Pf, xp, Pp = [], [], [], []
        for t in range(T):
            if t > 0:
                xpr = F @ x; Ppr = F @ P @ F.T + Q
            else:
                xpr = x.copy(); Ppr = P.copy()
            xp.append(xpr.copy()); Pp.append(Ppr.copy())
            R = np.array([[se[t]**2]])
            S = H @ Ppr @ H.T + R
            K = Ppr @ H.T / S[0, 0]
            x = xpr + K.flatten() * (bhat[t] - (H @ xpr)[0])
            P = (np.eye(2) - K @ H) @ Ppr
            xf.append(x.copy()); Pf.append(P.copy())

        xs = [xf[-1].copy()]; Ps = [Pf[-1].copy()]
        for t in range(T - 2, -1, -1):
            try:
                C = Pf[t] @ F.T @ np.linalg.inv(Pp[t + 1])
            except np.linalg.LinAlgError:
                C = Pf[t] @ F.T @ np.linalg.pinv(Pp[t + 1])
            xs.insert(0, xf[t] + C @ (xs[0] - xp[t + 1]))
            Ps.insert(0, Pf[t] + C @ (Ps[0] - Pp[t + 1]) @ C.T)

        lvl = np.array([x[0] for x in xs])
        slp = np.array([x[1] for x in xs])
        lse = np.array([np.sqrt(max(P[0, 0], 1e-12)) for P in Ps])
        sse = np.array([np.sqrt(max(P[1, 1], 1e-12)) for P in Ps])
        return lvl, slp, lse, sse, Ps


# ══════════════════════════════════════════════════════════════════════════════
# COMPETITIVE BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def loess_smooth(bhat, span=0.3):
    T = len(bhat)
    w = max(5, int(T * span))
    if w % 2 == 0:
        w += 1
    w = min(w, T - 2)
    if w < 5:
        return bhat.copy(), np.gradient(bhat)
    return savgol_filter(bhat, w, 3), savgol_filter(bhat, w, 3, deriv=1)


def parametric_linear(bhat, se, T_pre):
    """WLS fit of beta_t = a + b*t on pre-treatment, extrapolated."""
    T = len(bhat)
    t = np.arange(T) - T_pre
    pre_t, pre_b, pre_w = t[:T_pre], bhat[:T_pre], 1.0 / se[:T_pre]**2
    sw = pre_w.sum(); swt = (pre_w * pre_t).sum()
    swtt = (pre_w * pre_t**2).sum(); swy = (pre_w * pre_b).sum()
    swty = (pre_w * pre_t * pre_b).sum()
    denom = sw * swtt - swt**2
    if abs(denom) < 1e-15:
        return bhat.copy(), np.gradient(bhat)
    b = (sw * swty - swt * swy) / denom
    a = (swy - b * swt) / sw
    return a + b * t, np.full(T, b)


def james_stein(bhat, se, T_pre):
    """James-Stein shrinkage of pre-treatment beta_hat toward zero."""
    T = len(bhat)
    p = T_pre
    if p <= 2:
        return bhat.copy(), np.gradient(bhat)
    sigma2 = np.mean(se[:T_pre]**2)
    sum_b2 = np.sum(bhat[:T_pre]**2)
    B = max(0, 1 - (p - 2) * sigma2 / sum_b2)
    shrunk = bhat.copy()
    shrunk[:T_pre] = B * bhat[:T_pre]
    return shrunk, np.gradient(shrunk)


def empirical_bayes(bhat, se, T_pre):
    """Normal EB: beta_t ~ N(0, tau^2), posterior mean shrinkage."""
    T = len(bhat)
    tau2_pre = max(0, np.var(bhat[:T_pre]) - np.mean(se[:T_pre]**2))
    tau2_all = max(0, np.var(bhat) - np.mean(se**2))
    shrunk = bhat.copy()
    for t in range(T_pre):
        w = tau2_pre / (tau2_pre + se[t]**2) if tau2_pre > 0 else 0
        shrunk[t] = w * bhat[t]
    for t in range(T_pre, T):
        w = tau2_all / (tau2_all + se[t]**2) if tau2_all > 0 else 0
        shrunk[t] = w * bhat[t]
    return shrunk, np.gradient(shrunk)


# ══════════════════════════════════════════════════════════════════════════════
# DGP
# ══════════════════════════════════════════════════════════════════════════════

PATTERNS = ["no effect", "gradual", "immediate", "fadeout", "anticipation"]

def true_effect(T_pre, T_post, pattern):
    T = T_pre + T_post
    beta = np.zeros(T)
    t0 = T_pre
    if pattern == "gradual":
        for i in range(t0, T):
            beta[i] = 0.5 * (1 - np.exp(-0.3 * (i - t0 + 1)))
    elif pattern == "immediate":
        beta[t0:] = 0.4
    elif pattern == "fadeout":
        for i in range(t0, T):
            beta[i] = 0.5 * np.exp(-0.1 * (i - t0))
    elif pattern == "anticipation":
        for i in range(t0 - 3, t0):
            beta[i] = 0.05 * (i - (t0 - 4))
        for i in range(t0, T):
            beta[i] = 0.15 + 0.3 * (1 - np.exp(-0.3 * (i - t0 + 1)))
    elif pattern == "small pretrend":
        for i in range(t0 - 4, t0):
            beta[i] = 0.015 * (i - (t0 - 5))
        for i in range(t0, T):
            beta[i] = 0.06 + 0.3 * (1 - np.exp(-0.3 * (i - t0 + 1)))
    # "no effect" stays all zeros
    return beta


def simulate_bhat(beta_true, n_units, sigma, rng):
    T = len(beta_true)
    n_t = int(n_units * 0.5)
    n_c = n_units - n_t
    base_se = sigma * np.sqrt(1.0 / n_t + 1.0 / n_c)
    sc = np.ones(T)
    sc[:3] *= 1.5
    sc[-3:] *= 1.3
    sc *= (1 + 0.1 * rng.randn(T)).clip(0.5, 2.0)
    se = base_se * sc
    return beta_true + rng.randn(T) * se, se


# ══════════════════════════════════════════════════════════════════════════════
# TEST STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def test_raw_wald(bhat, se, T_pre, _ks):
    return np.sum((bhat[:T_pre] / se[:T_pre])**2)


def test_parametric_slope(bhat, se, T_pre, _ks):
    """t-test on the WLS slope coefficient."""
    T = len(bhat)
    t = np.arange(T) - T_pre
    pre_t, pre_b, pre_w = t[:T_pre], bhat[:T_pre], 1.0 / se[:T_pre]**2
    sw = pre_w.sum(); swt = (pre_w * pre_t).sum()
    swtt = (pre_w * pre_t**2).sum(); swy = (pre_w * pre_b).sum()
    swty = (pre_w * pre_t * pre_b).sum()
    denom = sw * swtt - swt**2
    if abs(denom) < 1e-15:
        return 0.0
    b = (sw * swty - swt * swy) / denom
    a = (swy - b * swt) / sw
    resid = pre_b - (a + b * pre_t)
    sig2 = np.sum(pre_w * resid**2) / max(T_pre - 2, 1)
    se_b = np.sqrt(sig2 / (swtt - swt**2 / sw))
    return (b / se_b)**2 if se_b > 1e-12 else 0.0


def test_eb_wald(bhat, se, T_pre, _ks):
    shrunk, _ = empirical_bayes(bhat, se, T_pre)
    return np.sum((shrunk[:T_pre] / se[:T_pre])**2)


def test_kalman_wald(bhat, se, T_pre, ks):
    lvl, _, lse, _, _ = ks.smooth(bhat, se)
    return np.sum((lvl[:T_pre] / lse[:T_pre])**2)


def test_kalman_deriv(bhat, se, T_pre, ks):
    _, slp, _, sse, _ = ks.smooth(bhat, se)
    return np.sum((slp[:T_pre] / sse[:T_pre])**2)


TESTS = {
    "Raw Wald": test_raw_wald,
    "Parametric slope": test_parametric_slope,
    "EB Wald": test_eb_wald,
    "Kalman Wald": test_kalman_wald,
    "Kalman deriv": test_kalman_deriv,
}


def bootstrap_cv(test_fn, T_pre, T_post, n_units, sigma, ks, B=499, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.RandomState(54321)
    T = T_pre + T_post
    beta0 = np.zeros(T)
    stats = []
    for _ in range(B):
        bh, se = simulate_bhat(beta0, n_units, sigma, rng)
        stats.append(test_fn(bh, se, T_pre, ks))
    return np.percentile(stats, 100 * (1 - alpha))


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

T_PRE, T_POST = 12, 12
N_SIMS = 1000
KS = KalmanSmoother(0.002, 0.001)

CONFIGS = [
    (200, 1.0, "$N{=}200, \\sigma{=}1$"),
    (500, 1.0, "$N{=}500, \\sigma{=}1$"),
    (200, 2.0, "$N{=}200, \\sigma{=}2$"),
    (100, 1.0, "$N{=}100, \\sigma{=}1$"),
]

METHOD_NAMES = ["Raw", "Parametric linear", "James-Stein", "Empirical Bayes",
                "LOESS", "Kalman smoother"]


def apply_method(mname, bhat, se, T_pre, ks):
    if mname == "Raw":
        return bhat.copy(), np.gradient(bhat)
    elif mname == "Parametric linear":
        return parametric_linear(bhat, se, T_pre)
    elif mname == "James-Stein":
        return james_stein(bhat, se, T_pre)
    elif mname == "Empirical Bayes":
        return empirical_bayes(bhat, se, T_pre)
    elif mname == "LOESS":
        return loess_smooth(bhat)
    elif mname == "Kalman smoother":
        lvl, slp, _, _, _ = ks.smooth(bhat, se)
        return lvl, slp


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: MSE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TABLE 1: Level and Derivative MSE")
print("=" * 70)

t1_rows = []
for n, sigma, clabel in CONFIGS:
    for pattern in PATTERNS:
        bt = true_effect(T_PRE, T_POST, pattern)
        td = np.gradient(bt)
        mse_l = {m: [] for m in METHOD_NAMES}
        mse_d = {m: [] for m in METHOD_NAMES}

        for sim in range(N_SIMS):
            rng = np.random.RandomState(sim)
            bh, se = simulate_bhat(bt, n, sigma, rng)
            for m in METHOD_NAMES:
                lvl, drv = apply_method(m, bh, se, T_PRE, KS)
                mse_l[m].append(np.mean((lvl - bt)**2))
                mse_d[m].append(np.mean((drv - td)**2))

        for m in METHOD_NAMES:
            t1_rows.append({
                "n": n, "sigma": sigma, "config": clabel, "pattern": pattern,
                "method": m,
                "level_mse": np.mean(mse_l[m]),
                "deriv_mse": np.mean(mse_d[m]),
            })

    print(f"  Config ({n}, {sigma}) done.")

t1 = pd.DataFrame(t1_rows)
t1.to_csv(f"{OUT_TABS}/table1_mse.csv", index=False)

# --- Generate LaTeX for Table 1 ---
def write_table1_tex(df, path):
    sub = df[(df["n"] == 200) & (df["sigma"] == 1.0)]
    methods = ["Raw", "Parametric linear", "James-Stein", "Empirical Bayes",
               "LOESS", "Kalman smoother"]
    short = {"Raw": "Raw", "Parametric linear": "Param.\\ lin.",
             "James-Stein": "J--S", "Empirical Bayes": "EB",
             "LOESS": "LOESS", "Kalman smoother": "Kalman"}
    pats = ["no effect", "gradual", "immediate", "fadeout", "anticipation"]
    pat_label = {"no effect": "No effect", "gradual": "Gradual",
                 "immediate": "Immediate", "fadeout": "Fadeout",
                 "anticipation": "Anticipation"}

    lines = []
    lines.append(r"\begin{tabular}{l rrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{6}{c}{Level MSE ($\times 10^3$)} \\")
    lines.append(r"\cmidrule(lr){2-7}")
    header = "Pattern & " + " & ".join(short[m] for m in methods) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for p in pats:
        vals = []
        for m in methods:
            r = sub[(sub["method"] == m) & (sub["pattern"] == p)]
            vals.append(f"{r['level_mse'].values[0]*1000:.2f}")
        lines.append(f"{pat_label[p]:15s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"& \multicolumn{6}{c}{Derivative MSE ($\times 10^3$)} \\")
    lines.append(r"\cmidrule(lr){2-7}")
    lines.append(header)
    lines.append(r"\midrule")
    for p in pats:
        vals = []
        for m in methods:
            r = sub[(sub["method"] == m) & (sub["pattern"] == p)]
            vals.append(f"{r['deriv_mse'].values[0]*1000:.2f}")
        lines.append(f"{pat_label[p]:15s} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

write_table1_tex(t1, f"{OUT_TABS}/table1_mse.tex")
print("  Wrote table1_mse.tex")

# Print summary for baseline
sub = t1[(t1["n"] == 200) & (t1["sigma"] == 1.0)]
print("\n  Baseline (N=200, sigma=1):")
for m in METHOD_NAMES:
    row_g = sub[(sub["method"] == m) & (sub["pattern"] == "gradual")]
    print(f"    {m:22s}: level={row_g['level_mse'].values[0]*1000:.2f}  "
          f"deriv={row_g['deriv_mse'].values[0]*1000:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: SIZE AND POWER
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TABLE 2: Bootstrap-Calibrated Size and Power")
print("=" * 70)

TEST_PATTERNS = ["no effect", "small pretrend", "anticipation", "gradual", "immediate"]

# Bootstrap CVs
print("  Computing bootstrap critical values...")
boot_rng = np.random.RandomState(54321)
cv_cache = {}
for n, sigma, clabel in CONFIGS:
    cv_cache[(n, sigma)] = {}
    for tname, tfn in TESTS.items():
        cv = bootstrap_cv(tfn, T_PRE, T_POST, n, sigma, KS, B=499, rng=boot_rng)
        cv_cache[(n, sigma)][tname] = cv
    print(f"    ({n}, {sigma}) done.")

# Run tests
print("  Running size/power simulations...")
t2_rows = []
for n, sigma, clabel in CONFIGS:
    cvs = cv_cache[(n, sigma)]
    for pattern in TEST_PATTERNS:
        bt = true_effect(T_PRE, T_POST, pattern)
        rej = {t: 0 for t in TESTS}
        for sim in range(N_SIMS):
            rng = np.random.RandomState(sim + 100000)
            bh, se = simulate_bhat(bt, n, sigma, rng)
            for tname, tfn in TESTS.items():
                if tfn(bh, se, T_PRE, KS) > cvs[tname]:
                    rej[tname] += 1
        for tname in TESTS:
            t2_rows.append({
                "n": n, "sigma": sigma, "config": clabel,
                "pattern": pattern, "test": tname,
                "rejection": rej[tname] / N_SIMS,
            })
    print(f"    ({n}, {sigma}) done.")

t2 = pd.DataFrame(t2_rows)
t2.to_csv(f"{OUT_TABS}/table2_size_power.csv", index=False)

# --- Generate LaTeX for Table 2 ---
def write_table2_tex(df, path, configs):
    test_names = list(TESTS.keys())
    test_short = {"Raw Wald": "Raw Wald", "Parametric slope": "Param.\\ slope",
                  "EB Wald": "EB Wald", "Kalman Wald": "Kalman Wald",
                  "Kalman deriv": "Kalman deriv"}
    pats = TEST_PATTERNS
    pat_short = {"no effect": "No effect", "small pretrend": "Sm.\\ pretrend",
                 "anticipation": "Anticipation", "gradual": "Gradual",
                 "immediate": "Immediate"}

    lines = []
    lines.append(r"\begin{tabular}{l l " + "r" * len(pats) + "}")
    lines.append(r"\toprule")
    header = r"$(N, \sigma)$ & Test & " + " & ".join(pat_short[p] for p in pats) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for ci, (n, sigma, clabel) in enumerate(configs):
        sub = df[(df["n"] == n) & (df["sigma"] == sigma)]
        for ti, tname in enumerate(test_names):
            prefix = f"$({n}, {sigma:.0f})$" if ti == 0 else ""
            vals = []
            for p in pats:
                r = sub[(sub["test"] == tname) & (sub["pattern"] == p)]
                vals.append(f"{r['rejection'].values[0]:.3f}")
            lines.append(f"{prefix} & {test_short[tname]} & " + " & ".join(vals) + r" \\")
        if ci < len(configs) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

write_table2_tex(t2, f"{OUT_TABS}/table2_size_power.tex", CONFIGS)
print("  Wrote table2_size_power.tex")

# Print baseline
sub = t2[(t2["n"] == 200) & (t2["sigma"] == 1.0)]
print("\n  Baseline (N=200, sigma=1):")
print(f"  {'Test':20s}  {'no effect':>10s}  {'sm pretr':>10s}  {'antic':>10s}  {'grad':>10s}  {'immed':>10s}")
for tname in TESTS:
    vals = [sub[(sub["test"] == tname) & (sub["pattern"] == p)]["rejection"].values[0]
            for p in TEST_PATTERNS]
    print(f"  {tname:20s}  {vals[0]:10.3f}  {vals[1]:10.3f}  {vals[2]:10.3f}  "
          f"{vals[3]:10.3f}  {vals[4]:10.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3: SENSITIVITY TO Q
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TABLE 3: Sensitivity to Q")
print("=" * 70)

Q_CONFIGS = [(0.0005, 0.0002), (0.001, 0.0005), (0.002, 0.001),
             (0.005, 0.002), (0.01, 0.005), (0.02, 0.01)]

t3_rows = []
for ql, qs in Q_CONFIGS:
    sm = KalmanSmoother(ql, qs)
    for pattern in ["no effect", "gradual", "anticipation"]:
        bt = true_effect(T_PRE, T_POST, pattern)
        td = np.gradient(bt)
        ml, md = [], []
        for sim in range(N_SIMS):
            rng = np.random.RandomState(sim)
            bh, se = simulate_bhat(bt, 200, 1.0, rng)
            lvl, slp, _, _, _ = sm.smooth(bh, se)
            ml.append(np.mean((lvl - bt)**2))
            md.append(np.mean((slp - td)**2))
        t3_rows.append({
            "q_level": ql, "q_slope": qs, "pattern": pattern,
            "level_mse": np.mean(ml), "deriv_mse": np.mean(md),
        })

t3 = pd.DataFrame(t3_rows)
t3.to_csv(f"{OUT_TABS}/table3_sensitivity.csv", index=False)

# --- Generate LaTeX for Table 3 ---
def write_table3_tex(df, path):
    lines = []
    lines.append(r"\begin{tabular}{rr rr}")
    lines.append(r"\toprule")
    lines.append(r"$q_\ell$ & $q_s$ & Level MSE ($\times 10^3$) & Deriv MSE ($\times 10^3$) \\")
    lines.append(r"\midrule")
    for _, row in df[df["pattern"] == "gradual"].iterrows():
        lines.append(f"{row['q_level']:.4f} & {row['q_slope']:.4f} & "
                     f"{row['level_mse']*1000:.2f} & {row['deriv_mse']*1000:.2f}" + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

write_table3_tex(t3, f"{OUT_TABS}/table3_sensitivity.tex")
print("  Wrote table3_sensitivity.tex")
print("  Done.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# --- Figure 1: Illustrative examples ---
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for i, pattern in enumerate(["gradual", "immediate", "fadeout", "anticipation"]):
    rng = np.random.RandomState(42)
    bt = true_effect(T_PRE, T_POST, pattern)
    bh, se = simulate_bhat(bt, 200, 1.0, rng)
    lvl, slp, lse, sse, _ = KS.smooth(bh, se)
    ll, ld = loess_smooth(bh)
    t = np.arange(len(bh)) - T_PRE
    td = np.gradient(bt)

    ax = axes[0, i]
    ax.fill_between(t, bh - 1.96 * se, bh + 1.96 * se, alpha=0.12, color="gray")
    ax.scatter(t, bh, s=15, color="gray", alpha=0.5, zorder=3)
    ax.plot(t, bt, "k--", lw=2, label="True $\\beta_t$")
    ax.plot(t, ll, color="#f28e2b", lw=1.5, alpha=0.8, label="LOESS")
    ax.plot(t, lvl, color="#4e79a7", lw=2, zorder=4, label="Kalman (RTS)")
    ax.fill_between(t, lvl - 1.96 * lse, lvl + 1.96 * lse, alpha=0.15, color="#4e79a7")
    ax.axvline(0, color="red", ls=":", lw=1, alpha=0.5)
    ax.axhline(0, color="black", lw=0.3)
    ax.set_title(pattern, fontsize=12, fontweight="bold")
    if i == 0:
        ax.set_ylabel("Treatment Effect $\\beta_t$")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    ax = axes[1, i]
    ax.scatter(t, np.gradient(bh), s=10, color="gray", alpha=0.4)
    ax.plot(t, td, "k--", lw=2, label="True $\\Delta\\beta_t$")
    ax.plot(t, ld, color="#f28e2b", lw=1.5, alpha=0.8, label="LOESS")
    ax.plot(t, slp, color="#4e79a7", lw=2, zorder=4, label="Kalman slope")
    ax.fill_between(t, slp - 1.96 * sse, slp + 1.96 * sse, alpha=0.15, color="#4e79a7")
    ax.axvline(0, color="red", ls=":", lw=1, alpha=0.5)
    ax.axhline(0, color="black", lw=0.3)
    ax.set_xlabel("Periods relative to treatment")
    if i == 0:
        ax.set_ylabel("$\\Delta\\beta_t$")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

plt.suptitle("Figure 1: Event Study Estimates --- Raw, LOESS, and Kalman Smoother",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/paper_fig1.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved paper_fig1.png")


# --- Figure 2: MSE across methods ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
colors = {"Raw": "#999999", "Parametric linear": "#f28e2b",
          "James-Stein": "#59a14f", "Empirical Bayes": "#b07aa1",
          "LOESS": "#edc949", "Kalman smoother": "#4e79a7"}

# (a) Level MSE by method, gradual pattern, across configs
ax = axes[0]
base_methods = ["Raw", "James-Stein", "Empirical Bayes", "LOESS", "Kalman smoother"]
x = np.arange(len(CONFIGS))
w = 0.15
for j, m in enumerate(base_methods):
    vals = [t1[(t1["n"] == c[0]) & (t1["sigma"] == c[1]) &
               (t1["method"] == m) & (t1["pattern"] == "gradual")]["level_mse"].values[0] * 1000
            for c in CONFIGS]
    ax.bar(x + (j - 2) * w, vals, w, label=m, color=colors[m], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["N=200\n$\\sigma$=1", "N=500\n$\\sigma$=1",
                     "N=200\n$\\sigma$=2", "N=100\n$\\sigma$=1"], fontsize=9)
ax.set_ylabel("Level MSE ($\\times 10^3$)")
ax.set_title("(a) Level MSE: gradual pattern")
ax.legend(fontsize=7, loc="upper left")
ax.grid(True, alpha=0.2, axis="y")

# (b) Derivative MSE by method, baseline
ax = axes[1]
sub = t1[(t1["n"] == 200) & (t1["sigma"] == 1.0) & (t1["pattern"] == "gradual")]
mnames = base_methods
vals = [sub[sub["method"] == m]["deriv_mse"].values[0] * 1000 for m in mnames]
bars = ax.bar(range(len(mnames)), vals, color=[colors[m] for m in mnames], alpha=0.8)
ax.set_xticks(range(len(mnames)))
ax.set_xticklabels(mnames, fontsize=8, rotation=25, ha="right")
ax.set_ylabel("Derivative MSE ($\\times 10^3$)")
ax.set_title("(b) Derivative MSE: gradual, $N{=}200$, $\\sigma{=}1$")
ax.grid(True, alpha=0.2, axis="y")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f}",
            ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_FIG}/paper_fig2.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved paper_fig2.png")


# --- Figure 3: Size and power ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
test_colors = {"Raw Wald": "#999999", "Parametric slope": "#f28e2b",
               "EB Wald": "#b07aa1", "Kalman Wald": "#76b7b2",
               "Kalman deriv": "#4e79a7"}

# (a) Baseline: size and power
ax = axes[0]
sub = t2[(t2["n"] == 200) & (t2["sigma"] == 1.0)]
plot_patterns = ["no effect", "small pretrend", "anticipation"]
x = np.arange(len(plot_patterns))
w = 0.15
for j, tname in enumerate(TESTS):
    vals = [sub[(sub["test"] == tname) & (sub["pattern"] == p)]["rejection"].values[0]
            for p in plot_patterns]
    ax.bar(x + (j - 2) * w, vals, w, label=tname,
           color=test_colors.get(tname, "gray"), alpha=0.8)
ax.axhline(0.05, color="black", ls="--", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["No effect\n(size)", "Small\npretrend", "Anticipation\n(power)"])
ax.set_ylabel("Rejection Rate")
ax.set_title("(a) $N{=}200$, $\\sigma{=}1$, bootstrap-calibrated 5\\%")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2, axis="y")

# (b) Power vs anticipation across configs
ax = axes[1]
for tname in TESTS:
    vals = [t2[(t2["n"] == c[0]) & (t2["sigma"] == c[1]) &
               (t2["test"] == tname) & (t2["pattern"] == "anticipation")]["rejection"].values[0]
            for c in CONFIGS]
    ax.plot(range(len(CONFIGS)), vals, "o-", label=tname,
            color=test_colors.get(tname, "gray"), lw=2, markersize=7)
ax.set_xticks(range(len(CONFIGS)))
ax.set_xticklabels(["N=200\n$\\sigma$=1", "N=500\n$\\sigma$=1",
                     "N=200\n$\\sigma$=2", "N=100\n$\\sigma$=1"], fontsize=9)
ax.set_ylabel("Power (anticipation pattern)")
ax.set_title("(b) Power across configurations")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUT_FIG}/paper_fig3.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved paper_fig3.png")


# --- Figure 4: Sensitivity to Q ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for k, metric in enumerate(["level_mse", "deriv_mse"]):
    ax = axes[k]
    for pat in ["no effect", "gradual", "anticipation"]:
        sub = t3[t3["pattern"] == pat]
        ax.plot(sub["q_level"], sub[metric] * 1000, "o-", label=pat, lw=2, markersize=6)
    ax.set_xlabel("$q_{\\ell}$")
    ax.set_ylabel(f"{'Level' if k == 0 else 'Derivative'} MSE ($\\times 10^3$)")
    ax.set_title(f"({'a' if k == 0 else 'b'}) {'Level' if k == 0 else 'Derivative'} MSE")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUT_FIG}/paper_fig4.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved paper_fig4.png")

print("\nAll tables and figures saved.")
