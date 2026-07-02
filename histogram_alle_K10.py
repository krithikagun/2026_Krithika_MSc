import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path.home() / "Desktop" / "MASTER" / "horten0033"

K        = 10         # endret fra 3 til 10
N_SIGMA  = 5
Q_VALUES = [1, 2, 3, 4, 5]

X_MIN,  X_MAX  =   0,  10
Z_MIN,  Z_MAX  = -60, -40

X_MIN2, X_MAX2 = 20, 30
Z_MIN2, Z_MAX2 = -50, -30

C_GAS   = "crimson"
C_QUIET = "steelblue"

print("Laster data ...")
cube     = np.load(DATA_DIR / "cube.npy")
X        = np.load(DATA_DIR / "X.npy")
Z        = np.load(DATA_DIR / "Z.npy")
cube_lin = 10.0 ** (cube / 10.0)
n_beams, n_samp, n_pings = cube.shape
ping_nrs = np.arange(n_pings) + 200
print(f"  Kube: {cube.shape}  — pings {ping_nrs[0]}–{ping_nrs[-1]}")

def img(p):
    return cube_lin[:, :, p - 200]

def rolling_bg(p, K):
    idx = p - 200
    before = list(range(max(0, idx - K), idx))
    if not before:
        return None
    return np.nanmedian(cube_lin[:, :, before], axis=2)

def get_residuals(ping_list, box):
    res = []
    for p in ping_list:
        BG = rolling_bg(p, K)
        if BG is None:
            continue
        R = (img(p) - BG)[box]
        R = R[np.isfinite(R)]
        res.append(R)
    return np.concatenate(res) if res else np.array([])

box_gas   = (X >= X_MIN)  & (X <= X_MAX)  & (Z >= Z_MIN)  & (Z <= Z_MAX)
box_quiet = (X >= X_MIN2) & (X <= X_MAX2) & (Z >= Z_MIN2) & (Z <= Z_MAX2)

print(f"  Boks med gass:    [{X_MIN},{X_MAX}]×[{Z_MIN},{Z_MAX}]: {box_gas.sum()} piksler")
print(f"  Boks uten gass:   [{X_MIN2},{X_MAX2}]×[{Z_MIN2},{Z_MAX2}]: {box_quiet.sum()} piksler")

def plot_3panel(R_bg, R_gas, mu, sigma, T, label_bg, label_gas, title, filename):
    fq  = float(np.mean(R_bg  > T)) * 100
    fg  = float(np.mean(R_gas > T)) * 100
    rho = fg / max(fq, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=12)

    all_r = np.concatenate([R_bg, R_gas])
    edge  = np.percentile(np.abs(all_r), 99.5)
    bins  = np.linspace(-edge, edge, 120)

    ax1 = axes[0]
    ax1.hist(R_bg,  bins=bins, alpha=0.6, color=C_QUIET, density=True, label=label_bg)
    ax1.hist(R_gas, bins=bins, alpha=0.6, color=C_GAS,   density=True, label=label_gas)
    ax1.axvline(T,  color="black", lw=2,   label=f"μ+{N_SIGMA}σ = {T:.5f}")
    ax1.axvline(mu, color="green", lw=1.5, ls="--", label="μ")
    ax1.set_xlabel("residual (lineær)"); ax1.set_ylabel("tetthet")
    ax1.set_title(f"A — Lineær skala\n(gass: {fg:.2f}%  ikke-gass: {fq:.2f}%)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(R_bg,  bins=bins, alpha=0.6, color=C_QUIET, density=True, label=label_bg)
    ax2.hist(R_gas, bins=bins, alpha=0.6, color=C_GAS,   density=True, label=label_gas)
    for k, ls, lbl in [(1,":",f"μ+1σ"), (3,"--","μ+3σ"), (N_SIGMA,"-",f"μ+{N_SIGMA}σ (N)")]:
        ax2.axvline(mu + k*sigma, color="black", ls=ls, lw=1.2, label=lbl)
    ax2.axvline(mu, color="green", lw=1.5, ls="--", label="μ")
    ax2.set_yscale("log")
    ax2.set_xlabel("residual (lineær)"); ax2.set_ylabel("tetthet (log)")
    ax2.set_title("B — Log-skala: halen blir synlig")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    tail_bg  = R_bg [R_bg  > T]
    tail_gas = R_gas[R_gas > T]
    t_max = max(tail_bg.max()  if len(tail_bg)  else T,
                tail_gas.max() if len(tail_gas) else T)
    bt = np.linspace(T, t_max, 60)
    ax3.hist(tail_bg,  bins=bt, alpha=0.7, color=C_QUIET,
             label=f"{label_bg}: {fq:.2f}% ({len(tail_bg)} px)")
    ax3.hist(tail_gas, bins=bt, alpha=0.7, color=C_GAS,
             label=f"{label_gas}: {fg:.2f}% ({len(tail_gas)} px)")
    ax3.axvline(T, color="black", lw=2, label=f"μ+{N_SIGMA}σ")
    ax3.set_xlabel("residual (lineær)"); ax3.set_ylabel("antall piksler")
    ax3.set_title(f"C — Kun piksler OVER terskelen\nρ = {rho:.1f}×")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Lagret: {filename}")
    print(f"    gass={fg:.2f}%  ikke-gass={fq:.2f}%  ρ={rho:.1f}×")
    return fq, fg, rho

# Fig 1
print("\n--- Fig 1: ping 280 vs ping 321 ---")
R_280 = (img(280) - rolling_bg(280, K))[box_gas]; R_280 = R_280[np.isfinite(R_280)]
R_321 = (img(321) - rolling_bg(321, K))[box_gas]; R_321 = R_321[np.isfinite(R_321)]
mu1 = float(np.mean(R_280)); sig1 = float(np.std(R_280)); T1 = mu1 + N_SIGMA*sig1
plot_3panel(R_280, R_321, mu1, sig1, T1,
            "Ping 280 (referanse)", "Ping 321 (referanse)",
            f"Fig 1 — To referanse-pinger: 280 vs 321 (begge uten gass)\n"
            f"Boks [{X_MIN},{X_MAX}]×[{Z_MIN},{Z_MAX}] m, K={K}",
            DATA_DIR / "hist_fig1_ping280_vs_321_K10.png")

# Fig 2a
print("\n--- Fig 2a: pooled FØR (200-298) vs gass (299-310) ---")
R_before = get_residuals(list(range(200, 299)), box_gas)
R_gas    = get_residuals(list(range(299, 311)), box_gas)
mu2a = float(np.mean(R_before)); sig2a = float(np.std(R_before)); T2a = mu2a + N_SIGMA*sig2a
plot_3panel(R_before, R_gas, mu2a, sig2a, T2a,
            "Ikke-gass FØR (pings 200–298)", "Gass (pings 299–310)",
            f"Fig 2a — Pooled FØR gass vs gass\n"
            f"Boks [{X_MIN},{X_MAX}]×[{Z_MIN},{Z_MAX}] m, K={K}",
            DATA_DIR / "hist_fig2a_before_vs_gas_K10.png")

# Fig 2b
print("\n--- Fig 2b: pooled ETTER (311-357) vs gass (299-310) ---")
R_after = get_residuals(list(range(311, 358)), box_gas)
mu2b = float(np.mean(R_after)); sig2b = float(np.std(R_after)); T2b = mu2b + N_SIGMA*sig2b
plot_3panel(R_after, R_gas, mu2b, sig2b, T2b,
            "Ikke-gass ETTER (pings 311–357)", "Gass (pings 299–310)",
            f"Fig 2b — Pooled ETTER gass vs gass\n"
            f"Boks [{X_MIN},{X_MAX}]×[{Z_MIN},{Z_MAX}] m, K={K}",
            DATA_DIR / "hist_fig2b_after_vs_gas_K10.png")

# Fig 2c
print("\n--- Fig 2c: FØR vs ETTER (kontroll, ingen gass) ---")
mu2c = float(np.mean(R_before)); sig2c = float(np.std(R_before)); T2c = mu2c + N_SIGMA*sig2c
plot_3panel(R_before, R_after, mu2c, sig2c, T2c,
            "Ikke-gass FØR (pings 200–298)", "Ikke-gass ETTER (pings 311–357)",
            f"Fig 2c — Kontroll: FØR vs ETTER gass (ingen gass i begge)\n"
            f"Boks [{X_MIN},{X_MAX}]×[{Z_MIN},{Z_MAX}] m, K={K}",
            DATA_DIR / "hist_fig2c_before_vs_after_K10.png")

# Fig 3
print("\n--- Fig 3: ping 305, boks med gass vs uten gass ---")
BG_305 = rolling_bg(305, K)
R_med  = (img(305) - BG_305)[box_gas];   R_med  = R_med[np.isfinite(R_med)]
R_uten = (img(305) - BG_305)[box_quiet]; R_uten = R_uten[np.isfinite(R_uten)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Fig 3 — Ping 305: boks med gass vs uten gass  (K={K})", fontsize=12)
edge = np.percentile(np.abs(np.concatenate([R_med, R_uten])), 99.5)
bins = np.linspace(-edge, edge, 100)
axes[0].hist(R_uten, bins=bins, alpha=0.6, color=C_QUIET, density=True,
             label=f"Uten gass [{X_MIN2},{X_MAX2}]×[{Z_MIN2},{Z_MAX2}]")
axes[0].hist(R_med,  bins=bins, alpha=0.6, color=C_GAS,   density=True,
             label=f"Med gass [{X_MIN},{X_MAX}]×[{Z_MIN},{Z_MAX}]")
axes[0].set_xlabel("residual (lineær)"); axes[0].set_ylabel("tetthet")
axes[0].set_title("Lineær skala"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[1].hist(R_uten, bins=bins, alpha=0.6, color=C_QUIET, density=True, label="Uten gass")
axes[1].hist(R_med,  bins=bins, alpha=0.6, color=C_GAS,   density=True, label="Med gass")
axes[1].set_yscale("log")
axes[1].set_xlabel("residual (lineær)"); axes[1].set_ylabel("tetthet (log)")
axes[1].set_title("Log-skala: tung høyrehale i gass-boksen")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
out3 = DATA_DIR / "hist_fig3_med_vs_uten_gass_K10.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Lagret: {out3}")

# Fig 4
print("\n--- Fig 4: andel over terskel for q=1..5 ---")
scenarios = {
    "Ping 280":               R_280,
    "Ping 321":               R_321,
    "Pooled FØR (200–298)":   R_before,
    "Gass (299–310)":         R_gas,
    "Pooled ETTER (311–357)": R_after,
}
colors_s = ["steelblue", "cornflowerblue", "gray", "crimson", "darkorange"]
markers  = ["o", "s", "^", "D", "v"]

fig, ax = plt.subplots(figsize=(10, 6))
for (label, R), color, marker in zip(scenarios.items(), colors_s, markers):
    fracs = []
    for q in Q_VALUES:
        T = mu2a + q * sig2a
        f = float(np.mean(R > T)) * 100
        fracs.append(f)
    ax.plot(Q_VALUES, fracs, color=color, marker=marker,
            linewidth=2, markersize=8, label=label)

ax.axvline(N_SIGMA, color="black", lw=1.5, ls="--", label=f"N=q={N_SIGMA} (valgt terskel)")
ax.set_xlabel("q  (terskel = μ + q·σ)", fontsize=12)
ax.set_ylabel("Andel piksler over terskel [%]", fontsize=12)
ax.set_title(f"Andel over μ+q·σ for ulike scenarier\n"
             f"μ og σ fra pooled FØR-bakgrunn (pings 200–298), K={K}", fontsize=12)
ax.set_xticks(Q_VALUES)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")
out4 = DATA_DIR / "hist_fig4_andel_vs_q_K10.png"
plt.tight_layout()
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Lagret: {out4}")
print("\nAlle figurer ferdig!")
