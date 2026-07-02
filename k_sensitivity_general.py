import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DS_ID    = "0032"                                            # datasett ID
DATA_DIR = Path.home() / "Desktop" / "MASTER" / "horten0032"  # mappe


PING_OFFSET = 0


REF_IDX = 50


GAS_START_FILE = None
GAS_END_FILE   = None

# Analyseboks (samme som for 0033)
X_MIN, X_MAX =   0, 10
Z_MIN, Z_MAX = -60, -40



K_VALUES   = [1, 2, 3, 5, 8, 10]
K_GOODNESS = list(range(1, 21))
N_SIGMA    = 5
BUFFER     = 15

#  LAST DATA
print(f"Laster data for datasett {DS_ID} ...")
cube = np.load(DATA_DIR / f"cube_{DS_ID}.npy")
X    = np.load(DATA_DIR / f"X_{DS_ID}.npy")
Z    = np.load(DATA_DIR / f"Z_{DS_ID}.npy")

n_beams, n_samp, n_pings = cube.shape
ping_nrs = np.arange(n_pings) + PING_OFFSET
print(f"  Kube: {cube.shape}  — fil-pings {ping_nrs[0]}–{ping_nrs[-1]}")
print(f"  Referanse-ping: fil-ping {ping_nrs[REF_IDX]} (kubindeks {REF_IDX})")

cube_lin = 10.0 ** (cube / 10.0)

box_mask = (X >= X_MIN) & (X <= X_MAX) & (Z >= Z_MIN) & (Z <= Z_MAX)
n_box = int(box_mask.sum())
print(f"  Analyseboks [{X_MIN},{X_MAX}] m × [{Z_MIN},{Z_MAX}] m: {n_box} piksler")


if GAS_START_FILE is not None and GAS_END_FILE is not None:
    GAS_START = GAS_START_FILE - PING_OFFSET
    GAS_END   = GAS_END_FILE   - PING_OFFSET
    BG_IDX = np.concatenate([
        np.arange(0, max(0, GAS_START - BUFFER)),
        np.arange(min(n_pings, GAS_END + 1 + BUFFER), n_pings)
    ])
    print(f"  Gass-vindu (kubindeks): {GAS_START}–{GAS_END}")
    print(f"  BG_IDX: {len(BG_IDX)} bakgrunns-pinger")
else:
    GAS_START = None
    GAS_END   = None
    BG_IDX = np.arange(n_pings)
    print("  Gass-vindu: ukjent — bruker alle pinger for godhetskriterie")

#  BAKGRUNNSFUNKSJONER
def bg_before(i, K):
    idx = list(range(max(0, i - K), i))
    if not idx: return None
    return np.nanmedian(cube_lin[:, :, idx], axis=2)

def bg_twosided(i, K):
    half  = K // 2
    extra = K % 2
    before = list(range(max(0, i - half - extra), i))
    after  = list(range(i + 1, min(n_pings, i + half + 1)))
    idx = before + after
    if not idx: return None
    return np.nanmedian(cube_lin[:, :, idx], axis=2)

def bg_after(i, K):
    idx = list(range(i + 1, min(n_pings, i + K + 1)))
    if not idx: return None
    return np.nanmedian(cube_lin[:, :, idx], axis=2)

strategies = [
    ("Strategi 1 — Ensidig FØR   (K pings foran)",       bg_before),
    ("Strategi 2 — Tosidig       (K/2 foran + K/2 bak)", bg_twosided),
    ("Strategi 3 — Ensidig ETTER (K pings bak)",          bg_after),
]

# SCORE BEREGNING
def compute_scores(bg_func, K):
    BG_ref = bg_func(REF_IDX, K)
    if BG_ref is None:
        BG_ref = np.nanmedian(cube_lin, axis=2)
    R_ref = (cube_lin[:, :, REF_IDX] - BG_ref)[box_mask]
    R_ref = R_ref[np.isfinite(R_ref)]
    mu    = float(np.nanmean(R_ref))
    sigma = float(np.nanstd(R_ref))
    thresh = mu + N_SIGMA * sigma

    scores = np.full(n_pings, np.nan)
    for i in range(n_pings):
        BG = bg_func(i, K)
        if BG is None: continue
        R = (cube_lin[:, :, i] - BG)[box_mask]
        R = R[np.isfinite(R)]
        if len(R) == 0: continue
        scores[i] = np.mean(R > thresh) * 100
    return scores, thresh

# GODHETSKRITERIE
def goodness_delta(bg_func, K):
    deltas = []
    for idx in range(len(BG_IDX) - 1):
        i     = int(BG_IDX[idx])
        i_nxt = int(BG_IDX[idx + 1])
        if i_nxt != i + 1:
            continue
        BG_i   = bg_func(i,     K)
        BG_nxt = bg_func(i_nxt, K)
        if BG_i is None or BG_nxt is None:
            continue
        diff = np.abs(BG_nxt[box_mask] - BG_i[box_mask])
        deltas.append(float(np.nanmean(diff)))
    return float(np.nanmean(deltas)) if deltas else np.nan

#  FIGUR 1: Score profiler
print("\nBeregner score-profiler ...")
colors = plt.cm.plasma(np.linspace(0.05, 0.9, len(K_VALUES)))
fig1, axes1 = plt.subplots(3, 1, figsize=(13, 14), sharex=True)
fig1.suptitle(
    f"K-sensitivity — Horten {DS_ID}  (terskel: μ + {N_SIGMA}σ)\n"
    f"Boks [{X_MIN},{X_MAX}] m × [{Z_MIN},{Z_MAX}] m  |  "
    f"Referanse-ping: fil-ping {ping_nrs[REF_IDX]}",
    fontsize=13
)

for ax, (title, bg_func) in zip(axes1, strategies):
    if GAS_START_FILE is not None:
        ax.axvspan(GAS_START_FILE, GAS_END_FILE, color="orange",
                   alpha=0.2, label="Gass-vindu")
    for K, color in zip(K_VALUES, colors):
        scores, _ = compute_scores(bg_func, K)
        valid = ~np.isnan(scores)
        ax.plot(ping_nrs[valid], scores[valid],
                "o-", color=color, markersize=3, linewidth=1.5, label=f"K={K}")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(f"% piksler over μ+{N_SIGMA}σ", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

axes1[-1].set_xlabel("Ping-nummer (i filen)", fontsize=11)
plt.tight_layout()
out1 = DATA_DIR / f"k_sensitivity_score_profiler_{DS_ID}.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Figur 1 lagret: {out1}")

# FIGUR 2: Godhetskriterie
print("\nBeregner godhetskriterie")
strat_colors = ["royalblue", "firebrick", "seagreen"]
fig2, ax2 = plt.subplots(figsize=(9, 5))

for (title, bg_func), col in zip(strategies, strat_colors):
    deltas = []
    for K in K_GOODNESS:
        print(f"  {title[:22]}  K={K} ...", end="\r")
        deltas.append(goodness_delta(bg_func, K))
    deltas = np.array(deltas)
    ax2.plot(K_GOODNESS, deltas, "o-", color=col, linewidth=2, markersize=6,
             label=title.split("—")[1].strip())

    if len(deltas) >= 3:
        d2 = np.abs(np.diff(deltas, n=2))
        knee_idx = int(np.argmax(d2)) + 1
        knee_K   = K_GOODNESS[knee_idx]
        ax2.axvline(knee_K, color=col, linestyle="--", alpha=0.6,
                    label=f"Knekkpunkt K={knee_K}")
        print(f"\n  {title[:22]}  → knekkpunkt K={knee_K}")

ax2.set_xlabel("K  (antall bakgrunns-pinger)", fontsize=12)
ax2.set_ylabel("Gj.snitt |BG(i+1,K) − BG(i,K)|  [lineær amplitude]", fontsize=11)
ax2.set_title(
    f"Godhetskriterie for K — Horten {DS_ID}\n"
    , fontsize=12
)
ax2.set_xticks(K_GOODNESS)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
out2 = DATA_DIR / f"k_goodness_criterion_{DS_ID}.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Figur 2 lagret: {out2}")
print("\nFerdig! ✓")
