import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

DATA_DIR = Path.home() / "Desktop" / "MASTER" / "horten0033"

K       = 10    # endret fra 3 til 10
N_SIGMA = 5
REF_IDX = 121

WX   = 10
WZ   = 20
STEP =  5

GAS_START_FILE, GAS_END_FILE = 299, 310
GAS_START = GAS_START_FILE - 200
GAS_END   = GAS_END_FILE   - 200

POSITIONS = {
    "Gass-boks [0,10]×[-60,-40]":           (5,   -50),
    "Havbunn-nær [0,10]×[-72,-62]":         (5,   -67),
    "Vannkolonne u/gass [40,50]×[-55,-35]":  (45, -45),
    "Ytre sektor [-60,-50]×[-30,-10]":       (-55, -20),
}

print("Laster data ...")
cube     = np.load(DATA_DIR / "cube.npy")
X        = np.load(DATA_DIR / "X.npy")
Z        = np.load(DATA_DIR / "Z.npy")
cube_lin = 10.0 ** (cube / 10.0)
n_beams, n_samp, n_pings = cube.shape
ping_nrs = np.arange(n_pings) + 200
print(f"  Kube: {cube.shape}  — pings {ping_nrs[0]}–{ping_nrs[-1]}")

print(f"Forberegner bakgrunn (K={K}) for alle {n_pings} pings ...")
BG_all = np.full_like(cube_lin, np.nan)
for i in range(n_pings):
    idx = list(range(max(0, i - K), i))
    if idx:
        BG_all[:, :, i] = np.nanmedian(cube_lin[:, :, idx], axis=2)
R_all = cube_lin - BG_all
print("  Ferdig.")

def score_for_box(box):
    if box.sum() < 10:
        return None, None
    R_box = R_all[box, :]
    R_ref = R_box[:, REF_IDX]
    R_ref = R_ref[np.isfinite(R_ref)]
    if len(R_ref) < 5:
        return None, None
    mu    = float(np.mean(R_ref))
    sigma = float(np.std(R_ref))
    T     = mu + N_SIGMA * sigma
    valid = np.isfinite(R_box)
    above = (R_box > T) & valid
    n_valid = valid.sum(axis=0)
    scores = np.where(n_valid >= 5,
                      above.sum(axis=0) / np.maximum(n_valid, 1) * 100,
                      np.nan)
    return scores, T

x_centers = np.arange(-120, 125, STEP)
z_centers  = np.arange(-70,   -8, STEP)

gas_map = np.full((len(x_centers), len(z_centers)), np.nan)
rho_map = np.full((len(x_centers), len(z_centers)), np.nan)

print(f"\nSlidende vindu {WX}m × {WZ}m, steg {STEP}m")
print(f"  {len(x_centers)} × {len(z_centers)} = {len(x_centers)*len(z_centers)} vinduer")

for ix, x0 in enumerate(x_centers):
    for iz, z0 in enumerate(z_centers):
        box = ((X >= x0 - WX/2) & (X <= x0 + WX/2) &
               (Z >= z0 - WZ/2) & (Z <= z0 + WZ/2))
        scores, _ = score_for_box(box)
        if scores is None:
            continue
        gas_sc = scores[GAS_START:GAS_END+1]
        bg_sc  = np.concatenate([scores[:GAS_START], scores[GAS_END+1:]])
        gas_mean = np.nanmean(gas_sc)
        bg_mean  = np.nanmean(bg_sc)
        gas_map[ix, iz] = gas_mean
        if bg_mean > 1e-9:
            rho_map[ix, iz] = gas_mean / bg_mean
    print(f"  x={x0:+.0f} m  ({ix+1}/{len(x_centers)})")

print("\nFerdig med sliding window!")

# Fig 1: Heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    f"Convolution detektor — Horten 0033  [K={K}]\n"
    f"Vindu {WX}m × {WZ}m, K={K}, terskel μ+{N_SIGMA}σ",
    fontsize=13
)

ZZ, XX = np.meshgrid(z_centers, x_centers)

ax = axes[0]
im = ax.pcolormesh(ZZ, XX, gas_map, cmap="hot", shading="auto")
plt.colorbar(im, ax=ax, label="Gj.snitt score gass-pinger [%]")
ax.set_xlabel("Z (m)"); ax.set_ylabel("X (m)")
ax.set_title("Score-kart (gass-pinger 299–310)\nLysere = høyere deteksjonsscore")
for label, (x0, z0) in POSITIONS.items():
    ax.add_patch(Rectangle((z0 - WZ/2, x0 - WX/2), WZ, WX,
                            edgecolor="cyan", facecolor="none", lw=1.5))
    ax.text(z0, x0, label[:8], color="cyan", fontsize=6, ha="center", va="center")
ax.grid(True, alpha=0.2)

ax2 = axes[1]
rho_clipped = np.clip(rho_map, 0, 50)
im2 = ax2.pcolormesh(ZZ, XX, rho_clipped, cmap="RdYlGn", shading="auto")
plt.colorbar(im2, ax=ax2, label="ρ = gass-score / bakgrunn-score (maks 50)")
ax2.set_xlabel("Z (m)"); ax2.set_ylabel("X (m)")
ax2.set_title("ρ-kart (gass/bakgrunn-ratio)\nGrønt = god diskriminering")
ax2.grid(True, alpha=0.2)

plt.tight_layout()
out1 = DATA_DIR / "convolution_heatmap_K10.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Lagret: {out1}")

# Fig 2: Score kurvene
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
fig.suptitle(
    f"Score-kurver for typiske posisjoner  [K={K}]\n"
    f"Vindu {WX}m × {WZ}m, K={K}, terskel μ+{N_SIGMA}σ",
    fontsize=13
)
axes = axes.flatten()

for ax, (label, (x0, z0)) in zip(axes, POSITIONS.items()):
    box = ((X >= x0 - WX/2) & (X <= x0 + WX/2) &
           (Z >= z0 - WZ/2) & (Z <= z0 + WZ/2))
    scores, T = score_for_box(box)
    ax.axvspan(GAS_START_FILE, GAS_END_FILE, color="orange", alpha=0.2, label="Gass-vindu")
    if scores is not None:
        valid = ~np.isnan(scores)
        ax.plot(ping_nrs[valid], scores[valid], "o-", color="royalblue", markersize=3, linewidth=1.5)
        gas_sc = scores[GAS_START:GAS_END+1]
        bg_sc  = np.concatenate([scores[:GAS_START], scores[GAS_END+1:]])
        rho = np.nanmean(gas_sc) / max(np.nanmean(bg_sc), 1e-12)
        ax.set_title(
            f"{label}\n"
            f"gass={np.nanmean(gas_sc):.2f}%  bakgrunn={np.nanmean(bg_sc):.2f}%  ρ={rho:.1f}×",
            fontsize=10
        )
    else:
        ax.set_title(f"{label}\n(ingen data)", fontsize=10)
    ax.set_ylabel(f"% piksler over μ+{N_SIGMA}σ", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

axes[-1].set_xlabel("Ping-nummer", fontsize=11)
axes[-2].set_xlabel("Ping-nummer", fontsize=11)
plt.tight_layout()
out2 = DATA_DIR / "convolution_posisjoner_K10.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Lagret: {out2}")
print("\nAlt ferdig!")
