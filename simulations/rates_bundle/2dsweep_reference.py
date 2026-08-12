"""Reference plot for rates_bundle.pkl (generator notebook not in repo)."""
import sys
from pathlib import Path

_SAFE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SAFE))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from helper.system import Hamiltonian

_HERE = Path(__file__).resolve().parent

# ── System parameters (same as qenergy_vs_app.py) ──────────────────────────
Ej = 30.19
Ec = 0.1
omega_c1 = 5.226
omega_c2 = 8.135
phi_ex = 0.2
bare_dim = [10, 1, 6]
trunc_dim = [5, 1, 4]

sc = Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)

# Dressed SNAIL frequency (zero-drive reference for detuning)
position = sc.state_index((1, 0, 0), sc.original_dim)
omega_s = sc.H_dressed[position, position].real  # in angular frequency
print(f"Dressed SNAIL frequency: {omega_s / (2 * np.pi):.4f} GHz")

# ── Bare transmon frequency derivative w.r.t. flux ─────────────────────────
delta_phi = 1e-6
sc_plus = Hamiltonian(phi_ex + delta_phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
sc_minus = Hamiltonian(phi_ex - delta_phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
pos_p = sc_plus.state_index((1, 0, 0), sc_plus.original_dim)
pos_m = sc_minus.state_index((1, 0, 0), sc_minus.original_dim)
omega_s_plus = sc_plus.H_dressed[pos_p, pos_p].real
omega_s_minus = sc_minus.H_dressed[pos_m, pos_m].real
domega_dphi = (omega_s_plus - omega_s_minus) / (2 * delta_phi)
print(f"d(omega_transmon)/d(phi) / 2pi = {domega_dphi / (2 * np.pi):.6f} GHz")
print(f"|d(omega_transmon)/d(phi)| / 2pi = {np.abs(domega_dphi) / (2 * np.pi):.6f} GHz")

# ── Load pre-computed rates from notebook ──────────────────────────────────
with open(_HERE / "rates_bundle.pkl", "rb") as f:
    data = pickle.load(f)

rates = data["rates"]
rates2 = data["rates2"]
print(f"Loaded rates: {rates.shape}, rates2: {rates2.shape}")

# ============================================================
# 1) TOP PANEL DATA (full range)
# ============================================================
N_det1 = 100
N_amp1 = 100
detu1 = 2e-3

omega_ds1 = np.linspace(6.159 - detu1, 6.159 + 0.025, N_det1) * 2 * np.pi
amplitudes1 = 2 * np.pi * np.linspace(0.01e-3, 10e-3, N_amp1)

det1 = (omega_ds1 / (2 * np.pi) - omega_s / (2 * np.pi)) * 1e3   # Delta_bd in MHz
amps1_mhz = amplitudes1 / (2 * np.pi) * 1e3                       # MHz
X1, Y1 = np.meshgrid(det1, amps1_mhz)

Z1_raw = np.array(rates, dtype=float, copy=True)
Z1_raw[~np.isfinite(Z1_raw) | (Z1_raw <= 0)] = np.nan
Z1 = Z1_raw / (np.abs(domega_dphi) / (2 * np.pi))

# ============================================================
# 2) BOTTOM PANEL DATA (zoomed subset)
# ============================================================
N_det2 = 100
N_amp2 = 100
detu2 = 2e-3

omega_ds2 = np.linspace(6.159 - detu2, 6.159 + detu2, N_det2) * 2 * np.pi
amplitudes2 = 2 * np.pi * np.linspace(0.01e-3, 0.2e-3, N_amp2)

det2 = (omega_ds2 / (2 * np.pi) - omega_s / (2 * np.pi)) * 1e3   # Delta_bd in MHz
amps2_mhz = amplitudes2 / (2 * np.pi) * 1e3                       # MHz
X2, Y2 = np.meshgrid(det2, amps2_mhz)

Z2_raw = np.array(rates2, dtype=float, copy=True)
Z2_raw[~np.isfinite(Z2_raw) | (Z2_raw <= 0)] = np.nan
Z2 = Z2_raw / (np.abs(domega_dphi) / (2 * np.pi))

# ============================================================
# 0) Utility: overlay resonant region |Omega_0 / Delta_bd| >= 1
# ============================================================
def add_resonant_region(ax, x_min, x_max, y_min, y_max, label=True):
    """Omega_0 >= |Delta_bd|  ⟺  y >= |x|  in MHz units."""
    y_line = np.linspace(y_min, y_max, 800)
    x_left = -y_line
    x_right = y_line

    x_left_clip = np.clip(x_left, x_min, x_max)
    x_right_clip = np.clip(x_right, x_min, x_max)

    # Light semi-transparent overlay with hatching
    ax.fill_betweenx(
        y_line, x_left_clip, x_right_clip,
        facecolor='white', alpha=0.22, hatch='///',
        edgecolor='gray', linewidth=0.0, zorder=3)

    # Thin boundary lines
    mask_l = (x_left >= x_min) & (x_left <= x_max)
    mask_r = (x_right >= x_min) & (x_right <= x_max)
    ax.plot(x_left[mask_l], y_line[mask_l], '-', color='gray', lw=1.2, zorder=4)
    lbl = 'resonant region' if label else None
    ax.plot(x_right[mask_r], y_line[mask_r], '-', color='gray', lw=1.2,
            zorder=4, label=lbl)

# ============================================================
# 3) Shared color normalization
# ============================================================
norm = LogNorm(vmin=1e-3, vmax=1e-1)

cmap = plt.cm.inferno.copy()
cmap.set_bad(cmap(0))

# ============================================================
# 4) Build figure — main plot with colorbar + inset
# ============================================================
fig, ax_main = plt.subplots(figsize=(8, 5), constrained_layout=True)

# ---- main plot (full range) ----
pcm1 = ax_main.pcolormesh(X1, Y1, Z1, shading='nearest', cmap=cmap, norm=norm)
ax_main.set_xlim(det1[0], det1[-1])
ax_main.set_ylim(amps1_mhz[0], 10.0)

add_resonant_region(ax_main, det1[0], det1[-1], amps1_mhz[0], 10.0, label=True)

ax_main.set_xlabel(r'$-\Delta_{bd}/2\pi$ (MHz)')
ax_main.set_ylabel(r'$\Omega_0/2\pi$ (MHz)')
ax_main.legend(loc='upper right', frameon=False, fontsize=9)

# Extract "undriven" D_0: max |detuning|, min amplitude
D0_undriv = Z1[0, -1]
print(f"Undriven D_0 = {D0_undriv:.4e}")

cbar = fig.colorbar(pcm1, ax=ax_main, pad=0.02)
cbar.set_label(r'$D_0$')

# Mark undriven value on colorbar with "(undriven)" beside the tick
cbar.ax.text(3.8, D0_undriv, '(undriven)', fontsize=7,
             va='center', ha='left',
             transform=cbar.ax.get_yaxis_transform())

# ---- inset (zoomed subset, small square, no labels) ----
ax_ins = ax_main.inset_axes([0.42, 0.55, 0.28, 0.38])

x0_ins, x1_ins = -1.0, det2[-1]
y0_ins, y1_ins = amps2_mhz[0], amps2_mhz[-1]

ax_ins.pcolormesh(X2, Y2, Z2, shading='nearest', cmap=cmap, norm=norm)
ax_ins.set_xlim(x0_ins, x1_ins)
ax_ins.set_ylim(y0_ins, y1_ins)

add_resonant_region(ax_ins, x0_ins, x1_ins, y0_ins, y1_ins, label=False)

# Analytical curves
analytic_det_mhz = 10.0 * amps2_mhz
mask_pos = (analytic_det_mhz >= x0_ins) & (analytic_det_mhz <= x1_ins)
mask_neg = (-analytic_det_mhz >= x0_ins) & (-analytic_det_mhz <= x1_ins)

ax_ins.plot(analytic_det_mhz[mask_pos], amps2_mhz[mask_pos],
            'w--', lw=1.2)
if np.any(mask_neg):
    ax_ins.plot(-analytic_det_mhz[mask_neg], amps2_mhz[mask_neg],
                'w-.', lw=1.2)

# No axis labels, small tick labels, white border
ax_ins.set_xlabel('')
ax_ins.set_ylabel('')
ax_ins.tick_params(labelsize=10)
for spine in ax_ins.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(1.5)

plt.savefig('combined_full_and_subset_shared_cbar.pdf', bbox_inches='tight')
plt.show()
