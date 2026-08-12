import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

sys.path.append("..")

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helper.system import Hamiltonian

# ── Style from SNAIL_new/plot_instruction ────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
_style = SourceFileLoader(
    "plot_instruction_perturbation",
    str(_ROOT / "plot_instruction"),
).load_module()
mpl.rcParams.update(_style.square_plot)

# ══════════════════════════════════════════════════════════════════
# System parameters
# ══════════════════════════════════════════════════════════════════
Ej = 30.19
Ec = 0.1
omega_c1 = 5.226
omega_c2 = 8.135
phi_ex = 0.2
bare_dim = [10, 1, 6]
trunc_dim = [5, 1, 4]

sc = Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)

position_b = sc.state_index((1, 0, 0), sc.original_dim)
omega_b_now = sc.H_dressed[position_b, position_b].real
print(f"Dressed transmon frequency (now): {omega_b_now / (2 * np.pi):.6f} GHz")

delta_phi = 1e-6
sc_p = Hamiltonian(phi_ex + delta_phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
sc_m = Hamiltonian(phi_ex - delta_phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
pos_p = sc_p.state_index((1, 0, 0), sc_p.original_dim)
pos_m = sc_m.state_index((1, 0, 0), sc_m.original_dim)
domega_b_dphi = (sc_p.H_dressed[pos_p, pos_p].real
                 - sc_m.H_dressed[pos_m, pos_m].real) / (2 * delta_phi)
norm_factor = np.abs(domega_b_dphi) / (2 * np.pi)  # GHz
print(f"|d(omega_b)/d(phi)| / 2pi = {norm_factor:.6f} GHz")

# ══════════════════════════════════════════════════════════════════
# 2D sweep data — use the same drive grids / reference as the sweep
# ══════════════════════════════════════════════════════════════════
_RATES_BUNDLE = Path(__file__).resolve().parents[1] / "simulations" / "rates_bundle" / "rates_bundle.pkl"
with open(_RATES_BUNDLE, "rb") as f:
    bundle = pickle.load(f)
rates = bundle["rates"]
rates2 = bundle["rates2"]
print(f"Loaded rates: {rates.shape}, rates2: {rates2.shape}")

# Reference dressed frequency when the Floquet sweep was run (NOT recomputed later)
if "omega_s" in bundle:
    omega_s_ref = float(bundle["omega_s"])
else:
    omega_s_ref = 6.1591 * 2 * np.pi
    print("Warning: omega_s missing from pickle; using 6.1591 GHz sweep reference")
print(f"Dressed transmon frequency (sweep ref): {omega_s_ref / (2 * np.pi):.6f} GHz")
if abs(omega_s_ref - omega_b_now) / (2 * np.pi) > 1e-4:
    print(
        f"Note: current omega_b differs by "
        f"{(omega_b_now - omega_s_ref) / (2 * np.pi) * 1e3:.2f} MHz — "
        "x-axis uses sweep reference so numerical data align with drive frequencies"
    )

# Main panel: full (Omega, detuning) grid
if "omega_ds_full" in bundle and "amplitudes_full" in bundle:
    omega_ds1 = np.asarray(bundle["omega_ds_full"], dtype=float)
    amplitudes1 = np.asarray(bundle["amplitudes_full"], dtype=float)
else:
    omega_ds1 = np.linspace(6.159 - 2e-3, 6.159 + 0.025, 100) * 2 * np.pi
    amplitudes1 = 2 * np.pi * np.linspace(0.01e-3, 10e-3, 100)

# Inset: zoom grid stored with the pickle
omega_ds2 = np.asarray(bundle["omega_ds"], dtype=float)
amplitudes2 = np.asarray(bundle["amplitudes"], dtype=float)

# x-axis: -Delta_bd/(2pi) = (omega_d - omega_s)/(2pi)  [MHz]
det1 = (omega_ds1 / (2 * np.pi) - omega_s_ref / (2 * np.pi)) * 1e3
amps1_mhz = amplitudes1 / (2 * np.pi) * 1e3
X1, Y1 = np.meshgrid(det1, amps1_mhz)
Z1_raw = np.array(rates, dtype=float, copy=True)
Z1_raw[~np.isfinite(Z1_raw) | (Z1_raw <= 0)] = np.nan
Z1 = Z1_raw / (2 * np.pi)

det2 = (omega_ds2 / (2 * np.pi) - omega_s_ref / (2 * np.pi)) * 1e3
amps2_mhz = amplitudes2 / (2 * np.pi) * 1e3
X2, Y2 = np.meshgrid(det2, amps2_mhz)
Z2_raw = np.array(rates2, dtype=float, copy=True)
Z2_raw[~np.isfinite(Z2_raw) | (Z2_raw <= 0)] = np.nan
Z2 = Z2_raw / (2 * np.pi)

# ══════════════════════════════════════════════════════════════════
# Main 2D colormap
# ══════════════════════════════════════════════════════════════════
_PLOT_FS = 13

fig_w, fig_h = _style.square_plot["figure.figsize"]
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_box_aspect(1)

# Color limits scaled so the colormap matches the previous plot after /(2pi).
_inv_2pi = 1.0 / (2 * np.pi)
color_norm = LogNorm(
    vmin=5e-4 * norm_factor * _inv_2pi,
    vmax=2e-2 * norm_factor * _inv_2pi,
)
cmap = plt.cm.inferno.copy()
cmap.set_bad(cmap(0))

pcm = ax.pcolormesh(X1, Y1, Z1, shading="nearest", cmap=cmap, norm=color_norm, rasterized=True)
xlim_lo, xlim_hi = det1[0], det1[-1]
ylim_lo, ylim_hi = amps1_mhz[0], amps1_mhz[-1]
ax.set_xlim(xlim_lo, xlim_hi)
ax.set_ylim(ylim_lo, ylim_hi)

# White dashed curve: 0.01 - Omega_0^2 (1/Delta_bd^2 - 1/(Delta_bd+chi)^2) = 0
chi_GHz = -2.0e-3  # chi/(2pi) in GHz
x_curve_mhz = np.linspace(xlim_lo, xlim_hi, 2000)
delta_bd_GHz = -x_curve_mhz / 1e3
denom = 1.0 / delta_bd_GHz**2 - 1.0 / (delta_bd_GHz + chi_GHz) ** 2
valid = denom > 0
y_GHz = np.full_like(delta_bd_GHz, np.nan)
y_GHz[valid] = np.sqrt(0.01 / denom[valid])
y_curve_mhz = y_GHz * 1e3
in_range = (y_curve_mhz > ylim_lo) & (y_curve_mhz < ylim_hi)
ax.plot(
    x_curve_mhz[valid & in_range], y_curve_mhz[valid & in_range],
    "w--", lw=1.2, zorder=5,
)

ax.set_xlabel(r"$-\Delta_{bd}/2\pi~(\mathrm{MHz})$", fontsize=_PLOT_FS)
ax.set_ylabel(r"$\Omega_0/2\pi~(\mathrm{MHz})$", fontsize=_PLOT_FS)
ax.tick_params(axis="both", which="major", labelsize=_PLOT_FS)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar = fig.colorbar(pcm, cax=cax)
cbar.ax.tick_params(labelsize=_PLOT_FS)
cbar.set_label(
    r"$\lvert D_{1,0} \rvert / 2\pi~(\mathrm{GHz}/\Phi_0)$",
    fontsize=_PLOT_FS,
    loc="top",
    labelpad=6,
)

tex_dir = Path("/Users/yunwei/Desktop/project/cavity dephasing/6949db0bfd19311f68336ca1/Sec 2")
tex_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(tex_dir / "figure2.pdf")
plt.savefig("figure2.pdf")
plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# Inset (zoomed subset) — separate figure
# ══════════════════════════════════════════════════════════════════
fig_ins, ax_ins = plt.subplots(figsize=(1.2, 1.2))

x0_ins, x1_ins = -1.0, float(det2[-1])
y0_ins, y1_ins = float(amps2_mhz[0]), float(amps2_mhz[-1])
ax_ins.pcolormesh(X2, Y2, Z2, shading="nearest", cmap=cmap, norm=color_norm, rasterized=True)
ax_ins.set_xlim(x0_ins, x1_ins)
ax_ins.set_ylim(y0_ins, y1_ins)

# Selective-limit sweet spots: |Delta_bd| = |Delta_bc/g| Omega_0  (paper: ~10)
analytic_det = 10.0 * amps2_mhz
mask_pos = (analytic_det >= x0_ins) & (analytic_det <= x1_ins)
mask_neg = (-analytic_det >= x0_ins) & (-analytic_det <= x1_ins)
ax_ins.plot(analytic_det[mask_pos], amps2_mhz[mask_pos], "w--", lw=0.9)
if np.any(mask_neg):
    ax_ins.plot(-analytic_det[mask_neg], amps2_mhz[mask_neg], "w--", lw=0.9)

ax_ins.set_xlabel("")
ax_ins.set_ylabel("")
ax_ins.set_xticks([-1, 0, 1])
ax_ins.set_xticks(np.arange(x0_ins, x1_ins + 0.01, 0.2), minor=True)
ax_ins.set_yticks([0.05, 0.10, 0.15, 0.20])
ax_ins.set_yticks(np.arange(y0_ins, y1_ins + 0.001, 0.01), minor=True)
ax_ins.tick_params(labelsize=8, which="both", direction="in", top=False, right=False)

fig_ins.savefig(tex_dir / "figure2_inse.pdf")
fig_ins.savefig("figure2_inse.pdf")
plt.close(fig_ins)
