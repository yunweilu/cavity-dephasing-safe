import os
import pickle
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

_DATA_PLOT = Path(__file__).resolve().parent
_SAFE = _DATA_PLOT.parent
sys.path.insert(0, str(_DATA_PLOT))
sys.path.insert(0, str(_SAFE))
os.chdir(_DATA_PLOT)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
from scipy.signal import hilbert

from drive_detuning_converter import A_to_detuning_mhz, sc
from helper.system import Hamiltonian

# ── Style (normal_plot from plot_instruction) ─────────────────────
_style_path = _SAFE / "plot_instruction"
_style_mod = SourceFileLoader("plot_instruction_mod_total_rate", str(_style_path)).load_module()
normal_plot = _style_mod.normal_plot
fig_width = _style_mod.fig_width

mpl.rcParams["svg.fonttype"] = "path"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams.update(normal_plot)

for _key in ("font.size", "axes.labelsize", "xtick.labelsize",
             "ytick.labelsize", "legend.fontsize"):
    mpl.rcParams[_key] = float(mpl.rcParams[_key]) + 5

NS_TO_MS = 1e6
A_LO, A_HI = 1e-6, 1e-5
REF_LO = 1.0 / 1.0      # 1/(1 ms) at A = 10^-6
REF_HI = 1.0 / 0.1      # 1/(0.1 ms) at A = 10^-5
COLOR_A = "#54278F"       # main purple (panel a dark / driven)
COLOR_A_LIGHT = "#9E9AC8"  # same hue, brighter (panel a, A=10^-6)
COLOR_BC = "#0047AB"       # shared color for panels (b) and (c)
COLOR_STAR = "#3A7CC9"     # lighter blue for stars / Driven in (d)


def _panel_label(ax, text):
    """Place panel tag just outside the top-left of the axes box."""
    ax.text(
        -0.02, 1.06, text, transform=ax.transAxes,
        ha="left", va="bottom", fontweight="bold", clip_on=False,
    )


# ══════════════════════════════════════════════════════════════════
# Load rate data
# ══════════════════════════════════════════════════════════════════
_SIMS = Path(__file__).resolve().parents[1] / "simulations"
with open(_SIMS / "total_rate" / "total_rate.pkl", "rb") as f:
    rate_data = pickle.load(f)

A_values = rate_data["A_values"]
S0_values = np.array(rate_data["S0_values"])
results_sel = rate_data["results"]

all_det_data = np.array([0.96, 6.0, 14.0, 19.0])
target_S0_list = [1e-5, 1e-6]

A_20mhz = sorted(A_values)[-1]
det_20mhz = A_to_detuning_mhz([A_20mhz])[0]
print(f"20 MHz case: A/2pi = {A_20mhz/(2*np.pi)*1e3:.2f} MHz, det = {det_20mhz:.1f} MHz")

avg_20 = np.array(results_sel[A_20mhz]["avg"]) * NS_TO_MS
std_20 = np.array(results_sel[A_20mhz]["std"]) * NS_TO_MS

# ── Analytical setup ─────────────────────────────────────────────
K = 0.1 * 2 * np.pi
gamma_b_down = 1 / 2e4
delta_phi = 1e-6
sc_plus = Hamiltonian(
    sc.phi_ex + delta_phi, sc.Ej / (2 * np.pi), sc.Ec / (2 * np.pi),
    sc.original_dim, sc.trunc_dim, sc.omega_c1, sc.omega_c2,
)
sc_minus = Hamiltonian(
    sc.phi_ex - delta_phi, sc.Ej / (2 * np.pi), sc.Ec / (2 * np.pi),
    sc.original_dim, sc.trunc_dim, sc.omega_c1, sc.omega_c2,
)
pos = sc.state_index((1, 0, 0), sc.original_dim)
domega_dphi = (
    sc_plus.H_dressed[pos, pos].real - sc_minus.H_dressed[pos, pos].real
) / (2 * delta_phi)


def total_ana_selective(S0, det_mhz):
    Delta_bd = det_mhz * 1e-3 * 2 * np.pi
    term_phi = 0.5 * gamma_b_down * K / Delta_bd * 0.1**4
    term_heat = (1 / 16) * gamma_b_down * (Delta_bd / K) ** 2 * (1 - 2 * Delta_bd / K)
    term_excite = 0.25 * domega_dphi**2 * S0**2 / (K / 2 / np.pi)
    return np.abs(term_phi) + np.abs(term_heat) + np.abs(term_excite)


def reference_rate_ms(A):
    frac = (A - A_LO) / (A_HI - A_LO)
    frac = np.clip(frac, 0.0, 1.0)
    return REF_LO + frac * (REF_HI - REF_LO)


def rate_vs_detuning_selective(target_S0):
    idx = np.argmin(np.abs(S0_values - target_S0))
    avg = np.array([results_sel[A]["avg"][idx] for A in A_values])
    std = np.array([results_sel[A]["std"][idx] for A in A_values])
    return avg, std


S0_fine = np.logspace(np.log10(S0_values.min()), np.log10(S0_values.max()), 200)
ana_20 = total_ana_selective(S0_fine, det_20mhz) * NS_TO_MS
eta_vs_A = reference_rate_ms(S0_values) / avg_20

# ══════════════════════════════════════════════════════════════════
# 1×4 row layout
# ══════════════════════════════════════════════════════════════════
panel_sz = fig_width
fig = plt.figure(figsize=(4.4 * panel_sz, 1.05 * panel_sz))
gs = GridSpec(
    1, 4, figure=fig,
    wspace=0.65,
    left=0.06, right=0.98, top=0.88, bottom=0.18,
)
labelpad = 8

ax_eta_det = fig.add_subplot(gs[0, 0])
ax_eta_A = fig.add_subplot(gs[0, 1])
ax_gamma = fig.add_subplot(gs[0, 2])
ax_ramsey = fig.add_subplot(gs[0, 3])

# ── (a) eta vs -Δ_bd for A = 10^-5 and 10^-6 (detuning > 1 MHz) ───
# Same hue, different brightness: dark for 10^-5, light for 10^-6
colors_a = [COLOR_A, COLOR_A_LIGHT]
for target_S0, color in zip(target_S0_list, colors_a):
    avg_sel, _ = rate_vs_detuning_selective(target_S0)
    dets = all_det_data[1:]  # 6, 14, 19 MHz (> 1 MHz)
    rates_ms = np.array(avg_sel) * NS_TO_MS
    eta = reference_rate_ms(target_S0) / rates_ms
    ax_eta_det.plot(
        dets, eta, "-o", markersize=4, color=color,
        label=rf"$A = 10^{{{int(np.log10(target_S0))}}}\,\Phi_0$",
    )

ax_eta_det.set_xlabel(r"$-\Delta_{bd}/2\pi$ (MHz)", labelpad=labelpad)
ax_eta_det.set_ylabel(r"$\eta$", labelpad=labelpad)
ax_eta_det.legend(frameon=False)
_panel_label(ax_eta_det, "(a)")

# ── (b) eta vs A at detuning = 20 MHz ─────────────────────────────
ax_eta_A.plot(S0_values, eta_vs_A, "-o", markersize=4, color=COLOR_BC, zorder=1)
ax_eta_A.plot(
    S0_values[-1], eta_vs_A[-1], "*", markersize=10, color=COLOR_STAR,
    zorder=3, clip_on=False,
)
ax_eta_A.set_xlabel(r"$A$ ($\Phi_0$)", labelpad=labelpad)
ax_eta_A.set_ylabel(r"$\eta$", labelpad=labelpad)
ax_eta_A.set_xscale("log")
_panel_label(ax_eta_A, "(b)")

# ── (c) gamma_cphi^ss vs A at 20 MHz ──────────────────────────────
ax_gamma.errorbar(
    S0_values, avg_20, yerr=std_20,
    fmt="-o", markersize=4, capsize=2, elinewidth=1, color=COLOR_BC,
)
ax_gamma.plot(S0_fine, ana_20, "--", color=COLOR_BC, lw=1.2)
ax_gamma.plot(
    S0_values[-1], avg_20[-1], "*", markersize=10, color=COLOR_STAR,
    zorder=3, clip_on=False,
)
ax_gamma.set_xlabel(r"$A$ ($\Phi_0$)", labelpad=labelpad)
ax_gamma.set_ylabel(r"$\Gamma_{c\phi}^{\mathrm{ss}}$ (1/ms)", labelpad=labelpad)
ax_gamma.set_xscale("log")
ax_gamma.set_yscale("log")
trans_c = blended_transform_factory(ax_gamma.transData, ax_gamma.transAxes)
ax_gamma.text(7e-6, 0.72, "Numerical", transform=trans_c,
              color="black", ha="left", va="bottom")
ax_gamma.text(5e-6, 0.35, "Analytical", transform=trans_c,
              color="black", ha="left", va="bottom")
_panel_label(ax_gamma, "(c)")

# ── (d) Ramsey envelopes ──────────────────────────────────────────
with open(_SIMS / "ramsey_sigmax" / "ramsey_sigmax_data.pkl", "rb") as f:
    ramsey_data = pickle.load(f)

time_pts = ramsey_data["time_pts"]
driven = ramsey_data["driven"]
undriven = ramsey_data["undriven"]
mask = time_pts <= 100000
env_driven = np.abs(hilbert(driven[mask]))
env_undriven = np.abs(hilbert(undriven[mask]))
trim = max(1, len(env_driven) // 50)
t_plot = time_pts[mask][trim:-trim] / 1e3
env_driven = env_driven[trim:-trim]
env_undriven = env_undriven[trim:-trim]

ax_ramsey.plot(t_plot, env_driven, color=COLOR_STAR)
ax_ramsey.plot(t_plot, env_undriven, color="black")
ax_ramsey.set_xlabel(r"$t$ ($\mu$s)", labelpad=labelpad)
ax_ramsey.set_ylabel(r"$\langle X \rangle$", labelpad=labelpad)
ax_ramsey.tick_params(axis="both", pad=4)

idx_d = np.argmin(np.abs(t_plot - 62))
idx_u = np.argmin(np.abs(t_plot - 38))
ax_ramsey.text(t_plot[idx_d], env_driven[idx_d] + 0.03, "Driven",
               color=COLOR_STAR, ha="left", va="bottom")
ax_ramsey.text(t_plot[idx_u], env_undriven[idx_u] - 0.06, "Undriven",
               color="black", ha="left", va="top")
_panel_label(ax_ramsey, "(d)")

fig.savefig("figure3.pdf", bbox_inches="tight", pad_inches=0.04)
print(f"Saved: {_DATA_PLOT / 'figure3.pdf'}")
plt.close(fig)
