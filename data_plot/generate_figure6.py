import sys
from pathlib import Path

_DATA_PLOT = Path(__file__).resolve().parent
_SAFE = _DATA_PLOT.parent
sys.path.insert(0, str(_SAFE))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from helper.system import Hamiltonian

# ══════════════════════════════════════════════════════════════════
# Style (from plot_instruction)
# ══════════════════════════════════════════════════════════════════
fig_width_pt = 246.0
inches_per_pt = 1.0 / 72.27
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width / 1.45

mpl.rcParams['svg.fonttype'] = 'path'
mpl.rcParams['pdf.fonttype'] = 42

mpl.rcParams.update({
    "figure.figsize": (fig_width, fig_height),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "STIXGeneral",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "mathtext.fontset": "stix",
    "lines.linewidth": 1.6,
    "lines.markersize": 4.0,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.grid": False,
})

# ══════════════════════════════════════════════════════════════════
# SNAIL parameters (from SNAILss.ipynb)
# ══════════════════════════════════════════════════════════════════
M = 3
EJ = 90
EC = 177e-3
beta = 0.147
omega_c1 = 8.045
omega_c2 = 5.0
bare_dim = [10, 1, 5]
trunc_dim = [5, 1, 3]
# SNAIL–cavity coupling used for this appendix figure (not the helper default).
g = 0.05 * 2 * np.pi

# Drive parameters
omega_d = 6.5 * 2 * np.pi
A_d = 3e-3 * 2 * np.pi

# Flux sweep range
phi_ext_qe = np.linspace(0.311, 0.315, 51)

# ══════════════════════════════════════════════════════════════════
# Numerical: cavity quasi-energy via Floquet
# ══════════════════════════════════════════════════════════════════
def get_quasi_energy(phi_ex):
    sc = Hamiltonian(phi_ex, EJ, EC, bare_dim, trunc_dim, omega_c1, omega_c2, N=M, beta=beta, g=g)
    eq1, eq2, eq3 = sc.quasi_energy(A_d, omega_d)
    return float(np.real(eq1)) / (2 * np.pi)  # GHz

print("Computing Floquet cavity quasi-energies …")
qe_cavity = np.array(
    Parallel(n_jobs=-1)(delayed(get_quasi_energy)(phi) for phi in phi_ext_qe))

# ══════════════════════════════════════════════════════════════════
# Analytical: cavity frequency shift
# ══════════════════════════════════════════════════════════════════
def get_analytical_cavity(phi_ex):
    sc = Hamiltonian(phi_ex, EJ, EC, bare_dim, trunc_dim, omega_c1, omega_c2, N=M, beta=beta, g=g)
    Hd = sc.H_dressed
    dim = sc.original_dim

    omega_c_bar = float(np.real(Hd[sc.state_index((0, 0, 1), dim),
                                   sc.state_index((0, 0, 1), dim)]))
    omega_s = float(np.real(Hd[sc.state_index((1, 0, 0), dim),
                                sc.state_index((1, 0, 0), dim)]))
    chi_val = sc.chi()
    delta = omega_s - omega_d

    omega_c_ana = omega_c_bar + A_d**2 / delta - A_d**2 / (delta + chi_val) - omega_d
    return float(np.real(omega_c_ana)) / (2 * np.pi)

print("Computing analytical cavity frequencies …")
results_ana = Parallel(n_jobs=-1)(
    delayed(get_analytical_cavity)(phi) for phi in phi_ext_qe)
qe_analytical = np.array(results_ana)

# ══════════════════════════════════════════════════════════════════
# Plot: cavity frequency only
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots()

ax.plot(phi_ext_qe, qe_cavity, color='C0', label='Numerical')
ax.plot(phi_ext_qe, qe_analytical, ls='--', color='C1', label='Analytical')
ax.set_xlabel(r'$\Phi/\Phi_0$')
ax.set_ylabel(r'$(\tilde{\omega}_{c,0\to1}\ \mathrm{mod}\ \omega_d)/2\pi$ (GHz)')
ax.set_title(
    rf'$\omega_d/2\pi = 6.5$ GHz,  $\Omega_0/2\pi = {A_d / 2 / np.pi * 1e3:.0f}$ MHz',
    fontsize=8.5)
ax.legend()

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ── Save ───────────────────────────────────────────────────────
out_pdf = _DATA_PLOT / "figure6.pdf"
plt.savefig(out_pdf)
print(f"Saved: {out_pdf}")
plt.close()
