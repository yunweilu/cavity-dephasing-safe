"""1D quasi-energy rate vs drive detuning (uses historical local system.py)."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar

from system import Hamiltonian

# Test parameters
Ej = 30.19
Ec = 0.1
omega_c1 = 5.226
omega_c2 = 8.135
phi_ex = 0.2
bare_dim = [10, 1, 6]
trunc_dim = [5, 1, 4]

sc = Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)

# Dressed transmon frequency
position_b = sc.state_index((1, 0, 0), sc.original_dim)
omega_b = sc.H_dressed[position_b, position_b].real
print(f"Dressed transmon frequency: {omega_b / (2 * np.pi):.6f} GHz")

# d(omega_b)/d(phi) via finite differences
delta_phi = 1e-6
sc_p = Hamiltonian(phi_ex + delta_phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
sc_m = Hamiltonian(phi_ex - delta_phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
pos_p = sc_p.state_index((1, 0, 0), sc_p.original_dim)
pos_m = sc_m.state_index((1, 0, 0), sc_m.original_dim)
domega_b_dphi = (sc_p.H_dressed[pos_p, pos_p].real - sc_m.H_dressed[pos_m, pos_m].real) / (2 * delta_phi)
norm_factor = np.abs(domega_b_dphi) / (2 * np.pi)  # GHz
print(f"|d(omega_b)/d(phi)| / 2pi = {norm_factor:.6f} GHz")

# Function to calculate static rate (equasi_gradient only)
def calculate_static_rate(omegad, amplitude):
    der = sc.equasi_gradient(amplitude, omegad)[0]
    return np.abs(der) / 2 / np.pi

# Single amplitude: A/2pi = 5 MHz
A = 5e-3 * 2 * np.pi

# Find minimum D_0 around Delta_bd ~ -0.5 MHz
def cost(omega_d):
    der = sc.equasi_gradient(A, omega_d)[0]
    return np.abs(der) / 2 / np.pi / norm_factor

det_center_ghz = -0.5e-3  # -0.5 MHz in GHz
omega_d_center = omega_b + det_center_ghz * 2 * np.pi
res = minimize_scalar(cost,
                      bounds=(omega_d_center - 0.5e-3 * 2 * np.pi,
                              omega_d_center + 0.5e-3 * 2 * np.pi),
                      method='bounded', options={'xatol': 1e-10})
opt_det_mhz = (res.x - omega_b) / (2 * np.pi) * 1e3
print(f"Minimum D_0 = {res.fun:.6e} at -Delta_bd/2pi = {opt_det_mhz:.6f} MHz")
print(f"  omega_d_opt / 2pi = {res.x / (2 * np.pi):.8f} GHz")

omega_ds = np.linspace(6.16, 6.22, 200) * 2 * np.pi

print("Computing equasi_gradient …")
static_rates = np.array(
    Parallel(n_jobs=-1)(delayed(calculate_static_rate)(w, A) for w in omega_ds))

# Set font sizes for paper-quality figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16,
    'mathtext.fontset': 'stix'
})

fig, ax = plt.subplots(figsize=(6, 4))

det_mhz = (omega_ds - omega_b) / (2 * np.pi) * 1e3
D0_num = static_rates / norm_factor
ax.plot(det_mhz, D0_num, label='Numerical', linewidth=2, color='blue', linestyle='-')
ax.set_xlabel(r'$-\Delta_{bd}/2\pi$ (MHz)')
ax.set_ylabel(r'$D_0$')
ax.legend()
ax.set_yscale('log')
ax.set_ylim(top=1e-1)
ax.set_title(r"$\Omega_0/2\pi = {:.1f}$ MHz".format(A * 1e3 / (2 * np.pi)))

plt.tight_layout()
out = Path(__file__).resolve().parent / 'compare.pdf'
plt.savefig(out, bbox_inches='tight')
print(f"Saved {out}")
plt.show()
