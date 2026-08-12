"""
Given a drive amplitude A, find the optimal drive frequency omega_d
that minimizes the transmon quasi-energy |e_quasi[1]|,
and return the detuning omega_d - omega_transmon.
"""
import sys
sys.path.append('..')

import numpy as np
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed
from helper.system import Hamiltonian

# Same parameters as flux_sweep_plots.py
Ej = 30.19
Ec = 0.1
omega_c1 = 5.226
omega_c2 = 8.135
phi_ex = 0.2
bare_dim = [10, 1, 6]
trunc_dim = [5, 1, 4]

sc = Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)

# Dressed transmon frequency
position = sc.state_index((1, 0, 0), sc.original_dim)
omega_transmon = sc.H_dressed[position, position].real
print(f"Dressed transmon freq: {omega_transmon / (2 * np.pi):.6f} GHz")


def find_sweet_spot_detuning(A):
    """
    Find the drive frequency that minimizes |d(e_quasi[1])/d(phi)| (sweet spot),
    and return the detuning omega_d - omega_transmon.

    Parameters
    ----------
    A : float
        Drive amplitude in angular frequency units (rad * GHz).

    Returns
    -------
    detuning : float
        omega_d_opt - omega_transmon (angular frequency units).
    omega_d_opt : float
        Optimal drive frequency (angular frequency units).
    grad_min : float
        The minimized |d(e_quasi[1])/d(phi)| value.
    """
    def cost(omega_d):
        grad = sc.equasi_gradient(A, omega_d)
        return np.abs(grad[0])

    # Search around the dressed transmon frequency
    bounds = (omega_transmon + 0.00 * 2 * np.pi,
              omega_transmon + 0.1 * 2 * np.pi)

    result = minimize_scalar(cost, bounds=bounds, method='bounded',
                             options={'xatol': 1e-8, 'maxiter': 100})
    omega_d_opt = result.x
    grad_min = result.fun
    
    detuning = omega_d_opt - omega_transmon

    return detuning, omega_d_opt, grad_min


def A_to_detuning_mhz(A_values, n_jobs=-1):
    """Convert a list of A values to detunings in MHz, in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(find_sweet_spot_detuning)(A) for A in A_values)
    return np.array([det / (2 * np.pi) * 1e3 for det, _, _ in results])


if __name__ == '__main__':
    A_values = [0.5e-4 * 2 * np.pi, 1e-3 * 2 * np.pi, 5e-3 * 2 * np.pi,
                10e-3 * 2 * np.pi]

    print(f"\n{'A/2pi (MHz)':>12s}  {'detuning/2pi (MHz)':>18s}  "
          f"{'omega_d/2pi (GHz)':>17s}  {'|grad|/2pi':>16s}")
    print("-" * 72)

    for A in A_values:
        detuning, omega_d_opt, grad_min = find_sweet_spot_detuning(A)
        print(f"{A/(2*np.pi)*1e3:12.2f}  {detuning/(2*np.pi)*1e3:18.4f}  "
              f"{omega_d_opt/(2*np.pi):17.6f}  {grad_min/(2*np.pi):16.6e}")
