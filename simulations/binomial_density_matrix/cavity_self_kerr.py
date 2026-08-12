"""Cavity self-Kerr used for undriven Wigner phase correction (figure 4)."""
import numpy as np
from system import sort_eigenpairs
from hamiltonian_generator import Hamiltonian


def _cavity_self_kerr_from_diag(diag_h, n_cavity):
    """Estimate cavity self-Kerr from |0,0>, |0,1>, |0,2> ladder."""
    if n_cavity < 3 or diag_h.shape[0] < 3:
        return float("nan")
    e0 = np.real(diag_h[0, 0])
    e1 = np.real(diag_h[1, 1])
    e2 = np.real(diag_h[2, 2])
    return float(e2 - 2.0 * e1 + e0)


def cavity_self_kerr(A, n_transmon=3, n_cavity=5):
    """Return cavity self-Kerr of the rotating-frame Floquet Hamiltonian."""
    phi_ex = 0.2
    Ej = 30.19
    Ec = 0.1
    sc_tmp = Hamiltonian(phi_ex, Ej, Ec, [5, 10])
    optimal_omega, _ = sc_tmp.optimal_omegad(A)
    optimal_omega = optimal_omega * 2 * np.pi

    sc = Hamiltonian(phi_ex, Ej, Ec, [n_transmon, n_cavity])
    sc.setup_floquet_system(A, optimal_omega)
    total_dim = n_transmon * n_cavity

    n_s = np.kron(np.diag(np.arange(n_transmon)), np.eye(n_cavity))
    n_c = np.kron(np.eye(n_transmon), np.diag(np.arange(n_cavity)))

    H_control_filtered = np.zeros_like(np.asarray(sc.H_control), dtype=complex)
    for i in range(total_dim - n_cavity):
        j = i + n_cavity
        H_control_filtered[i, j] = sc.H_control[i, j]
        H_control_filtered[j, i] = sc.H_control[j, i]

    H0_diag = np.diag(np.diag(sc.H) - sc.H[0, 0])
    omegac = H0_diag[1, 1]
    H0_rot_raw = H0_diag - optimal_omega * n_s - omegac * n_c + (A / 2.0) * H_control_filtered
    evals_rot, U = np.linalg.eigh(H0_rot_raw)
    evals_rot, _ = sort_eigenpairs(evals_rot, U)
    evals_rot = np.real(evals_rot - evals_rot[0])
    H0_rot = np.diag(evals_rot)
    return _cavity_self_kerr_from_diag(H0_rot, n_cavity)
