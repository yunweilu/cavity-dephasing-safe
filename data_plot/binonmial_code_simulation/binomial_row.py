"""
Figure 4 row-0 helpers: binomial dephasing curves + reduced-cavity Wigner panels.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import qutip as qt
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap

# SAFE/helper for binomial Floquet rates
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helper.system import Hamiltonian

_SIM_DIR = Path(__file__).resolve().parents[2] / "simulations" / "binomial_density_matrix"
_RHO_PATH = _SIM_DIR / "avg_density_matrix_over_time_undriven_driven.pkl"

WIGNER_CMAP = LinearSegmentedColormap.from_list(
    "wigner_blue_white_red",
    ["#2166ac", "#ffffff", "#b2182b"],
    N=256,
)
_WIGNER_IQ_AXIS_SCALE = np.sqrt(2.0)
_WIGNER_IQ_VALUE_SCALE = 0.5


# ── Binomial dephasing curves ─────────────────────────────────────

def _quasi_energy_full(sc_obj, A, omega):
    H0_qt = qt.Qobj(np.array(sc_obj.H_dressed))
    Hc_qt = qt.Qobj(np.array(sc_obj.H_control_dressed))
    T = (2 * np.pi) / omega
    H = [H0_qt, [Hc_qt, lambda t, _args=None: A * np.cos(omega * t)]]
    fb = qt.FloquetBasis(H, T)
    evecs = np.column_stack([m.full() for m in fb.mode(0)])
    evals = np.array(fb.e_quasi)
    evals_s, _ = sc_obj.sort_eigenpairs(evals, evecs)
    return evals_s - evals_s[0]


def _build_system(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2, g_val):
    sc = Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
    sc.g = g_val
    sc.H, sc.H_control, sc.H_flux_drive = sc.get_H()
    sc.H_dressed, sc.H_control_dressed, sc.H_flux_drive_dressed = sc.dressed_basis()
    return sc


def get_binomial_plot_data(n_jobs=1):
    """Detuning sweep of binomial-code dephasing rates (figure 4 panel a)."""
    Ej, Ec = 30.19, 0.1
    omega_c1, omega_c2 = 5.226, 8.135
    phi_ex = 0.2
    bare_dim, trunc_dim = [10, 1, 8], [3, 1, 6]
    g_val = 0.025 * 2 * np.pi

    sc = _build_system(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2, g_val)
    dim = sc.original_dim
    idx_0 = sc.state_index((0, 0, 0), dim)
    idx_1 = sc.state_index((0, 0, 1), dim)
    idx_2 = sc.state_index((0, 0, 2), dim)
    idx_3 = sc.state_index((0, 0, 3), dim)
    idx_4 = sc.state_index((0, 0, 4), dim)
    idx_b = sc.state_index((1, 0, 0), dim)
    omega_s = sc.H_dressed[idx_b, idx_b].real

    delta_phi = 1e-6

    def omega_b_vs_phi(phi_val):
        sc_tmp = _build_system(phi_val, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2, g_val)
        return (sc_tmp.H_dressed[idx_b, idx_b].real - sc_tmp.H_dressed[idx_0, idx_0].real) / (2 * np.pi)

    domega_b_dphi = np.abs(
        (omega_b_vs_phi(phi_ex + delta_phi) - omega_b_vs_phi(phi_ex - delta_phi)) / (2 * delta_phi)
    )

    def compute_derivatives(omegad, A):
        delta = 1e-6

        def get_eq(p):
            sc_t = _build_system(p, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2, g_val)
            return _quasi_energy_full(sc_t, A, omegad)

        grad = (get_eq(phi_ex + delta) - get_eq(phi_ex - delta)) / (2 * delta)
        return (
            np.abs(grad[idx_2] - grad[idx_0]) / (2 * np.pi) / domega_b_dphi,
            np.abs(grad[idx_4] - grad[idx_2]) / (2 * np.pi) / domega_b_dphi,
            np.abs(grad[idx_4] - grad[idx_0]) / (2 * np.pi) / domega_b_dphi,
            np.abs(grad[idx_3] - grad[idx_0]) / (2 * np.pi) / domega_b_dphi,
            np.abs(grad[idx_4] - grad[idx_3]) / (2 * np.pi) / domega_b_dphi,
            np.abs(grad[idx_2] - grad[idx_1]) / (2 * np.pi) / domega_b_dphi,
            np.abs(grad[idx_3] - grad[idx_1]) / (2 * np.pi) / domega_b_dphi,
        )

    A = 10e-3 * 2 * np.pi
    detunings_ghz = np.logspace(np.log10(0.02), np.log10(0.08), 200)
    omega_ds = omega_s + detunings_ghz * 2 * np.pi
    if int(n_jobs) == 1:
        results = [compute_derivatives(wd, A) for wd in omega_ds]
    else:
        results = Parallel(n_jobs=int(n_jobs), verbose=0)(
            delayed(compute_derivatives)(wd, A) for wd in omega_ds
        )

    gamma_scale = np.sqrt(8.0) * 1e-5 * domega_b_dphi * 2 * np.pi
    labels = ["0->2", "2->4", "0->4", "0->3", "3->4", "1->2", "1->3"]
    gamma_curves = {
        lab: np.array([r[i] for r in results]) * gamma_scale for i, lab in enumerate(labels)
    }
    return {"detunings_ghz": detunings_ghz, "gamma_curves": gamma_curves}


def _plot_binomial_panel(ax):
    data = get_binomial_plot_data(n_jobs=1)
    x_mhz = np.asarray(data["detunings_ghz"], dtype=float) * 1e3
    g = data["gamma_curves"]
    colors = ["#d73027", "#fc8d59", "#fee090", "#b8860b", "#e0f3f8", "#91bfdb", "#4575b4"]
    labels = ["0->2", "2->4", "0->4", "0->3", "3->4", "1->2", "1->3"]
    for c, lab in zip(colors, labels):
        a, b = lab.split("->")
        ax.plot(x_mhz, np.asarray(g[lab]), lw=1.8, color=c, label=rf"$\gamma_{{\phi;{b},{a}}}$")

    target_mhz = 30.0
    idx_star = int(np.argmin(np.abs(x_mhz - target_mhz)))
    ax.plot(
        x_mhz[idx_star],
        float(np.asarray(g["0->2"])[idx_star]),
        marker="*",
        color="black",
        markersize=11,
        label="_nolegend_",
        zorder=6,
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$-\Delta_{bd}/2\pi$ (MHz)")
    ax.set_ylabel(r"$\gamma_\phi\ (1/\mathrm{ns})$")
    ax.set_title("Binomial encoding")
    ax.legend(loc="best")


# ── Reduced-cavity Wigner data / phase correction ─────────────────

def _load_reduced_cavity_data(data_path, case="undriven"):
    with Path(data_path).open("rb") as f:
        payload = pickle.load(f)
    if case not in payload:
        raise KeyError(f"Case '{case}' not found. Available keys: {list(payload.keys())}")
    time_points = np.asarray(payload[case]["time_points"])
    avg_rho_cav = np.asarray(payload[case]["avg_rho_cav"])
    if avg_rho_cav.ndim != 3 or avg_rho_cav.shape[1:] != (5, 5):
        raise ValueError("Expected avg_rho_cav with shape (n_t, 5, 5).")
    return time_points, avg_rho_cav


def _apply_self_kerr_phase_removal(avg_rho_cav, time_points, self_kerr):
    n_cavity = avg_rho_cav.shape[1]
    n_op = qt.num(n_cavity)
    H_kerr = 0.5 * float(self_kerr) * n_op * (n_op - qt.qeye(n_cavity))
    corrected = np.empty_like(avg_rho_cav, dtype=complex)
    for idx, t in enumerate(time_points):
        rho = qt.Qobj(avg_rho_cav[idx], dims=[[n_cavity], [n_cavity]])
        U_t = (1j * H_kerr * float(t)).expm()
        corrected[idx] = (U_t * rho * U_t.dag()).full()
    return corrected


def _apply_driven_phase_align_to_t0(avg_rho_cav, grid_points=721):
    n_cavity = avg_rho_cav.shape[1]
    n_op = qt.num(n_cavity)
    G = 0.5 * n_op * (n_op - qt.qeye(n_cavity))
    phi_grid = np.linspace(-np.pi, np.pi, int(grid_points), endpoint=True)

    corrected = np.empty_like(avg_rho_cav, dtype=complex)
    phases, fids = [0.0], [1.0]
    rho_ref = qt.Qobj(np.asarray(avg_rho_cav[0], dtype=complex), dims=[[n_cavity], [n_cavity]])
    corrected[0] = rho_ref.full()

    for idx in range(1, avg_rho_cav.shape[0]):
        rho = qt.Qobj(np.asarray(avg_rho_cav[idx], dtype=complex), dims=[[n_cavity], [n_cavity]])
        best_phi, best_fid, best_rho = 0.0, -1.0, rho
        for phi in phi_grid:
            U = (1j * G * float(phi)).expm()
            rho_corr = U * rho * U.dag()
            fid = float(qt.metrics.fidelity(rho_ref, rho_corr))
            if fid > best_fid:
                best_fid, best_phi, best_rho = fid, float(phi), rho_corr
        corrected[idx] = best_rho.full()
        phases.append(best_phi)
        fids.append(best_fid)

    return corrected, np.asarray(phases), np.asarray(fids)


def _get_undriven_self_kerr(data_path):
    with Path(data_path).open("rb") as f:
        payload = pickle.load(f)
    if "undriven" not in payload:
        return 0.0
    params = payload.get("params", {})
    A_case = payload["undriven"].get("A", 0.0)
    n_transmon = int(params.get("n_transmon", 3))
    n_cavity = int(params.get("n_cavity", 5))

    _path_before = sys.path[:]
    try:
        if str(_SIM_DIR) not in sys.path:
            sys.path.insert(0, str(_SIM_DIR))
        from cavity_self_kerr import cavity_self_kerr
        return float(cavity_self_kerr(A_case, n_transmon=n_transmon, n_cavity=n_cavity))
    finally:
        sys.path[:] = _path_before


def _apply_case_correction(data_path, case, time_points, avg_rho_cav):
    if case == "driven":
        corrected, phases, fids = _apply_driven_phase_align_to_t0(avg_rho_cav)
        print(
            "Driven per-slice phase alignment to t=0: "
            f"median(phi)={np.median(phases):.6f} rad, "
            f"min fidelity={np.min(fids):.6f}, median fidelity={np.median(fids):.6f}"
        )
        return corrected

    k_case = _get_undriven_self_kerr(data_path)
    print(f"Using self-Kerr correction ({case}): K={k_case:.6e}")
    return _apply_self_kerr_phase_removal(avg_rho_cav, time_points, k_case)


def _wigner_on_iq_grid(rho, q_lim=6.0, grid=201):
    q = np.linspace(-q_lim, q_lim, grid)
    W_qp = qt.wigner(rho, q, q)
    return q * _WIGNER_IQ_AXIS_SCALE, W_qp * _WIGNER_IQ_VALUE_SCALE


# ── Public API for generate_figure4 ───────────────────────────────

def plot_binomial_and_wigner_row(_fig, ax_gamma, axes_wigner, *, style_mod=None):
    """
    Panel (a): binomial dephasing curves.
    Panels (b–d): reduced-cavity Wigner snapshots (t=0, undriven, driven).
    """
    _path_before = sys.path[:]
    try:
        _plot_binomial_panel(ax_gamma)

        if str(_SIM_DIR) not in sys.path:
            sys.path.insert(0, str(_SIM_DIR))

        t_u, rho_u = _load_reduced_cavity_data(_RHO_PATH, case="undriven")
        t_d, rho_d = _load_reduced_cavity_data(_RHO_PATH, case="driven")
        rho_u = _apply_case_correction(_RHO_PATH, "undriven", t_u, rho_u)
        rho_d = _apply_case_correction(_RHO_PATH, "driven", t_d, rho_d)

        idx_uf = len(t_u) - 1
        idx_df = len(t_d) - 1
        rho_u0 = qt.Qobj(rho_u[0], dims=[[5], [5]])
        rho_uf = qt.Qobj(rho_u[idx_uf], dims=[[5], [5]])
        rho_df = qt.Qobj(rho_d[idx_df], dims=[[5], [5]])

        iq, W_u0 = _wigner_on_iq_grid(rho_u0)
        _, W_uf = _wigner_on_iq_grid(rho_uf)
        _, W_df = _wigner_on_iq_grid(rho_df)
        vmax = max(np.max(np.abs(W_u0)), np.max(np.abs(W_uf)), np.max(np.abs(W_df)))
        levels = np.linspace(-vmax, vmax, 121)

        ax_w0, ax_w1, ax_w2 = axes_wigner
        cf = ax_w0.contourf(iq, iq, W_u0, levels=levels, cmap=WIGNER_CMAP)
        ax_w0.set_title(r"$t=0\,\mu s$")
        ax_w0.set_xlabel(r"$I$")
        ax_w0.set_ylabel(r"$Q$")
        ax_w0.set_aspect("equal")

        cf = ax_w1.contourf(iq, iq, W_uf, levels=levels, cmap=WIGNER_CMAP)
        ax_w1.set_title(rf"Undriven, $t={t_u[idx_uf]/1e3:.0f}\,\mu s$")
        ax_w1.set_xlabel(r"$I$")
        ax_w1.set_ylabel("")
        ax_w1.set_aspect("equal")

        cf = ax_w2.contourf(iq, iq, W_df, levels=levels, cmap=WIGNER_CMAP)
        ax_w2.set_title(rf"Driven, $t={t_d[idx_df]/1e3:.0f}\,\mu s$")
        ax_w2.set_xlabel(r"$I$")
        ax_w2.set_ylabel("")
        ax_w2.set_aspect("equal")

        if style_mod is not None:
            style_mod.add_axis_arrows(ax_gamma)
            for ax in axes_wigner:
                style_mod.add_axis_arrows(ax)
        return cf
    finally:
        sys.path[:] = _path_before
