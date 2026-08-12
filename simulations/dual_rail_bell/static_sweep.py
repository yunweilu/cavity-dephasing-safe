import pickle
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# Load local system.py explicitly (avoid collision with other folders named system).
_system = SourceFileLoader("dual_rail_bell_system_static", str(_HERE / "system.py")).load_module()
Hamiltonian = _system.Hamiltonian

# Same flux-noise proxy scaling as SNAIL_new/data_plot/plot_dual_rail.py:
# gamma_phi \propto |d tilde{omega}/d Phi| with 1/f prefactor absorbed here.
GAMMA_SCALE = np.sqrt(8.0) * 1e-5 * 2 * np.pi

# Match paper-style figure (dual-rail panel)
COLOR_CAVITY_1 = "#D4AF37"  # gold — gamma^(1)
COLOR_CAVITY_2 = "#17BECF"  # teal — gamma^(2)


def _style_setup():
    style_path = Path(__file__).resolve().parents[2] / "plot_instruction"
    style_mod = SourceFileLoader("plot_instruction_mod_dual_rail", str(style_path)).load_module()
    mpl.rcParams.update(style_mod.normal_plot)
    return style_mod


def _get_array(data, candidates, required=True):
    for key in candidates:
        if key in data:
            return np.array(data[key])
    if required:
        raise KeyError(f"Missing required key. Tried: {candidates}")
    return None


def _extract_series(data):
    # Primary expected keys
    d_c1 = _get_array(data, ["d_c1_arr", "d_c1", "dc1"])
    d_c2 = _get_array(data, ["d_c2_arr", "d_c2", "dc2"])

    # X-axis detuning (GHz preferred internally)
    detunings_ghz = _get_array(data, ["detunings_ghz"], required=False)
    if detunings_ghz is None:
        detunings_mhz = _get_array(data, ["detunings_mhz"], required=False)
        if detunings_mhz is not None:
            detunings_ghz = detunings_mhz * 1e-3

    # Fallback if only omega sweep and reference frequency are stored
    if detunings_ghz is None and "omega_ds" in data and "omega_s" in data:
        omega_ds = np.array(data["omega_ds"])
        omega_s = float(data["omega_s"])
        detunings_ghz = (omega_ds - omega_s) / (2 * np.pi)

    if detunings_ghz is None:
        raise KeyError("Could not infer detuning axis. Need detunings_ghz/detunings_mhz or omega_ds+omega_s.")

    return detunings_ghz, d_c1, d_c2


def _extract_norm(data):
    for key in [
        "domega_b_dphi",
        "d_omega_b_dphi",
        "domega_b_dphi_fd",
        "omega_b_phi_derivative",
        "omega_b_derivative",
    ]:
        if key in data:
            return float(data[key])
    raise KeyError(
        "Missing normalization derivative |d omega_b / dPhi| in data. "
        "Expected one of: domega_b_dphi, d_omega_b_dphi, domega_b_dphi_fd, "
        "omega_b_phi_derivative, omega_b_derivative."
    )


def derivatives_to_gamma_phi(d1, d2):
    """Map |d E / d Phi| quasi-energy derivatives to dephasing rate proxy (1/ns)."""
    g1 = np.abs(np.asarray(d1, dtype=float)) * GAMMA_SCALE
    g2 = np.abs(np.asarray(d2, dtype=float)) * GAMMA_SCALE
    return g1, g2


def _intersection_point(x, y1, y2):
    """Linear interpolation where y1 and y2 cross; fallback to minimum |y1-y2|."""
    x = np.asarray(x, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    diff = y1 - y2
    for i in range(len(diff) - 1):
        if diff[i] == 0:
            return x[i], y1[i]
        if diff[i] * diff[i + 1] < 0:
            t = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))
            xc = x[i] + t * (x[i + 1] - x[i])
            yc = y1[i] + t * (y1[i + 1] - y1[i])
            return xc, yc
    j = int(np.argmin(np.abs(diff)))
    return x[j], float(np.sqrt(max(y1[j] * y2[j], 1e-30)))


def _compute_norm_from_model(data=None):
    """Compute two cavities quasi-energy derivatives and sweep over drive frequency."""
    if data is None:
        data = {}
    Ej = float(data.get("Ej", 30.19))
    Ec = float(data.get("Ec", 0.1))
    omega_c1 = float(data.get("omega_c1", 5.226))
    omega_c2 = float(data.get("omega_c2", 7.335))
    phi_ex = float(data.get("phi_ex", 0.2))
    bare_dim = data.get("bare_dim", [10, 6, 6])
    trunc_dim = data.get("trunc_dim", [5, 2, 2])
    g_val = float(data.get("g_val", 0.05 * 2 * np.pi))

    def build(phi):
        sc = Hamiltonian(phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
        sc.g = g_val
        sc.H, sc.H_control, sc.H_flux_drive, sc.noise, sc.s = sc.get_H()
        sc.H_dressed, sc.H_control_dressed, sc.H_flux_drive_dressed = sc.dressed_basis()
        return sc

    # Base Hamiltonian
    sc0 = build(phi_ex)
    dim = sc0.original_dim
    idx_0 = sc0.state_index((0, 0, 0), dim)
    idx_b = sc0.state_index((1, 0, 0), dim)

    # Transmon (dressed) frequency
    omega_q = sc0.H_dressed[idx_b, idx_b].real - sc0.H_dressed[idx_0, idx_0].real

    # Detuning sweep: omega_d = omega_q + (Delta MHz) * 2pi * 1e-3  =>  -Delta_bd/2pi ~ Delta MHz
    detunings_mhz = np.linspace(20, 50, 201)
    drive_freqs = omega_q + (detunings_mhz * 1e-3) * 2 * np.pi

    # Drive amplitude 10 MHz (same as paper-style dual-rail plot)
    A = 10e-3 * 2 * np.pi

    delta_phi = 1e-6
    sc_p = build(phi_ex + delta_phi)
    sc_m = build(phi_ex - delta_phi)

    derivs_c1 = []
    derivs_c2 = []

    for omegad in drive_freqs:
        E_c1_p, E_c2_p = sc_p.quasi_energy(A, omegad)
        E_c1_m, E_c2_m = sc_m.quasi_energy(A, omegad)

        grad_c1 = (E_c1_p - E_c1_m) / (2 * delta_phi) / (2 * np.pi)
        grad_c2 = (E_c2_p - E_c2_m) / (2 * delta_phi) / (2 * np.pi)

        derivs_c1.append(grad_c1)
        derivs_c2.append(grad_c2)

    return detunings_mhz, np.array(derivs_c1), np.array(derivs_c2)


def plot_dual_rail_rate(detuning_mhz, dc1, dc2, ax=None, title="Dual-rail rate"):
    """
    Log-scale dephasing proxy for logical transition vs -Delta_bd / 2pi (MHz).

    Curves are per-cavity inherited dephasing on the relevant transition, in the
    same spirit as gamma_phi^(i) in the dual-rail manuscript figure.
    """
    g1, g2 = derivatives_to_gamma_phi(dc1, dc2)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        detuning_mhz,
        g1,
        "-",
        color=COLOR_CAVITY_1,
        lw=2.0,
        label=r"$\gamma_{\phi,\,|0_L\rangle \rightarrow |1_L\rangle}^{(1)}$",
    )
    ax.plot(
        detuning_mhz,
        g2,
        "-",
        color=COLOR_CAVITY_2,
        lw=2.0,
        label=r"$\gamma_{\phi,\,|0_L\rangle \rightarrow |1_L\rangle}^{(2)}$",
    )

    sx, sy = _intersection_point(detuning_mhz, g1, g2)
    ax.plot(sx, sy, "*", color="black", markersize=14, zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(20.0, 50.0)
    ax.set_ylim(1e-8, 1e-5)
    ax.set_xlabel(r"$-\Delta_{bd}/2\pi$ (MHz)")
    ax.set_ylabel(r"$\gamma_{\phi}$ (1/ns)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(False)
    return ax, (sx, sy)


def main():
    print("Computing quasi-energy derivatives...")
    detuning, dc1, dc2 = _compute_norm_from_model({})
    g1, g2 = derivatives_to_gamma_phi(dc1, dc2)
    print("Done computing. Sample (first, middle, last):")
    for label, arr in [("gamma1", g1), ("gamma2", g2)]:
        print(f"  {label}: {arr[0]:.3e}, {arr[len(arr)//2]:.3e}, {arr[-1]:.3e} 1/ns")

    try:
        _style_setup()
    except Exception as e:
        print(f"Plot style setup failed (continuing with default): {e}")

    fig, ax = plt.subplots(figsize=(8, 6))
    _, (sx, sy) = plot_dual_rail_rate(detuning, dc1, dc2, ax=ax)
    print(f"Curve intersection marker at (-Delta_bd/2pi, gamma) ~ ({sx:.2f} MHz, {sy:.3e} 1/ns)")
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "dual_rail_rate_static_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved figure to {out}")
    plt.show()


if __name__ == "__main__":
    main()
