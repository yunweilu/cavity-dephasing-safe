import argparse
import os
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import matrix_histogram


LABELS_2Q_LOGICAL = [
    r"$|00_{L}\rangle$",
    r"$|01_{L}\rangle$",
    r"$|10_{L}\rangle$",
    r"$|11_{L}\rangle$",
]
PLOT_CMAP = mpl.cm.RdBu_r


def load_case_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "logical_rhos" not in data or len(data["logical_rhos"]) == 0:
        raise ValueError(f"No logical density matrices found in {path}.")
    return data


def to_qobj_4x4(rho):
    if isinstance(rho, qt.Qobj):
        out = rho
    else:
        out = qt.Qobj(np.asarray(rho, dtype=complex), dims=[[4], [4]])
    if out.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 logical density matrix, got shape {out.shape}.")
    tr = out.tr()
    if abs(tr) > 0:
        out = out / tr
    return out


def bell_phi_plus_density():
    ket00 = qt.basis(4, 0)
    ket11 = qt.basis(4, 3)
    psi = (ket00 + ket11).unit()
    return qt.ket2dm(psi)


def bell_fidelity(rho):
    target = bell_phi_plus_density()
    return float(qt.metrics.fidelity(target, rho))


def phase_corrected_logical_rho(rho, label="", grid_points=721):
    """
    Optimize number-conserving Kerr-like phase phi to maximize Bell fidelity.
    U(phi) = exp(i * phi * n_L),  n_L = diag(0, 1, 1, 2) in the logical basis.
    """
    target = bell_phi_plus_density()
    n_L = qt.Qobj(np.diag([0.0, 1.0, 1.0, 2.0]), dims=[[4], [4]])
    phi_grid = np.linspace(-np.pi, np.pi, int(grid_points), endpoint=True)
    best_phi = 0.0
    best_fid = -1.0
    best_rho = rho
    for phi in phi_grid:
        U = (1j * n_L * float(phi)).expm()
        rho_corr = U * rho * U.dag()
        fid = float(qt.metrics.fidelity(target, rho_corr))
        if fid > best_fid:
            best_fid = fid
            best_phi = float(phi)
            best_rho = rho_corr
    if label:
        print(f"  [{label}] optimal phase phi = {best_phi:.4f} rad")
    return best_rho


def plot_triptych(target_rho, undriven_rho, driven_rho, sim_time_ns, filename):
    t_us = sim_time_ns / 1000.0
    panels = [
        (r"t=0$\mu$s", target_rho),
        (fr"Undrive, t={t_us:.0f}$\mu$s", undriven_rho),
        (fr"Driven, t={t_us:.0f}$\mu$s", driven_rho),
    ]

    fig = plt.figure(figsize=(13.2, 4.6))
    for idx, (title, rho) in enumerate(panels):
        rho_abs = qt.Qobj(np.abs(rho.full()), dims=rho.dims)
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        matrix_histogram(
            rho_abs,
            LABELS_2Q_LOGICAL,
            LABELS_2Q_LOGICAL,
            fig=fig,
            ax=ax,
            limits=[0.0, 0.5],
            bar_style="abs",
            color_style="abs",
            color_limits=[0.0, 0.5],
            cmap=PLOT_CMAP,
            colorbar=False,
        )
        ax.tick_params(axis="x", labelsize=6, pad=1)
        ax.tick_params(axis="y", labelsize=6, pad=1)
        ax.set_title(title)
        ax.view_init(azim=-55, elev=45)
        ax.set_zlim(0.0, 0.5)
        ax.set_zticks([])

    shared_norm = mpl.colors.Normalize(vmin=0.0, vmax=0.5)
    shared_mappable = mpl.cm.ScalarMappable(norm=shared_norm, cmap=PLOT_CMAP)
    shared_mappable.set_array([])
    fig.subplots_adjust(wspace=0.2, right=0.87)
    cax = fig.add_axes([0.89, 0.2, 0.015, 0.62])
    cbar = fig.colorbar(shared_mappable, cax=cax)
    cbar.set_label(r"$|\rho_L|$")

    out = Path(filename).resolve()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved Bell comparison plot to: {out}")


def plot_single_matrix(rho, title, filename):
    rho_abs = qt.Qobj(np.abs(rho.full()), dims=rho.dims)
    fig = plt.figure(figsize=(5.0, 4.6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    matrix_histogram(
        rho_abs,
        LABELS_2Q_LOGICAL,
        LABELS_2Q_LOGICAL,
        fig=fig,
        ax=ax,
        limits=[0.0, 0.5],
        bar_style="abs",
        color_style="abs",
        color_limits=[0.0, 0.5],
        cmap=PLOT_CMAP,
        colorbar=False,
    )
    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    ax.set_title(title)
    ax.view_init(azim=-55, elev=45)
    ax.set_zlim(0.0, 0.5)
    shared_norm = mpl.colors.Normalize(vmin=0.0, vmax=0.5)
    shared_mappable = mpl.cm.ScalarMappable(norm=shared_norm, cmap=PLOT_CMAP)
    shared_mappable.set_array([])
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.18, 0.03, 0.64])
    cbar = fig.colorbar(shared_mappable, cax=cax)
    cbar.set_label(r"$|\rho_L|$")
    out = Path(filename).resolve()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved logical matrix plot to: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Load saved two-dual-rail data and compare final logical states with Bell |Phi+>."
    )
    parser.add_argument(
        "--undriven-data",
        type=str,
        default="final_data_undriven.pkl",
        help="Pickle output from two_dual_rail_simulation.py (undriven case).",
    )
    parser.add_argument(
        "--driven-data",
        type=str,
        default="final_data_driven.pkl",
        help="Pickle output from two_dual_rail_simulation.py (driven case).",
    )
    parser.add_argument(
        "--triptych-output",
        type=str,
        default="bell_state_comparison.png",
        help="Output path for Bell target / undriven final / driven final comparison.",
    )
    parser.add_argument(
        "--driven-output",
        type=str,
        default="bell_state_driven_final.png",
        help="Output path for driven final matrix plot.",
    )
    parser.add_argument(
        "--undriven-output",
        type=str,
        default="bell_state_undriven_final.png",
        help="Output path for undriven final matrix plot.",
    )
    args = parser.parse_args()

    undriven_data = load_case_data(args.undriven_data)
    driven_data = load_case_data(args.driven_data)

    # Step 1: ensemble average already stored in pickle (logical_rhos are averaged over trajectories).
    undriven_rho = to_qobj_4x4(undriven_data["logical_rhos"][-1])
    driven_rho = to_qobj_4x4(driven_data["logical_rhos"][-1])
    target_rho = bell_phi_plus_density()

    sim_time_ns = float(driven_data["time_points"][-1])

    # Step 2: phase correction — optimize Kerr-like phase to maximize Bell fidelity.
    print("Applying phase correction...")
    undriven_rho = phase_corrected_logical_rho(undriven_rho, label="Undriven")
    driven_rho = phase_corrected_logical_rho(driven_rho, label="Driven")

    # Step 3: print fidelity.
    fid_undriven = bell_fidelity(undriven_rho)
    fid_driven = bell_fidelity(driven_rho)
    t_us = sim_time_ns / 1e3
    print(f"Bell-state fidelity (undriven, t={t_us:.1f} us, after phase correction): {fid_undriven:.6f}")
    print(f"Bell-state fidelity (driven,   t={t_us:.1f} us, after phase correction): {fid_driven:.6f}")

    plot_triptych(target_rho, undriven_rho, driven_rho, sim_time_ns, args.triptych_output)
    plot_single_matrix(undriven_rho, "Undriven Final (Logical Matrix)", args.undriven_output)
    plot_single_matrix(driven_rho, "Driven Final (Logical Matrix)", args.driven_output)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
