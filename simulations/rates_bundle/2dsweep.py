"""Generate rates_bundle.pkl (2D Floquet rate sweeps) and plot with historical system.py.

Usage (from this directory):
  python 2dsweep.py              # full 100x100 grids → rates_bundle.pkl + PDF
  python 2dsweep.py --plot-only  # load existing rates_bundle.pkl and plot
  python 2dsweep.py --n 3        # small smoke-test grid (writes same pickle name unless --out)
"""
from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm

from system import Hamiltonian

_HERE = Path(__file__).resolve().parent

# ── System parameters (same as qenergy_vs_app.py) ──────────────────────────
Ej = 30.19
Ec = 0.1
omega_c1 = 5.226
omega_c2 = 8.135
phi_ex = 0.2
bare_dim = [10, 1, 6]
trunc_dim = [5, 1, 4]
detu = 2e-3


def build_hamiltonian():
    return Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)


def dressed_omega(sc):
    position = sc.state_index((1, 0, 0), sc.original_dim)
    return float(sc.H_dressed[position, position].real)


def flux_derivative():
    delta_phi = 1e-6
    sc_plus = Hamiltonian(phi_ex + delta_phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
    sc_minus = Hamiltonian(phi_ex - delta_phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
    pos_p = sc_plus.state_index((1, 0, 0), sc_plus.original_dim)
    pos_m = sc_minus.state_index((1, 0, 0), sc_minus.original_dim)
    return (
        sc_plus.H_dressed[pos_p, pos_p].real - sc_minus.H_dressed[pos_m, pos_m].real
    ) / (2 * delta_phi)


def static_rate(sc, omegad, amplitude):
    der = sc.equasi_gradient(amplitude, omegad)[0]
    return float(np.abs(der) / 2 / np.pi)


def sweep(sc, omega_ds, amplitudes, label):
    n_amp, n_det = len(amplitudes), len(omega_ds)
    tasks = [
        (i_amp, i_det, omega_ds[i_det], amplitudes[i_amp])
        for i_amp in range(n_amp)
        for i_det in range(n_det)
    ]
    print(f"{label}: {n_amp} x {n_det} = {len(tasks)} points …")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(static_rate)(sc, omegad, amp) for (_, _, omegad, amp) in tasks
    )
    rates = np.zeros((n_amp, n_det), dtype=float)
    for idx, (i_amp, i_det, _, _) in enumerate(tasks):
        rates[i_amp, i_det] = results[idx]
    return rates


def generate_bundle(n: int, out_path: Path):
    sc = build_hamiltonian()
    omega_s = dressed_omega(sc)
    print(f"Dressed SNAIL frequency: {omega_s / (2 * np.pi):.6f} GHz")

    omega_ds_full = np.linspace(6.159 - detu, 6.159 + 0.025, n) * 2 * np.pi
    amplitudes_full = 2 * np.pi * np.linspace(0.01e-3, 10e-3, n)
    omega_ds = np.linspace(6.159 - detu, 6.159 + detu, n) * 2 * np.pi
    amplitudes = 2 * np.pi * np.linspace(0.01e-3, 0.2e-3, n)

    rates = sweep(sc, omega_ds_full, amplitudes_full, "rates (full)")
    rates2 = sweep(sc, omega_ds, amplitudes, "rates2 (zoom)")

    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "N_amp": n,
        "N_det": n,
        "detu": detu,
        "omega_ds": omega_ds,
        "amplitudes": amplitudes,
        "rates": rates,
        "rates2": rates2,
        "shapes": {
            "rates": rates.shape,
            "rates2": rates2.shape,
            "omega_ds": omega_ds.shape,
            "amplitudes": amplitudes.shape,
        },
        "units": {
            "omega_ds": "rad/s",
            "amplitudes": "rad/s",
            "rates": "Hz",
            "rates2": "Hz",
        },
        "omega_s": omega_s,
        "amplitudes_full": amplitudes_full,
        "omega_ds_full": omega_ds_full,
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Wrote {out_path}")
    return payload


def add_resonant_region(ax, x_min, x_max, y_min, y_max, label=True):
    y_line = np.linspace(y_min, y_max, 800)
    x_left = -y_line
    x_right = y_line
    x_left_clip = np.clip(x_left, x_min, x_max)
    x_right_clip = np.clip(x_right, x_min, x_max)
    ax.fill_betweenx(
        y_line, x_left_clip, x_right_clip,
        facecolor="white", alpha=0.22, hatch="///",
        edgecolor="gray", linewidth=0.0, zorder=3,
    )
    mask_l = (x_left >= x_min) & (x_left <= x_max)
    mask_r = (x_right >= x_min) & (x_right <= x_max)
    ax.plot(x_left[mask_l], y_line[mask_l], "-", color="gray", lw=1.2, zorder=4)
    lbl = "resonant region" if label else None
    ax.plot(
        x_right[mask_r], y_line[mask_r], "-", color="gray", lw=1.2,
        zorder=4, label=lbl,
    )


def plot_bundle(data, out_pdf: Path):
    omega_s = dressed_omega(build_hamiltonian())
    domega_dphi = flux_derivative()
    print(f"Dressed SNAIL frequency: {omega_s / (2 * np.pi):.4f} GHz")
    print(f"|d(omega_transmon)/d(phi)| / 2pi = {np.abs(domega_dphi) / (2 * np.pi):.6f} GHz")

    rates = data["rates"]
    rates2 = data["rates2"]
    omega_ds1 = data["omega_ds_full"]
    amplitudes1 = data["amplitudes_full"]
    omega_ds2 = data["omega_ds"]
    amplitudes2 = data["amplitudes"]
    print(f"Loaded rates: {rates.shape}, rates2: {rates2.shape}")

    det1 = (omega_ds1 / (2 * np.pi) - omega_s / (2 * np.pi)) * 1e3
    amps1_mhz = amplitudes1 / (2 * np.pi) * 1e3
    X1, Y1 = np.meshgrid(det1, amps1_mhz)
    Z1 = np.array(rates, dtype=float, copy=True)
    Z1[~np.isfinite(Z1) | (Z1 <= 0)] = np.nan
    Z1 = Z1 / (np.abs(domega_dphi) / (2 * np.pi))

    det2 = (omega_ds2 / (2 * np.pi) - omega_s / (2 * np.pi)) * 1e3
    amps2_mhz = amplitudes2 / (2 * np.pi) * 1e3
    X2, Y2 = np.meshgrid(det2, amps2_mhz)
    Z2 = np.array(rates2, dtype=float, copy=True)
    Z2[~np.isfinite(Z2) | (Z2 <= 0)] = np.nan
    Z2 = Z2 / (np.abs(domega_dphi) / (2 * np.pi))

    norm = LogNorm(vmin=1e-3, vmax=1e-1)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(cmap(0))

    fig, ax_main = plt.subplots(figsize=(8, 5), constrained_layout=True)
    pcm1 = ax_main.pcolormesh(X1, Y1, Z1, shading="nearest", cmap=cmap, norm=norm)
    ax_main.set_xlim(det1[0], det1[-1])
    ax_main.set_ylim(amps1_mhz[0], float(amps1_mhz[-1]))
    add_resonant_region(
        ax_main, det1[0], det1[-1], amps1_mhz[0], float(amps1_mhz[-1]), label=True
    )
    ax_main.set_xlabel(r"$-\Delta_{bd}/2\pi$ (MHz)")
    ax_main.set_ylabel(r"$\Omega_0/2\pi$ (MHz)")
    ax_main.legend(loc="upper right", frameon=False, fontsize=9)

    D0_undriv = Z1[0, -1]
    print(f"Undriven D_0 = {D0_undriv:.4e}")
    cbar = fig.colorbar(pcm1, ax=ax_main, pad=0.02)
    cbar.set_label(r"$D_0$")
    cbar.ax.text(
        3.8, D0_undriv, "(undriven)", fontsize=7, va="center", ha="left",
        transform=cbar.ax.get_yaxis_transform(),
    )

    ax_ins = ax_main.inset_axes([0.42, 0.55, 0.28, 0.38])
    x0_ins, x1_ins = -1.0, det2[-1]
    y0_ins, y1_ins = amps2_mhz[0], amps2_mhz[-1]
    ax_ins.pcolormesh(X2, Y2, Z2, shading="nearest", cmap=cmap, norm=norm)
    ax_ins.set_xlim(x0_ins, x1_ins)
    ax_ins.set_ylim(y0_ins, y1_ins)
    add_resonant_region(ax_ins, x0_ins, x1_ins, y0_ins, y1_ins, label=False)

    analytic_det_mhz = 10.0 * amps2_mhz
    mask_pos = (analytic_det_mhz >= x0_ins) & (analytic_det_mhz <= x1_ins)
    mask_neg = (-analytic_det_mhz >= x0_ins) & (-analytic_det_mhz <= x1_ins)
    ax_ins.plot(analytic_det_mhz[mask_pos], amps2_mhz[mask_pos], "w--", lw=1.2)
    if np.any(mask_neg):
        ax_ins.plot(-analytic_det_mhz[mask_neg], amps2_mhz[mask_neg], "w-.", lw=1.2)
    ax_ins.set_xlabel("")
    ax_ins.set_ylabel("")
    ax_ins.tick_params(labelsize=10)
    for spine in ax_ins.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.5)

    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=100,
        help="grid size N_amp = N_det (default 100, paper bundle)",
    )
    parser.add_argument(
        "--out", type=Path, default=_HERE / "rates_bundle.pkl",
        help="output pickle path",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="skip generation; load --out and plot",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="generate pickle only",
    )
    parser.add_argument(
        "--pdf", type=Path,
        default=_HERE / "combined_full_and_subset_shared_cbar.pdf",
    )
    args = parser.parse_args()

    if args.plot_only:
        with args.out.open("rb") as f:
            data = pickle.load(f)
    else:
        data = generate_bundle(args.n, args.out)

    if not args.no_plot:
        plot_bundle(data, args.pdf)


if __name__ == "__main__":
    main()
