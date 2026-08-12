import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Style from SNAIL_new/plot_instruction (normal_plot)
fig_width_pt = 246.0
inches_per_pt = 1.0 / 72.27
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width / 1.45

mpl.rcParams["svg.fonttype"] = "path"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams.update(
    {
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
    }
)


# Import noise generator from SAFE/helper
_SAFE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SAFE))
from helper.noise_generator import GenerateNoise


def main():
    # Setup from pinknoise.ipynb (lines 1-11)
    sample_rate = 1  # per ns
    tnoise_max = int(1e5)
    omega_ir = 2 * np.pi / tnoise_max
    S0 = 1e-5
    relative_psd_strength = S0**2
    num_realizations = 100
    ifwhite = False
    N = int(tnoise_max * sample_rate)

    print(f"sample_rate = {sample_rate} 1/ns")
    print(f"tnoise_max = {tnoise_max}")
    print(f"N = {N}")
    print(f"omega_ir = {omega_ir:.3e} rad/ns")
    print(f"S0 = {S0:.3e}")
    print(f"relative_PSD_strength = {relative_psd_strength:.3e}")
    print(f"num_realizations = {num_realizations}")
    print(f"ifwhite = {ifwhite}")

    gn = GenerateNoise(
        sample_rate=sample_rate,
        t_max=tnoise_max,
        relative_PSD_strength=relative_psd_strength,
        num_realizations=num_realizations,
        ifwhite=ifwhite,
    )

    # Generate trajectories and compute averaged PSD (same normalization as generator)
    trajectories = gn.generate_colored_noise()
    freqs = np.fft.rfftfreq(N, d=1 / sample_rate)
    psds = np.abs(np.fft.rfft(trajectories, axis=1)) ** 2 / sample_rate**2 / tnoise_max
    avg_psd = np.mean(psds, axis=0)

    # Exclude f=0 for log-log display
    mask = freqs > 0
    f_plot = freqs[mask]
    psd_plot = avg_psd[mask]

    fig, ax = plt.subplots()
    ax.loglog(f_plot, psd_plot, color="#7f7f7f", lw=1.4)
    ax.set_xlabel(r"$|\omega/2\pi|$ (GHz)")
    ax.set_ylabel(r"$S_{1/f}(\omega)$ ($\Phi_0^2\cdot$ns)")
    # Add arrowheads to axis ends
    ax.annotate(
        "",
        xy=(1.02, 0.0),
        xytext=(1.0, 0.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8),
    )
    ax.annotate(
        "",
        xy=(0.0, 1.02),
        xytext=(0.0, 1.0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8),
    )

    local_dir = Path(__file__).resolve().parent
    fig.tight_layout()
    out_pdf = local_dir / "figure7.pdf"
    out_png = local_dir / "noise_spectrum_plot.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
