import os
import sys
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

_DATA_PLOT = Path(__file__).resolve().parent
_SAFE = _DATA_PLOT.parent
sys.path.insert(0, str(_DATA_PLOT))
sys.path.insert(0, str(_SAFE))

import numpy as np
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt
import matplotlib as mpl

# Dual-rail static sweep + Bell comparison
_SIM_BINOM = _SAFE / "simulations" / "binomial_density_matrix"
_SIM_DUAL = _SAFE / "simulations" / "dual_rail_bell"
_QEC_PDF = _DATA_PLOT / "figure4.pdf"

_PI_PATH = _DATA_PLOT.parent / "plot_instruction"
if not _PI_PATH.is_file():
    _PI_PATH = _DATA_PLOT / "plot_instruction"
_plot_instruction = SourceFileLoader("plot_instruction_qec", str(_PI_PATH)).load_module()

# Manuscript typography and layout (plot_instruction.normal_plot)
mpl.rcParams["svg.fonttype"] = "path"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams.update(_plot_instruction.normal_plot)

fig_width = _plot_instruction.fig_width
fig_height = _plot_instruction.fig_height

_QEC_FS = float(mpl.rcParams["axes.labelsize"])
_QEC_FS_LEGEND = float(mpl.rcParams["legend.fontsize"])
_QEC_FS_TICK = float(mpl.rcParams["xtick.labelsize"])
_QEC_FS_TITLE = float(mpl.rcParams["font.size"])
_QEC_FS_PANEL = _QEC_FS + 1.0
LW = float(mpl.rcParams["lines.linewidth"])


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_full_box(ax):
    """Keep all four spines visible (plot_instruction axes frame)."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)


_CORNER_CELLS = ((0, 0), (0, 3), (3, 0), (3, 3))


def _annotate_corner_rho_cells(ax, arr, *, threshold=1e-3):
    """Label non-zero corner blocks with two-decimal values centered in each cell."""
    for row, col in _CORNER_CELLS:
        val = float(arr[row, col])
        if val <= threshold:
            continue
        text_color = "white" if val > 0.35 else "black"
        ax.text(
            col,
            row,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=_QEC_FS_TICK,
            color=text_color,
        )


def _plot_bell_triptych(axes_2d, undriven_pkl: Path, driven_pkl: Path):
    """Three 2D |(rho_L)_{mu,nu}| heatmaps (same data as bell_state.plot_triptych)."""
    bell = _load_module("bell_state_plot_all", _SIM_DUAL / "bell_state.py")
    LABELS = bell.LABELS_2Q_LOGICAL
    PLOT_CMAP = mpl.colormaps["Reds"]  # 0 → white/light, 0.5 → red
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.5)

    undriven_data = bell.load_case_data(str(undriven_pkl))
    driven_data = bell.load_case_data(str(driven_pkl))
    undriven_rho = bell.to_qobj_4x4(undriven_data["logical_rhos"][-1])
    driven_rho = bell.to_qobj_4x4(driven_data["logical_rhos"][-1])
    target_rho = bell.bell_phi_plus_density()
    sim_time_ns = float(driven_data["time_points"][-1])
    t_us = sim_time_ns / 1000.0

    undriven_rho = bell.phase_corrected_logical_rho(undriven_rho, label="")
    driven_rho = bell.phase_corrected_logical_rho(driven_rho, label="")

    panels = [
        (r"$t=0\,\mu s$", target_rho, False),
        (rf"Undriven, $t={t_us:.0f}\,\mu s$", undriven_rho, False),
        (rf"Driven, $t={t_us:.0f}\,\mu s$", driven_rho, True),
    ]
    last_im = None
    tick_pos = np.arange(4)
    for i, (ax, (title, rho, zero_center)) in enumerate(zip(axes_2d, panels)):
        arr = np.abs(np.asarray(rho.full(), dtype=complex))
        if zero_center:
            arr[1:3, 1:3] = 0.0
        last_im = ax.imshow(
            arr,
            cmap=PLOT_CMAP,
            norm=norm,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
            extent=(-0.5, 3.5, 3.5, -0.5),
        )
        _annotate_corner_rho_cells(ax, arr)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(
            LABELS,
            rotation=0,
            ha="center",
            va="top",
            fontsize=_QEC_FS_TICK,
        )
        ax.set_yticks(tick_pos)
        if i == 0:
            ax.set_yticklabels(LABELS, fontsize=_QEC_FS_TICK)
        else:
            ax.set_yticklabels([])
        ax.tick_params(
            axis="both",
            which="both",
            length=0,
            width=0,
            pad=1.5,
        )
        ax.set_title(title, fontsize=_QEC_FS_TITLE)
        _ensure_full_box(ax)
    return last_im


def _add_wigner_colorbar(fig, axes_w, cf_wigner):
    """Colorbar for row-0 Wigner strip; same geometry recipe as Bell row for alignment."""
    pos0 = axes_w[0].get_position()
    pos2 = axes_w[2].get_position()
    pad, cw = 0.014, 0.012
    cax = fig.add_axes([pos2.x1 + pad, pos0.y0, cw, pos0.height])
    vmin, vmax = cf_wigner.get_clim()
    tick_vals = np.linspace(vmin, vmax, 7)
    cbar = fig.colorbar(cf_wigner, cax=cax, ticks=tick_vals, format="%.2f")
    cbar.ax.minorticks_off()
    cbar.set_label("")
    cbar.ax.set_title(r"$W(I,Q)$", pad=4, fontsize=_QEC_FS)
    cbar.ax.tick_params(labelsize=_QEC_FS)


def _add_bell_colorbar(fig, axes_2d, mappable):
    pos0 = axes_2d[0].get_position()
    pos2 = axes_2d[2].get_position()
    pad, cw = 0.014, 0.012
    cax = fig.add_axes([pos2.x1 + pad, pos0.y0, cw, pos0.height])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("")
    cbar.ax.set_title(r"$(\rho_L)_{\mu\nu}$", pad=4, fontsize=_QEC_FS)
    cbar.ax.tick_params(labelsize=_QEC_FS)

# ============================================================
# Figure helpers
# ============================================================


def _apply_qec_typography_2d(ax_bin, axes_w, ax_d):
    """Unify tick/label/title font sizes on panels (a), (b), (c)."""
    for ax in (ax_bin, ax_d):
        ax.tick_params(axis="both", labelsize=_QEC_FS_TICK, which="major")
        ax.xaxis.label.set_fontsize(_QEC_FS)
        ax.yaxis.label.set_fontsize(_QEC_FS)
        if ax.get_title():
            ax.title.set_fontsize(_QEC_FS_TITLE)
        _ensure_full_box(ax)
    for ax in axes_w:
        ax.tick_params(axis="both", labelsize=_QEC_FS_TICK, which="major")
        ax.xaxis.label.set_fontsize(_QEC_FS)
        if ax.yaxis.label.get_text():
            ax.yaxis.label.set_fontsize(_QEC_FS)
        if ax.get_title():
            ax.title.set_fontsize(_QEC_FS_TITLE)
        _ensure_full_box(ax)
    for ax in (ax_bin, ax_d):
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(_QEC_FS_LEGEND)


def _gamma_phi_label(a: str, b: str, suffix: str = "") -> str:
    """Dephasing rate label: gamma_{phi;b,a} (not a -> b arrow notation)."""
    if suffix:
        return rf"$\gamma_{{\phi;{b},{a}}}^{{{suffix}}}$"
    return rf"$\gamma_{{\phi;{b},{a}}}$"


def _relabel_dual_rail_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        _gamma_phi_label(r"0_L", r"1_L", "(1)"),
        _gamma_phi_label(r"0_L", r"1_L", "(2)"),
    ]
    for i in range(min(2, len(labels))):
        labels[i] = new_labels[i]
    ax.legend(handles, labels, loc="lower right", fontsize=_QEC_FS_LEGEND)


def _align_row_axes(left_ax, ref_ax):
    """Match left-column axis box to the right-column reference (same row): aligned tops/bottoms."""
    pl = left_ax.get_position()
    pr = ref_ax.get_position()
    left_ax.set_position([pl.x0, pr.y0, pl.width, pr.height])


def _uniform_title_pad(axes, pad_pts: float = 5.0):
    for ax in axes:
        t = ax.get_title()
        if t:
            ax.set_title(t, pad=pad_pts, fontsize=_QEC_FS_TITLE)


def _tag_panels_in_figure(fig, ax_a, ax_b, ax_c, ax_d_panel):
    """Panel letters in figure coords (after layout) so they do not overlap y-axis ticks."""
    kw = dict(
        ha="right",
        va="bottom",
        fontsize=_QEC_FS_PANEL,
        fontweight="bold",
        transform=fig.transFigure,
    )
    dx_fig, dy_fig = 0.006, 0.016
    for ax, lab in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d_panel, "(d)")):
        pos = ax.get_position()
        fig.text(pos.x0 - dx_fig, pos.y1 + dy_fig, lab, **kw)


# ============================================================
# Create figure: row0 = binomial (bononmial_plot) + Wigner triptych;
# row1 = dual-rail rate + Bell 2D triptych
# Panel tags: (a)(b) row 0; (c)(d) row 1 (one tag per row-half).
# ============================================================
fig = plt.figure(figsize=(3 * fig_width, 2 * fig_height))
# 2×2 outer grid: same left-column width (a)/(c) and same right-column width (b)/(d).
# A 3-column layout lets tight_layout treat rows differently; this keeps columns aligned.
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.0, 2.35],
    height_ratios=[1.0, 1.0],
    wspace=0.13,
    hspace=0.32,
)
ax_bin = fig.add_subplot(gs[0, 0])
ax_d = fig.add_subplot(gs[1, 0])
gs_w = gs[0, 1].subgridspec(1, 3, wspace=0.18)
ax_w0 = fig.add_subplot(gs_w[0, 0])
ax_w1 = fig.add_subplot(gs_w[0, 1])
ax_w2 = fig.add_subplot(gs_w[0, 2])
gs_bell = gs[1, 1].subgridspec(1, 3, wspace=0.18)
ax_b0 = fig.add_subplot(gs_bell[0, 0])
ax_b1 = fig.add_subplot(gs_bell[0, 1])
ax_b2 = fig.add_subplot(gs_bell[0, 2])

# ---- Row 0: binomial dephasing + Wigner triptych ----
bon = _load_module(
    "binomial_row_fig4",
    _DATA_PLOT / "binonmial_code_simulation" / "binomial_row.py",
)
print("Computing binomial static curves + Wigner panels (may take a minute)...")
cf_wigner = bon.plot_binomial_and_wigner_row(
    fig, ax_bin, [ax_w0, ax_w1, ax_w2], style_mod=None
)
ax_bin.set_title("")

# ---- (c) Dual-rail rate (static_sweep quasi-energy derivatives) ----
# Binomial row / cavity_self_kerr can leave `system` / `hamiltonian_generator`
# imported from binomial_density_matrix; clear before dual-rail imports.
for _mod in ("system", "hamiltonian_generator", "noise_generator", "2dualrail"):
    sys.modules.pop(_mod, None)
static_sweep = _load_module("static_sweep_plot_all", _SIM_DUAL / "static_sweep.py")
print("Computing dual-rail static sweep (Floquet); this may take a minute...")
det_dr, dc1, dc2 = static_sweep._compute_norm_from_model({})
static_sweep.plot_dual_rail_rate(det_dr, dc1, dc2, ax=ax_d, title="")
for line in ax_d.get_lines():
    if line.get_marker() == "*":
        line.set_markersize(7)
    else:
        line.set_linewidth(LW)
_relabel_dual_rail_legend(ax_d)

# ---- Typography on (a), (b), (c) ----
_apply_qec_typography_2d(ax_bin, [ax_w0, ax_w1, ax_w2], ax_d)
ax_bin.legend(
    loc="lower right",
    fontsize=_QEC_FS_LEGEND,
    handlelength=1.5,
    labelspacing=0.18,
    borderpad=0.18,
    borderaxespad=0.25,
)

# ---- (d) Bell-state logical density triptych (bell_state.py) ----
undriven_pkl = _SIM_DUAL / "final_data_undriven.pkl"
driven_pkl = _SIM_DUAL / "final_data_driven.pkl"
_bell_ok = undriven_pkl.is_file() and driven_pkl.is_file()
_bell_im = None
if _bell_ok:
    _bell_im = _plot_bell_triptych([ax_b0, ax_b1, ax_b2], undriven_pkl, driven_pkl)
else:
    for a in (ax_b0, ax_b1, ax_b2):
        a.text(0.05, 0.5, "Missing final_data_*.pkl", transform=a.transAxes)

# Manual layout keeps row halves aligned after subplots_adjust.
fig.subplots_adjust(left=0.10, right=0.88, top=0.93, bottom=0.10)
# Same vertical span for (a) vs (b) strip and (c) vs (d) strip.
_align_row_axes(ax_bin, ax_w0)
_align_row_axes(ax_d, ax_b0)
_uniform_title_pad(
    [ax_w0, ax_w1, ax_w2]
    + ([ax_b0, ax_b1, ax_b2] if _bell_ok else []),
    pad_pts=5.0,
)
_tag_panels_in_figure(fig, ax_bin, ax_w0, ax_d, ax_b0)
_add_wigner_colorbar(fig, [ax_w0, ax_w1, ax_w2], cf_wigner)
if _bell_ok and _bell_im is not None:
    _add_bell_colorbar(fig, [ax_b0, ax_b1, ax_b2], _bell_im)
_QEC_PDF.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(_QEC_PDF, bbox_inches="tight")
print(f"Saved {_QEC_PDF}")
if matplotlib.get_backend().lower() != "agg":
    plt.show()
