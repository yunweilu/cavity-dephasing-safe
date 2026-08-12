import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

_DATA_PLOT = Path(__file__).resolve().parent
_SAFE = _DATA_PLOT.parent
sys.path.insert(0, str(_DATA_PLOT))
sys.path.insert(0, str(_SAFE))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from drive_detuning_converter import A_to_detuning_mhz

_style_path = _SAFE / "plot_instruction"
_style_mod = SourceFileLoader("plot_instruction_mod", str(_style_path)).load_module()
normal_plot = _style_mod.normal_plot
add_axis_arrows = _style_mod.add_axis_arrows

_PREP_NPZ = _SAFE / "simulations" / "preparation_fidelity" / "preparation_fidelity_data.npz"
data = np.load(_PREP_NPZ, allow_pickle=True)
ramp_times = data['ramp_times']

# ── Style (normal_plot from plot_instruction) ────────────────────
mpl.rcParams.update(normal_plot)

colors3 = ['#cbc9e2', '#9e9ac8', '#6a51a3']

fig, ax = plt.subplots()

A_values = data['A_values']
infidelity_matrix = data['infidelity_matrix']
std_error_matrix = data['std_error_matrix']

all_A = list(A_values) + [data['A_fixed']]
all_det = A_to_detuning_mhz(all_A)
det_per_A = all_det[:-1]
det_fixed_mhz = all_det[-1]

for i, (A, det_mhz) in enumerate(zip(A_values, det_per_A)):
    infidelity = infidelity_matrix[:, i]
    std_error = std_error_matrix[:, i]
    ax.errorbar(ramp_times, infidelity, yerr=std_error, fmt='-o',
                label=rf'$-\Delta_{{bd}}/2\pi$ = {det_mhz:.0f} MHz (DRAG)',
                markersize=4, capsize=2, elinewidth=1,
                color=colors3[i % len(colors3)])

sweep_results = data['sweep_results']
sweep_std_errors = data['sweep_std_errors']
ax.errorbar(ramp_times, sweep_results, yerr=sweep_std_errors, fmt='-^',
            label=rf'$-\Delta_{{bd}}/2\pi$ = {det_fixed_mhz:.0f} MHz (Gaussian)',
            markersize=4, capsize=2, elinewidth=1,
            color=colors3[2])

ax.set_xlabel('Ramp time (ns)')
ax.set_ylabel('Infidelity')
ax.legend()
ax.set_yscale('log')
add_axis_arrows(ax)
_out = _DATA_PLOT / "figure8.pdf"
plt.savefig(_out)
print(f"Saved: {_out}")
plt.close()
