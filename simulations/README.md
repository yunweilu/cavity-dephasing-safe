# SAFE simulations

Each subdirectory is self-contained: Monte Carlo / Floquet generator code, output data, and local `system.py` / `noise_generator.py` / `hamiltonian_generator.py` where needed.

Run generators from their folder (or set cwd accordingly) so relative output paths resolve.

| Folder | Output | Generator | Used by |
|--------|--------|-----------|---------|
| `total_rate/` | `total_rate.pkl` | `full_simulation_totalrate.ipynb` | `data_plot/generate_figure3.py` |
| `ramsey_sigmax/` | `ramsey_sigmax_data.pkl` | `full_simulation_ramsey.ipynb` | `data_plot/generate_figure3.py` |
| `preparation_fidelity/` | `preparation_fidelity_data.npz` | `full_simulation_plot.ipynb` | `data_plot/generate_figure8.py` |
| `binomial_density_matrix/` | `avg_density_matrix_over_time_undriven_driven.pkl` | `average_density_matrix_simulation.py` | `data_plot/generate_figure4.py` |
| `rates_bundle/` | `rates_bundle.pkl` | *(source notebook not in repo)* | `data_plot/generate_figure2.py` |
| `dual_rail_bell/` | `final_data_undriven.pkl`, `final_data_driven.pkl` | `two_dual_rail_simulation.py` | `data_plot/generate_figure4.py` |

Sources copied from `SNAIL/full_simulation copy/` (and `rates_bundle.pkl` from prior `SNAIL_new/static/`).
