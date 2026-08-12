# SAFE — cavity dephasing figures & simulations

Reproducible code and data for generating the manuscript figures (2–4, 6–8) for the cavity-dephasing project.

## Requirements

- Python **3.11+**
- A working TeX installation (for `matplotlib` `text.usetex` figure styling)
- macOS / Linux recommended

## Install

```bash
git clone https://github.com/yunweilu/cavity-dephasing-safe.git
cd cavity-dephasing-safe
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

For Jupyter notebooks under `simulations/`:

```bash
pip install -e ".[notebooks]"
```

After a successful editable install you can import the helper package from anywhere:

```python
from helper.system import Hamiltonian
from helper.noise_generator import GenerateNoise
```

## Repository layout

| Path | Contents |
|------|----------|
| `helper/` | Shared Hamiltonian / noise utilities (installed as a package) |
| `data_plot/` | `generate_figure*.py` scripts and output PDFs |
| `simulations/` | Simulation generators and saved `.pkl` / `.npz` data used by the figures |
| `plot_instruction` | Shared matplotlib style settings |
| `pyproject.toml` | Package metadata and dependencies |

## Regenerate figures

Run scripts from `data_plot/` so relative paths resolve correctly:

```bash
cd data_plot
export MPLBACKEND=Agg   # non-interactive backend (recommended for scripts)

python generate_figure2.py
python generate_figure3.py
python generate_figure4.py
python generate_figure6.py
python generate_figure7.py
python generate_figure8.py
```

Outputs are written as `figure2.pdf` … `figure8.pdf` (and `figure2_inse.pdf`, `noise_spectrum_plot.png`) in `data_plot/`.

### Data used by each figure

| Figure | Input data |
|--------|------------|
| 2 | `simulations/rates_bundle/rates_bundle.pkl` |
| 3 | `simulations/total_rate/total_rate.pkl`, `simulations/ramsey_sigmax/ramsey_sigmax_data.pkl` |
| 4 | `simulations/binomial_density_matrix/avg_density_matrix_over_time_undriven_driven.pkl`, `simulations/dual_rail_bell/final_data_{undriven,driven}.pkl` |
| 6 | none (computed on the fly) |
| 7 | none (noise generated on the fly) |
| 8 | `simulations/preparation_fidelity/preparation_fidelity_data.npz` |

See `simulations/README.md` for how those data files were produced.

## Dependencies

Pinned versions match the `fluxonium` development environment and are listed in `pyproject.toml`, including:

`numpy`, `scipy`, `matplotlib`, `qutip`, `scqubits`, `colorednoise`, `joblib`, `jax` / `jaxlib`

## License

MIT (see `pyproject.toml`).
