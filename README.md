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
python -m ipykernel install --user --name=cavity-dephasing-safe --display-name="Python (SAFE)"
```

Then select the **Python (SAFE)** kernel in Jupyter / VS Code / Cursor.

After a successful editable install you can import the helper package from anywhere **while your working directory is inside the cloned repo**:

```python
from helper.system import Hamiltonian
from helper.noise_generator import GenerateNoise
from helper.paths import repo_root, simulation_dir
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

You can run scripts from **any cwd** (paths are resolved from `__file__`):

```bash
cd cavity-dephasing-safe
export MPLBACKEND=Agg   # non-interactive backend (recommended for scripts)

python data_plot/generate_figure2.py
python data_plot/generate_figure3.py
python data_plot/generate_figure4.py
python data_plot/generate_figure6.py
python data_plot/generate_figure7.py
python data_plot/generate_figure8.py
```

Outputs are written into `data_plot/` (`figure2.pdf` … `figure8.pdf`, plus `figure2_inse.pdf` and `noise_spectrum_plot.png`).

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

## Run simulation notebooks

Start Jupyter **from the cloned repository** (repo root or any subdirectory), then open:

| Notebook | Folder |
|----------|--------|
| `full_simulation_ramsey.ipynb` | `simulations/ramsey_sigmax/` |
| `full_simulation_totalrate.ipynb` | `simulations/total_rate/` |
| `full_simulation_plot.ipynb` | `simulations/preparation_fidelity/` |

The first cell uses `helper.paths.simulation_dir(...)` to `chdir` into the correct simulation folder and put local modules (`system.py`, `hamiltonian_generator.py`, `noise_generator.py`) on `sys.path`.

## Dependencies

Pinned versions match the `fluxonium` development environment and are listed in `pyproject.toml`, including:

`numpy`, `scipy`, `matplotlib`, `qutip`, `scqubits`, `colorednoise`, `joblib`, `jax` / `jaxlib`

## License

MIT (see `pyproject.toml`).
