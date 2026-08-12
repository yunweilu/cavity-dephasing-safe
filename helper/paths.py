"""Locate the cloned SAFE repository and simulation subfolders."""
from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | str | None = None) -> Path:
    """
    Find the SAFE repo root (directory with pyproject.toml, helper/, simulations/).

    Walks upward from ``start`` (default: cwd). After ``pip install -e .``, this
    still requires the working directory to be inside the clone (or a child).
    """
    start_path = Path(start or Path.cwd()).resolve()
    for path in (start_path, *start_path.parents):
        if (
            (path / "pyproject.toml").is_file()
            and (path / "helper").is_dir()
            and (path / "simulations").is_dir()
            and (path / "data_plot").is_dir()
        ):
            return path
    raise FileNotFoundError(
        "Could not locate the SAFE repository root from "
        f"{start_path}. Clone https://github.com/yunweilu/cavity-dephasing-safe, "
        "cd into that directory (or a subdirectory), then re-run."
    )


def simulation_dir(name: str, start: Path | str | None = None) -> Path:
    """Return ``<repo>/simulations/<name>`` and verify it exists."""
    path = repo_root(start) / "simulations" / name
    if not path.is_dir():
        raise FileNotFoundError(f"Simulation directory not found: {path}")
    return path


def data_plot_dir(start: Path | str | None = None) -> Path:
    """Return ``<repo>/data_plot``."""
    return repo_root(start) / "data_plot"
