"""Resolve explicitly supplied research directories without inspecting their contents."""
import os
from pathlib import Path

def _required_directory(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to an authorized existing directory; see docs/INPUT_CONTRACT.md.")
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{name} must name an existing absolute directory.")
    return path.resolve()

def analysis_root():
    path = _required_directory('ILI_ANALYSIS_ROOT')
    if not all((path / child).is_dir() for child in ('protocol', 'audit', 'results')):
        raise ValueError('The analysis directory must contain protocol, audit and results directories.')
    return path

def project_root():
    return _required_directory('ILI_PROJECT_ROOT')
