# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:03:11 2025

@author: Alberto
"""

from pathlib import Path

# Path to this file: .../statistical-signal-processing/codigo/config.py
_THIS_FILE = Path(__file__).resolve()

# Project root = two levels up: codigo/ → my_project/
PROJECT_ROOT = _THIS_FILE.parents[1]

# Useful subfolders
DATA_DIR = PROJECT_ROOT / "data"
CODE_DIR = PROJECT_ROOT / "codigo"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"