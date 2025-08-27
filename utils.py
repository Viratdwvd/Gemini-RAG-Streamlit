# def find_files(root, exts): return []
# utils.py - small helper utilities

import os
from typing import List

def find_files(root: str, exts: List[str] = None) -> List[str]:
    """Return list of file paths under root matching extensions in exts (e.g., ['.pdf', '.txt'])."""
    exts = exts or [".pdf", ".txt"]
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if any(f.lower().endswith(e) for e in exts):
                paths.append(os.path.join(dirpath, f))
    return paths
