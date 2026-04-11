"""
Streamlit Cloud entry point.

Streamlit's deploy step expects a real directory here; a git symlink is not
always treated as a directory after clone. The implementation stays in
``05_annotation_app/app.py``; we delegate with ``runpy`` so ``__file__`` in the
real module resolves to that path (corpus/codebook paths stay correct).
"""
from __future__ import annotations

import os
import runpy

_ROOT = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.normpath(os.path.join(_ROOT, "..", "05_annotation_app", "app.py"))

if not os.path.isfile(_TARGET):
    raise FileNotFoundError(
        "Missing 05_annotation_app/app.py — Cloud main file should stay "
        f"annotation_app_cloud/app.py; expected {_TARGET!r}"
    )

runpy.run_path(_TARGET, run_name="__main__")
