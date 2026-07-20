from __future__ import annotations

import subprocess
import sys


def test_core_import_does_not_require_hnsw_backend():
    script = """
import builtins
import importlib.metadata

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'leann_backend_hnsw' or name.startswith('leann_backend_hnsw.'):
        raise AssertionError('core import must not require the HNSW backend')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
importlib.metadata.distributions = lambda: []
import leann.api
"""

    subprocess.run([sys.executable, "-c", script], check=True)
