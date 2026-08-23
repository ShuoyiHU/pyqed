import runpy
import sys
from pathlib import Path

import setuptools


def test_linux_su2_extension_links_blas_and_lapack(monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.delenv("PYQED_BUILD_EXTENSIONS", raising=False)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    setup_module = runpy.run_path(str(repository / "setup.py"))

    monkeypatch.setenv("PYQED_BUILD_EXTENSIONS", "1")
    monkeypatch.setenv("PYQED_EXTENSION_GROUPS", "mps")
    monkeypatch.setattr(sys, "platform", "linux")
    extensions = setup_module["_optional_extensions"]()

    assert len(extensions) == 1
    extension = extensions[0]
    assert extension.name == "pyqed.mps.nonabelian._su2_kernel"
    assert ("PYQED_USE_CBLAS", "1") in extension.define_macros
    assert ("PYQED_USE_LAPACK", "1") in extension.define_macros
    assert "blas" in extension.libraries
    assert "lapack" in extension.libraries
