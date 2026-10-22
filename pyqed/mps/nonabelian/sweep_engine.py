"""Explicitly build/load the source-hashed upstream reduced sweep engine.

Build with ``python -m pyqed.mps.nonabelian.sweep_engine``. Compilation is
separate from calculation timing; no silent fallback to an older binary.
"""
from pathlib import Path
import hashlib
import importlib.util
import os
import sys
import sysconfig


def build_directory():
    root = Path(__file__).resolve().parents[2]
    paths = [root / 'mps/nonabelian' / name for name in
             ('_su2_kernel.pyx', 'su2_dmrg_engine.cpp', 'su2_dmrg_engine.hpp', 'su2_coupling_core.hpp')]
    paths.append(root / 'linalg/davidson.hpp')
    paths.append(Path(__file__))
    digest = hashlib.sha256((sys.version + sys.platform + sysconfig.get_config_var('SOABI')
        + os.environ.get('PYQED_SWEEP_LIBRARIES', 'blas,lapack')).encode())
    for path in paths:
        digest.update(path.read_bytes())
    return Path(os.environ.get('PYQED_SWEEP_CACHE', Path.home()/'.cache/pyqed/sweeps')) / digest.hexdigest()[:20]


def load_engine():
    path = build_directory() / ('_sweep_engine' + sysconfig.get_config_var('EXT_SUFFIX'))
    if not path.exists():
        raise RuntimeError('Build the current engine first: python -m pyqed.mps.nonabelian.sweep_engine')
    name = 'pyqed.mps.nonabelian._sweep_engine'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__) != path:
            raise RuntimeError('Engine source changed in this process; restart Python')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def build():
    from setuptools import Extension, Distribution
    from Cython.Build import cythonize
    import numpy as np
    source = Path(__file__).resolve().parent
    output = build_directory()
    output.mkdir(parents=True, exist_ok=True)
    flags = ['-O3', '-std=c++17']
    link = ['-framework', 'Accelerate'] if sys.platform == 'darwin' else ['-ldl', '-fopenmp']
    if sys.platform != 'darwin':
        flags.append('-fopenmp')
    libraries = [] if sys.platform == 'darwin' else os.environ.get('PYQED_SWEEP_LIBRARIES', 'blas,lapack').split(',')
    library_dirs = [] if sys.platform == 'darwin' else [str(Path(sys.prefix)/'lib')]
    ext = Extension('_sweep_engine', [str(source/'_su2_kernel.pyx'), str(source/'su2_dmrg_engine.cpp')],
                    include_dirs=[np.get_include(), str(source)], language='c++',
                    libraries=libraries, library_dirs=library_dirs, runtime_library_dirs=library_dirs,
                    extra_compile_args=flags, extra_link_args=link)
    dist = Distribution({'ext_modules': cythonize([ext], build_dir=str(output/'cython'),
                          compiler_directives={'language_level': 3})})
    command = dist.get_command_obj('build_ext')
    command.build_lib = str(output)
    command.build_temp = str(output/'build')
    command.ensure_finalized()
    command.run()
    return output


if __name__ == '__main__':
    print(build())
