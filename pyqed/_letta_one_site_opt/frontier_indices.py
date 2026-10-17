"""Exact C integer routing for reduced LETTA frontier embeddings.

Builds a portable C11 shared library in a content-addressed temporary cache.
No numerical cutoff or solver behavior is changed. A compiler is required;
build errors are explicit instead of silently using a different engine.
"""
from functools import lru_cache
from pathlib import Path
import ctypes
import hashlib
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import numpy as np


@lru_cache(maxsize=1)
def _kernel():
    source = Path(__file__).with_suffix('.c')
    compiler = shlex.split(os.environ.get('CC', 'cc'))
    digest = hashlib.sha256(source.read_bytes() + repr((compiler, sys.platform, sysconfig.get_platform())).encode()).hexdigest()[:20]
    directory = Path(tempfile.gettempdir()) / ('pyqed-frontier-' + str(os.getuid()))
    directory.mkdir(mode=0o700, exist_ok=True)
    library = directory / (digest + '.so')
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=directory) as build:
            output = Path(build) / 'indices.so'
            command = compiler + ['-O3', '-std=c11', '-fPIC',
                '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                str(source), '-o', str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            os.replace(output, library)
    handle = ctypes.CDLL(str(library))
    function = handle.frontier_indices
    ptr = ctypes.POINTER(ctypes.c_int64)
    integer = ctypes.c_int64
    function.argtypes = [ptr, integer, ptr, integer, ptr, integer, ptr, integer,
                         ptr, integer, integer, integer, ptr, ptr]
    function.restype = integer
    return function


def block_indices(shape, left_variables, right_variables, neighborhood,
                  physical, site, nsites, radix):
    shape = tuple(int(x) for x in shape)
    groups = (left_variables, right_variables, neighborhood)
    if len(shape) != len(neighborhood)+2 or any(x < 0 for x in shape):
        raise ValueError('invalid frontier block shape')
    if not 0 <= site < nsites or radix < 1:
        raise ValueError('invalid frontier site or radix')
    if any(not 0 <= x < nsites for group in groups for x in group):
        raise ValueError('invalid frontier variable')
    if len(physical) != shape[1] or any(not 0 <= x < radix for x in physical):
        raise ValueError('invalid physical labels')
    if any(x != radix for x in shape[2:-1]):
        raise ValueError('invalid tie dimension')
    limit = np.iinfo(np.int64).max
    import math
    if any(x>limit for x in (*shape,radix**len(left_variables),radix**len(right_variables))) or math.prod(shape) > limit or math.prod((radix**len(left_variables),
            radix**len(right_variables), shape[0], shape[1], shape[-1])) > limit:
        raise OverflowError('frontier indices exceed int64')
    arrays = [np.ascontiguousarray(x, dtype=np.int64) for x in
              (shape, left_variables, right_variables, neighborhood, physical)]
    ptr = ctypes.POINTER(ctypes.c_int64)
    pointers = [x.ctypes.data_as(ptr) for x in arrays]
    args = (pointers[0], len(shape), pointers[1], len(left_variables),
            pointers[2], len(right_variables), pointers[3], len(neighborhood),
            pointers[4], site, nsites, radix)
    function = _kernel()
    count = function(*args, None, None)
    if count < 0:
        raise MemoryError('frontier routing workspace allocation failed')
    source = np.empty(count, dtype=np.int64)
    target = np.empty(count, dtype=np.int64)
    actual = function(*args, source.ctypes.data_as(ptr), target.ctypes.data_as(ptr))
    if actual != count:
        raise RuntimeError('frontier routing count mismatch')
    return source, target
