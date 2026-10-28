"""Prepared two-sided reduced-sector maps executed in C through BLAS.

The Cython source generates C, using SciPy's BLAS interface so the numerical
library matches the installed SciPy. This is an exact contraction rewrite:
no new rank threshold, sparsification, or solver approximation is introduced.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
import tempfile
import numpy as np


@lru_cache(maxsize=1)
def _engine():
    import Cython
    import scipy
    source = Path(__file__).with_name('pair_metric_kernels.pyx')
    signature = (sys.version, sysconfig.get_platform(), Cython.__version__, scipy.__version__,
                 os.environ.get('CC'), os.environ.get('CFLAGS'), os.environ.get('LDFLAGS'))
    digest = hashlib.sha256(source.read_bytes()+repr(signature).encode()).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / ('pyqed-pair-metric-'+str(os.getuid())) / digest
    cache.mkdir(parents=True, exist_ok=True)
    name = '_pair_metric_kernels'
    library = cache / (name + sysconfig.get_config_var('EXT_SUFFIX'))
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=cache) as build:
            build = Path(build)
            copied = build / source.name
            copied.write_bytes(source.read_bytes())
            script = """import sys
from setuptools import Extension, Distribution
from Cython.Build import cythonize
extension=Extension('_pair_metric_kernels',[sys.argv[1]],extra_compile_args=['-O3'])
dist=Distribution({'ext_modules':cythonize([extension],quiet=True,compiler_directives={'language_level':3})})
command=dist.get_command_obj('build_ext')
command.build_lib=sys.argv[2]
command.build_temp=sys.argv[2]+'/objects'
command.ensure_finalized()
command.run()
"""
            run = subprocess.run([sys.executable,'-c',script,str(copied),str(build)],
                                 capture_output=True,text=True)
            if run.returncode:
                raise RuntimeError('Pair metric compilation failed:\n'+run.stdout+run.stderr)
            os.replace(build/library.name, library)
    spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=4096)
def _metric_left_first(l, p, q, r, a, b):
    """Cache only NumPy's shape-dependent greedy path, never tensor values."""
    scalar=np.zeros(())
    left=np.broadcast_to(scalar,(a,l))
    right=np.broadcast_to(scalar,(b,r))
    vector=np.broadcast_to(scalar,(l,p,q,r))
    path=np.einsum_path('al,br,lpqr->apqb',left,right,vector,optimize=True)[0]
    if path[1]==(0,1):
        m=p*q
        return a*l*m*r+a*m*r*b <= l*m*r*b+a*l*m*b
    return path[1]==(0,2)


class PairMetricMap:
    """Prepare Y[a,p,q,b] = scale L[a,l] X[l,p,q,r] R[b,r].

    Preparation validates layouts once; repeated actions use a packed route
    table and two BLAS products per nonzero sector, without Python block
    dictionaries or einsum path planning. Identical fixed matrices share one
    packed copy across routes. Workspaces belong to each call.
    """
    def __init__(self, source_layout, target_layout, factors):
        if set(source_layout.keys) != set(target_layout.keys) or set(factors) != set(source_layout.keys):
            raise ValueError('metric map sectors must agree')
        self.input_size, self.output_size = source_layout.size, target_layout.size
        routes, scales, left, right = [], [], [], []
        workspace = 0
        dtype = np.result_type(float, *[np.result_type(l,r) for l,r,s in factors.values()])
        self.dtype = np.dtype(np.complex128 if np.issubdtype(dtype,np.complexfloating) else np.float64)
        parts=[left,right];sizes=[0,0];by_id=[{},{}];by_value=[{},{}]
        self.unshared_factor_bytes=0
        def matrix_offset(side,array):
            self.unshared_factor_bytes+=array.size*self.dtype.itemsize
            if id(array) in by_id[side]:return by_id[side][id(array)]
            matrix=np.ascontiguousarray(array,dtype=self.dtype)
            signature=(matrix.shape,hashlib.blake2b(matrix.view(np.uint8)).digest())
            flat=matrix.ravel()
            bucket=by_value[side].setdefault(signature,[])
            for offset,previous in bucket:
                if np.array_equal(flat,previous):break
            else:
                offset=sizes[side];sizes[side]+=flat.size
                parts[side].append(flat);bucket.append((offset,flat))
            by_id[side][id(array)]=offset
            return offset
        for key in source_layout.keys:
            shape, out = source_layout.shapes[key], target_layout.shapes[key]
            lmat, rmat, scale = factors[key]
            if len(shape)!=4 or len(out)!=4 or shape[1:3]!=out[1:3]:
                raise ValueError('metric maps require matching physical dimensions')
            l,r = shape[0],shape[-1]
            a,b = out[0],out[-1]
            m = shape[1]*shape[2]
            if np.shape(lmat)!=(a,l) or np.shape(rmat)!=(b,r):
                raise ValueError('metric factor shape mismatch')
            if max(l,r,a,b,m*r,a*m,l*m,m*b) > np.iinfo(np.int32).max:
                raise OverflowError('metric block exceeds BLAS integer range')
            # Preserve the established greedy contraction order. A cheaper FLOP
            # order can amplify cancellation in ill-conditioned raw factors.
            left_first=_metric_left_first(l,shape[1],shape[2],r,a,b)
            lo,ro=matrix_offset(0,lmat),matrix_offset(1,rmat)
            routes.append((lo,ro,source_layout.offsets[key][0],target_layout.offsets[key][0],l,r,a,b,m,int(left_first)))
            scales.append(float(scale))
            count = a*m*r if left_first else l*m*b
            workspace=max(workspace,count)
        self.routes=np.asarray(routes,dtype=np.int64).reshape(-1,10)
        self.scales=np.asarray(scales,dtype=np.float64)
        self.left=np.concatenate(left) if left else np.empty(0,dtype=self.dtype)
        self.right=np.concatenate(right) if right else np.empty(0,dtype=self.dtype)
        self.factor_bytes=self.left.nbytes+self.right.nbytes
        self.workspace_size=workspace
        self._complex_factors=None

    def __call__(self, vector):
        vector=np.asarray(vector)
        if vector.size!=self.input_size:
            raise ValueError('metric vector has incompatible size')
        dtype=np.complex128 if np.iscomplexobj(vector) or self.dtype.kind=='c' else np.float64
        vector=np.ascontiguousarray(vector,dtype=dtype).reshape(-1)
        if not vector.flags.writeable:
            vector=vector.copy()
        left,right=self.left,self.right
        if dtype==np.complex128 and self.dtype.kind!='c':
            if self._complex_factors is None:
                self._complex_factors=(left.astype(dtype),right.astype(dtype))
            left,right=self._complex_factors
        result=np.zeros(self.output_size,dtype=dtype)
        work=np.empty(self.workspace_size,dtype=dtype)
        _engine().contract(self.routes,self.scales,left,right,vector,result,work)
        return result
