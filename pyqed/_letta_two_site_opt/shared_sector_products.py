"""Exact, action-local sharing of repeated first matrix products.

RAM and optional scratch-file budgets refer to intermediates, not total RSS.
No directions are truncated. Products exceeding budgets are recomputed.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack
import os
import tempfile
import time
import numpy as np


class SharedSectorProducts:
    def __init__(self, operator, ram_bytes, disk_bytes=0):
        self.operator = operator
        self.keys = []
        self.rows = {}
        groups = defaultdict(list)
        sizes, costs = {}, {}
        for index, row in enumerate(operator.routes):
            lo, ro, xo, yo, l, r, a, b, p, o, npi, npo = map(int, row)
            left = a*l*r + a*r*b <= l*r*b + a*l*b
            key = (left, lo if left else ro, xo, l, r, a if left else b, p, npi)
            self.keys.append(key)
            self.rows[key] = tuple(map(int, row))
            sizes[key] = a*r if left else l*b
            costs[key] = a*l*r if left else l*r*b
            groups[yo, o].append(index)
        counts = Counter(self.keys)
        self.ram, self.disk = {}, {}
        used_ram = used_disk = 0
        for key in sorted(counts, key=lambda k: (counts[k]-1)*costs[k]/max(1,sizes[k]), reverse=True):
            if counts[key] < 2:
                continue
            size = sizes[key]
            if (used_ram+size)*16 <= ram_bytes:
                self.ram[key] = (used_ram, size)
                used_ram += size
            elif (used_disk+size)*16 <= disk_bytes:
                self.disk[key] = (used_disk, size)
                used_disk += size
        self.ram_elements, self.disk_elements = used_ram, used_disk
        # One owner per output slice; preserve summation order. Splitting only
        # cached products avoids repeating uncached first products across tiles.
        self.tasks = []
        for indices in groups.values():
            a = int(operator.routes[indices[0], 6])
            tile = 128 if all(self.keys[i] in self.ram or self.keys[i] in self.disk for i in indices) else a
            for start in range(0, a, max(1, tile)):
                self.tasks.append((indices, start, min(a, start+tile)))

    def apply(self, vector, workers, pool):
        op = self.operator
        dtype = np.result_type(vector, op.dtype)
        result = np.zeros(op.layout.size, dtype=dtype)
        started = time.perf_counter()
        with ExitStack() as stack:
            ram = np.empty(self.ram_elements, dtype=dtype)
            disk = None
            if self.disk_elements:
                handle = stack.enter_context(tempfile.TemporaryFile(dir=os.environ.get('PYQED_PRODUCT_SCRATCH')))
                handle.truncate(self.disk_elements*np.dtype(dtype).itemsize)
                disk = np.memmap(handle, mode='r+', dtype=dtype, shape=(self.disk_elements,))
                stack.callback(disk._mmap.close)
            def product(key):
                lo, ro, xo, yo, l, r, a, b, p, o, npi, npo = self.rows[key]
                x = vector[xo:xo+l*npi*r].reshape(l,npi,r)[:,p,:]
                if key[0]:
                    return op.left[lo:lo+a*l].reshape(a,l) @ x
                return x @ op.right[ro:ro+b*r].reshape(b,r).T
            def fill(key):
                target, plan = (ram,self.ram) if key in self.ram else (disk,self.disk)
                offset, size = plan[key]
                target[offset:offset+size] = product(key).ravel()
            def execute(fn, items):
                if workers == 1:
                    for item in items: fn(item)
                else:
                    for _ in pool.map(fn, items): pass
            execute(fill, list(self.ram)+list(self.disk))
            first_seconds = time.perf_counter()-started
            def second(task):
                indices, start, stop = task
                for index in indices:
                    lo, ro, xo, yo, l, r, a, b, p, o, npi, npo = map(int,op.routes[index])
                    key = self.keys[index]
                    if key in self.ram or key in self.disk:
                        target, plan = (ram,self.ram) if key in self.ram else (disk,self.disk)
                        offset,size = plan[key]
                        mid = target[offset:offset+size].reshape((a,r) if key[0] else (l,b))
                    else:
                        mid = product(key)
                    if key[0]:
                        value = mid[start:stop] @ op.right[ro:ro+b*r].reshape(b,r).T
                    else:
                        value = op.left[lo:lo+a*l].reshape(a,l)[start:stop] @ mid
                    result[yo:yo+a*npo*b].reshape(a,npo,b)[start:stop,o,:] += op.coefficients[index]*value
            execute(second, self.tasks)
            second_seconds = time.perf_counter()-started-first_seconds
        result *= op.scale
        return result, dict(shared_first_seconds=first_seconds, shared_second_seconds=second_seconds,
            shared_ram_bytes=self.ram_elements*np.dtype(dtype).itemsize,
            shared_disk_bytes=self.disk_elements*np.dtype(dtype).itemsize,
            shared_products=len(self.ram)+len(self.disk))
