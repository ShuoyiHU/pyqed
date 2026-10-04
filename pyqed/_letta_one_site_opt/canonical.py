"""Persistent, content-validated canonical norm environments.

Certificates live with a state, not a particular sweep cache. Tensor content
signatures also detect public in-place edits; identities alone are insufficient.
"""
from dataclasses import dataclass
from hashlib import blake2b
from copy import copy

import numpy as np


def tensor_signature(tensor):
    tensor = np.ascontiguousarray(tensor)
    return (tensor.shape, tensor.dtype.str,
            blake2b(tensor.view(np.uint8), digest_size=16).digest())


def side_signature(state, cut, direction):
    sites = range(cut) if direction == 'lr' else range(cut, state.nsites)
    return tuple((state.site_neighborhood(i), tensor_signature(state.tensors[i])) for i in sites)


@dataclass(frozen=True)
class CanonicalNormEnvironment:
    cut: int
    direction: str
    physical: tuple
    diagonal: np.ndarray
    environment: np.ndarray
    signature: tuple
    report: object
    error: float


def get_canonical(state, cut, direction, environment=None):
    saved = getattr(state, '_canonical_norm_environments', {})
    key = (direction, cut)
    record = saved.get(key)
    if record is None:
        return None
    if record.signature != side_signature(state, cut, direction):
        del saved[key]
        return None
    if environment is not None and environment is not record.environment:
        candidate = np.asarray(environment)
        if candidate.shape != record.environment.shape or not np.allclose(
                candidate, record.environment, rtol=0., atol=max(record.error * 2, 1e-13)):
            return None
    return record


def save_canonical(state, cut, direction, physical, diagonal, environment, report, error):
    diagonal, environment = np.array(diagonal, copy=True), np.array(environment, copy=True)
    diagonal.flags.writeable = environment.flags.writeable = False
    record = CanonicalNormEnvironment(cut, direction, tuple(physical), diagonal,
                                      environment, side_signature(state, cut, direction), report, error)
    if not hasattr(state, '_canonical_norm_environments'):
        state._canonical_norm_environments = {}
    state._canonical_norm_environments[(direction, cut)] = record
    return record


def copy_canonical_state(state):
    """Copy without the constructor's global rescaling that undoes a gauge."""
    result = copy(state)
    result.tensors = [a.copy() for a in state.tensors]
    result._canonical_norm_environments = {}
    for direction, cut in tuple(getattr(state, '_canonical_norm_environments', {})):
        record = get_canonical(state, cut, direction)
        if record is not None:
            result._canonical_norm_environments[(direction, cut)] = record
    return result
