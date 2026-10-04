"""Additional open 1D nearest-neighbor Hamiltonians for CBE benchmarks.

Spin matrices are angular momentum operators (hbar=1), not Pauli matrices.
Spinless fermions use |0>, |1> and Jordan-Wigner ordering from left to right.
Alternation starts on bond/site zero. No particle-number or parity sector is
fixed by the benchmark runner.
"""
from __future__ import annotations

import numpy as np


CHAIN_MODEL_DEFAULTS = {
    "ssh": {"t1": 0.6, "t2": 1.4, "mu": 0.0},
    "rice_mele": {"t1": 0.6, "t2": 1.4, "stagger": 0.4, "mu": 0.0},
    "spinless_tv": {"t": 1.0, "V": 2.0, "mu": 0.0},
    "kitaev": {"t": 1.0, "pairing": 0.6, "mu": 0.5},
    "xy": {"J": 1.0, "gamma": 0.5, "h": 0.3},
    "xyz": {"Jx": 1.0, "Jy": 0.7, "Jz": 1.3, "h": 0.2},
    "dimerized_heisenberg": {"J": 1.0, "dimer": 0.4, "delta": 1.0, "h": 0.0},
    "spin1_heisenberg": {"J": 1.0, "delta": 1.0, "single_ion": 0.0, "h": 0.0},
    "blume_capel": {"J": 1.0, "h": 0.7, "single_ion": 0.5},
}

CHAIN_HAMILTONIAN_LABELS = {
    "ssh": ("Spinless SSH", (
        r"$H=-\sum_i t_i(c_i^\dagger c_{i+1}+\mathrm{h.c.})-\mu\sum_i n_i$",
        r"$t_i=t_1\ (i\ \mathrm{even}),\ t_2\ (i\ \mathrm{odd});\quad i=0,\ldots,L-2$",
    )),
    "rice_mele": ("Spinless Rice-Mele", (
        r"$H=-\sum_i t_i(c_i^\dagger c_{i+1}+\mathrm{h.c.})"
        r"+\sum_i[(-1)^i m-\mu]n_i$",
        r"$t_i=t_1\ (i\ \mathrm{even}),\ t_2\ (i\ \mathrm{odd});\quad m=\mathrm{stagger}$",
    )),
    "spinless_tv": ("Spinless t-V", (
        r"$H=-t\sum_i(c_i^\dagger c_{i+1}+\mathrm{h.c.})"
        r"+V\sum_i q_iq_{i+1}-\mu\sum_i q_i$",
        r"$q_i=n_i-\frac{1}{2}$",
    )),
    "kitaev": ("Kitaev chain", (
        r"$H=-t\sum_i(c_i^\dagger c_{i+1}+\mathrm{h.c.})"
        r"-\mu\sum_i(n_i-\frac{1}{2})$",
        r"$\quad+\Delta_p\sum_i(c_ic_{i+1}+\mathrm{h.c.});\quad\Delta_p=\mathrm{pairing}$",
    )),
    "xy": ("Spin-1/2 XY", (
        r"$H=J\sum_i[(1+\gamma)S_i^xS_{i+1}^x"
        r"+(1-\gamma)S_i^yS_{i+1}^y]-h\sum_i S_i^z$",
    )),
    "xyz": ("Spin-1/2 XYZ", (
        r"$H=\sum_i[J_xS_i^xS_{i+1}^x+J_yS_i^yS_{i+1}^y"
        r"+J_zS_i^zS_{i+1}^z]-h\sum_i S_i^z$",
    )),
    "dimerized_heisenberg": ("Dimerized spin-1/2 XXZ", (
        r"$H=\sum_i J_i[S_i^xS_{i+1}^x+S_i^yS_{i+1}^y"
        r"+\Delta S_i^zS_{i+1}^z]-h\sum_i S_i^z$",
        r"$J_i=J[1+(-1)^i\delta_d];\quad\delta_d=\mathrm{dimer}$",
    )),
    "spin1_heisenberg": ("Spin-1 XXZ with single-ion anisotropy", (
        r"$H=J\sum_i[S_i^xS_{i+1}^x+S_i^yS_{i+1}^y+\Delta S_i^zS_{i+1}^z]$",
        r"$\quad+A\sum_i(S_i^z)^2-h\sum_i S_i^z;\quad A=\mathrm{single\_ion}$",
    )),
    "blume_capel": ("Spin-1 quantum Blume-Capel", (
        r"$H=-J\sum_i S_i^zS_{i+1}^z-h\sum_i S_i^x+A\sum_i(S_i^z)^2$",
        r"$A=\mathrm{single\_ion}$",
    )),
}


def build_chain_terms(name, nsites, bonds, parameters):
    """Return local dimension and exact one-site/two-site product terms."""
    from .condensed_models import ProductTerm

    p = parameters
    terms = []
    if name in {"ssh", "rice_mele", "spinless_tv", "kitaev"}:
        c = np.array([[0., 1.], [0., 0.]])
        parity = np.diag([1., -1.])
        number = c.T @ c
        charge = number - 0.5 * np.eye(2)
        for left, right in bonds:
            hopping = p["t1" if left % 2 == 0 else "t2"] if name in {
                "ssh", "rice_mele"} else p["t"]
            terms.extend([
                ProductTerm(-hopping, {left: c.T @ parity, right: c}),
                ProductTerm(-hopping, {left: parity @ c, right: c.T}),
            ])
            if name == "spinless_tv":
                terms.append(ProductTerm(p["V"], {left: charge, right: charge}))
            elif name == "kitaev":
                # c_i c_{i+1} carries c*parity on its left endpoint.
                terms.extend([
                    ProductTerm(p["pairing"], {left: c @ parity, right: c}),
                    ProductTerm(p["pairing"], {left: parity @ c.T, right: c.T}),
                ])
        for site in range(nsites):
            onsite = -p["mu"] * (charge if name in {"spinless_tv", "kitaev"} else number)
            if name == "rice_mele":
                onsite = onsite + (-1) ** site * p["stagger"] * number
            terms.append(ProductTerm(1., {site: onsite}))
        return 2, tuple(terms)

    spin = 1.0 if name in {"spin1_heisenberg", "blume_capel"} else 0.5
    physical_dim = int(2 * spin + 1)
    m = np.arange(spin, -spin - 1, -1)
    sz = np.diag(m)
    sp = np.diag(np.sqrt(spin * (spin + 1) - m[1:] * (m[1:] + 1)), 1)
    sm = sp.T
    for left, right in bonds:
        if name == "blume_capel":
            terms.append(ProductTerm(-p["J"], {left: sz, right: sz}))
            continue
        if name == "xy":
            jx, jy, jz = p["J"] * (1 + p["gamma"]), p["J"] * (1 - p["gamma"]), 0.
        elif name == "xyz":
            jx, jy, jz = p["Jx"], p["Jy"], p["Jz"]
        else:
            coupling = p["J"] * (1 + (-1) ** left * p.get("dimer", 0.))
            jx, jy, jz = coupling, coupling, coupling * p["delta"]
        # Ladder operators keep these real Hamiltonians real throughout.
        terms.extend([
            ProductTerm((jx + jy) / 4, {left: sp, right: sm}),
            ProductTerm((jx + jy) / 4, {left: sm, right: sp}),
            ProductTerm((jx - jy) / 4, {left: sp, right: sp}),
            ProductTerm((jx - jy) / 4, {left: sm, right: sm}),
            ProductTerm(jz, {left: sz, right: sz}),
        ])
    for site in range(nsites):
        field = (sp + sm) / 2 if name == "blume_capel" else sz
        onsite = -p["h"] * field + p.get("single_ion", 0.) * (sz @ sz)
        terms.append(ProductTerm(1., {site: onsite}))
    return physical_dim, tuple(terms)
