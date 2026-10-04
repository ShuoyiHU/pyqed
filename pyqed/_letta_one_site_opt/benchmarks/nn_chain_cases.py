"""Editable 1D nearest-neighbor CBE comparison presets.

Each entry contains (model, length, bond dimension, parameter overrides).
All use open boundaries and the native nearest-tied LETTA shape (1, length).
"""

NN_CHAIN_PRESETS = {
    "ising_critical_8_4": ("ising", 8, 4, {"h": 1.0}),
    "ising_ordered_8_4": ("ising", 8, 4, {"h": 0.5}),
    "ising_field_8_4": ("ising", 8, 4, {"h": 1.5}),
    "ising_critical_12_4": ("ising", 12, 4, {"h": 1.0}),
    "heisenberg_chain_8_4": ("heisenberg", 8, 4, {}),
    "heisenberg_chain_12_4": ("heisenberg", 12, 4, {}),
    "xx_chain_8_4": ("heisenberg", 8, 4, {"delta": 0.0}),
    "xxz_easy_plane_8_4": ("heisenberg", 8, 4, {"delta": 0.5}),
    "xxz_easy_axis_8_4": ("heisenberg", 8, 4, {"delta": 2.0}),
    "xy_anisotropic_8_4": ("xy", 8, 4, {}),
    "xyz_chain_8_4": ("xyz", 8, 4, {}),
    "dimerized_heisenberg_8_4": ("dimerized_heisenberg", 8, 4, {}),
    "dimerized_heisenberg_weak_8_4": ("dimerized_heisenberg", 8, 4, {"dimer": 0.1}),
    "ssh_weak_end_8_4": ("ssh", 8, 4, {"t1": 0.6, "t2": 1.4}),
    "ssh_strong_end_8_4": ("ssh", 8, 4, {"t1": 1.4, "t2": 0.6}),
    "ssh_uniform_8_4": ("ssh", 8, 4, {"t1": 1.0, "t2": 1.0}),
    "ssh_weak_end_12_4": ("ssh", 12, 4, {"t1": 0.6, "t2": 1.4}),
    "rice_mele_8_4": ("rice_mele", 8, 4, {}),
    "spinless_tv_weak_8_4": ("spinless_tv", 8, 4, {"V": 1.0}),
    "spinless_tv_critical_8_4": ("spinless_tv", 8, 4, {"V": 2.0}),
    "spinless_tv_strong_8_4": ("spinless_tv", 8, 4, {"V": 3.0}),
    "kitaev_small_mu_8_4": ("kitaev", 8, 4, {"mu": 0.5}),
    "kitaev_transition_8_4": ("kitaev", 8, 4, {"mu": 2.0}),
    "kitaev_large_mu_8_4": ("kitaev", 8, 4, {"mu": 3.0}),
    "spin1_heisenberg_6_3": ("spin1_heisenberg", 6, 3, {}),
    "spin1_single_ion_6_3": ("spin1_heisenberg", 6, 3, {"single_ion": 1.0}),
    "spin1_easy_axis_6_3": ("spin1_heisenberg", 6, 3, {"delta": 1.5}),
    "blume_capel_6_3": ("blume_capel", 6, 3, {}),
    "blume_capel_single_ion_6_3": ("blume_capel", 6, 3, {"single_ion": 1.0}),
}

NN_CHAIN_CASES = {
    name: (model, "1d", length, bond_dim)
    for name, (model, length, bond_dim, parameters) in NN_CHAIN_PRESETS.items()
}
NN_CHAIN_PARAMETERS = {
    name: dict(parameters)
    for name, (model, length, bond_dim, parameters) in NN_CHAIN_PRESETS.items()
}
