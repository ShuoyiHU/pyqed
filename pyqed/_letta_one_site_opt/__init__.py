"""Shared finite-LETTA state, contractions, and one-site optimization."""

from .._letta_compression import MetricCompressionOptions

from .contractions import (
    BlockDiagonalMetric,
    DiagonalMetric,
    BoundaryMPS,
    IdentityEnvironmentCache,
    LETTAEnvironmentCache,
    network_expectation,
    network_operator_matrix,
    network_overlap,
)
from .operators import (
    LatticeMPO,
    exact_ground_state,
    identity_mpo,
    one_site_expectation,
    state_vector_one_site_expectation,
    state_vector_two_site_expectation,
    two_site_expectation,
)
from .solver import (
    LETTADMRGResult,
    LETTADMROptions,
    LETTASiteUpdate,
    LETTASweep,
    automatic_bond_schedule,
    letta_dmrg,
)
from .state import LatticeLETTA
from .transfer import SegmentTransfer, CompressedSegmentTransfer
from .gauge import (
    FrontierGaugeCut,
    FrontierGaugeReport,
    canonicalize_frontier,
    frontier_gauge_cuts,
    shift_frontier_gauge,
)
from .symmetry import AbelianSymmetry
from .reduced_symmetry import (
    ReducedBasisState,
    ReducedPhysicalBasis,
    ReducedSymmetry,
)
from .reduced_state import ReducedLatticeLETTA
from .reduced_gauge import (
    canonicalize_reduced_frontier, reduced_frontier_grams, shift_reduced_frontier_gauge,
)
from .reduced_frontier import (
    FrontierSiteEmbedding,
    ReducedFrontier,
    reduced_mps_state_vector,
)
from .reduced_operators import (
    ReducedMPOHamiltonian,
    physical_leg_from_reduced_basis,
    su2_heisenberg_mpo,
    su2_spin_operator,
)
from .reduced_solver import (
    ReducedLocalProblem,
    optimize_reduced_site,
    reduced_letta_dmrg,
    reduced_local_frame,
    reduced_local_problem,
)

from .abelian_backend import AbelianReducedMap, abelian_dmrg

__all__ = [
    "AbelianReducedMap",
    "abelian_dmrg",
    "canonicalize_reduced_frontier",
    "reduced_frontier_grams",
    "shift_reduced_frontier_gauge",
    "SegmentTransfer",
    "CompressedSegmentTransfer",
    "MetricCompressionOptions",
    "FrontierGaugeCut",
    "FrontierGaugeReport",
    "canonicalize_frontier",
    "frontier_gauge_cuts",
    "shift_frontier_gauge",
    "BlockDiagonalMetric",
    "DiagonalMetric",
    "AbelianSymmetry",
    "BoundaryMPS",
    "IdentityEnvironmentCache",
    "LETTADMRGResult",
    "LETTADMROptions",
    "LETTAEnvironmentCache",
    "LETTASiteUpdate",
    "LETTASweep",
    "LatticeLETTA",
    "LatticeMPO",
    "ReducedBasisState",
    "FrontierSiteEmbedding",
    "ReducedFrontier",
    "ReducedLatticeLETTA",
    "ReducedLocalProblem",
    "ReducedMPOHamiltonian",
    "ReducedPhysicalBasis",
    "ReducedSymmetry",
    "physical_leg_from_reduced_basis",
    "reduced_mps_state_vector",
    "optimize_reduced_site",
    "reduced_letta_dmrg",
    "reduced_local_frame",
    "reduced_local_problem",
    "su2_heisenberg_mpo",
    "su2_spin_operator",
    "automatic_bond_schedule",
    "exact_ground_state",
    "identity_mpo",
    "letta_dmrg",
    "network_expectation",
    "network_operator_matrix",
    "network_overlap",
    "one_site_expectation",
    "state_vector_one_site_expectation",
    "state_vector_two_site_expectation",
    "two_site_expectation",
]
