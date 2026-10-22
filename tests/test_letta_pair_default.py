from test_letta_reduced_cbe import incomplete
from pyqed._letta_two_site_opt.reduced_solver import reduced_pair_problem, _expand_reduced_pair_space

def test_pair_uses_growth_safe_support_by_default():
    s,h=incomplete()
    expanded=_expand_reduced_pair_space(s,0,4)
    problem=reduced_pair_problem(expanded,h,0,matrix_free=True,dense_solver_threshold=0)
    assert problem.frontier.prune, 'Pair paths are still Cartesian'
