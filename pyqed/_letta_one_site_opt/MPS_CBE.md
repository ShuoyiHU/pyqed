# SU(2) MPS controlled bond expansion

Reference: A. Gleis, J.-W. Li, and J. von Delft, “Controlled Bond Expansion for Density Matrix Renormalization Group Ground State Search at Single-Site Costs,” Physical Review Letters **130**, 246402 (2023), https://doi.org/10.1103/PhysRevLett.130.246402.

Use `letta_dmrg` with an open-boundary `ReducedLatticeLETTA` whose neighborhoods are all singleton tuples, and `LETTADMROptions(cbe_enabled=True, cbe_selector="mps-dd", cbe_expansion_dimension=1, gauge_mode="frontier")`. The expansion dimension counts SU(2) multiplets. The fixed cap is the initial maximum bond multiplet count, as in the reduced sweep driver.

## Fidelity and scope

This is an adaptation of the paper's exact/full discarded-discarded (DD) residual selection and expand–one-site-optimize–Schmidt-trim cycle. It does **not** implement the economical two-stage shrewd preselection. Pair residual tensors are materialized, although no two-site eigenproblem or dense pair metric is constructed. Therefore the paper's single-site computational scaling claim does not apply. No empirical convergence comparison in the paper is a guarantee for this implementation or for arbitrary initial states.

For each intermediate spin J, canonicalize the outer MPS environments and form the spin-weighted pair Hamiltonian action. Project it with `(I − Q_L Q_L†) R (I − Q_R Q_R†)`, where Q_L and Q_R span the kept left/right spaces. Sector SVDs select up to the requested number of additional multiplets, ranked globally by `(2J+1) s²`. The directional discarded factor is appended to the neighboring tensor and the active tensor receives zero padding: the represented state is unchanged. Only the expanded one-site Hamiltonian is diagonalized. Finally, a physical-norm Schmidt split restores the bond cap and may change sector allocation.

The reduced Clebsch–Gordan convention requires pair Schmidt matrices weighted by `sqrt(dim(right)/dim(J))`. No determinant-space projection or magnetic-component expansion is used by the solver. Tests may explicitly expand tiny reference states.

There is no ALS compression, alternating two-factor energy refinement, or strict LETTA one-site-baseline selection. The expanded local solve is variational; finite Schmidt truncation can increase energy. `cbe_trim_loss` records the Schmidt discarded weight. This is deliberately distinct from the tied LETTA `exact` selector and its acceptance/energy-refinement policy. Numerical rank decisions still depend on `metric_tolerance`.

Only native reduced SU(2) MPOs and untied OBC MPS are supported. Tied LETTA and periodic states are unsupported. The existing local eigensolver is reused rather than reproducing the paper's Lanczos implementation. Solver convergence reports local residual/energy stationarity, not global optimality.
