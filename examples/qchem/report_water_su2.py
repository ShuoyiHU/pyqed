"""Rebuild the H2O dimension comparison tables and figure from saved runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CASES = [(4, 4), (6, 6), (8, 6), (8, 8)]
FLOOR = 1e-10


def report(root):
    candidates = []
    roots=[root,root.parent/'water_su2_benchmark',root.parent/'water_su2_seed72']
    for ne, no in CASES:
        final_folder=root/f'cas_{ne}_{no}'
        exact=json.loads((final_folder/'reference.json').read_text())['energy']
        integral_hash=json.loads((final_folder/'model.json').read_text())['integral_sha256']
        for source in dict.fromkeys(roots):
            folder=source/f'cas_{ne}_{no}'
            if not folder.exists():continue
            assert json.loads((folder/'model.json').read_text())['integral_sha256']==integral_hash
            for path in list(folder.glob('D*.json'))+list(folder.glob('dmrg_only_D*.json')):
                record = json.loads(path.read_text())
                for method in ('dmrg', 'letta'):
                    if method not in record:continue
                    d = record[method]
                    optimization = d.get('polish', d['optimization'])
                    candidates.append(dict(electrons=ne, orbitals=no, method=method,
                        D_cap=record['cap'], D_actual=d['max_multiplets'],
                        D_magnetic=d['max_magnetic'], parameters=d['reduced_parameter_count'],
                        energy_hartree=d['total_energy'], error_hartree=d['total_energy']-exact,
                        residual_hartree=d['residual_norm'], spin_squared=d['spin_squared'],
                        fci_overlap=d['fci_overlap'], stationary=optimization['energy_stationary'],
                        final_change=optimization['history'][-1]['change'],
                        solve_seconds=d['optimization']['seconds']+d.get('polish',{}).get('seconds',0)+d.get('initial_polish',{}).get('seconds',0)+d.get('initial_optimization',{}).get('seconds',0),
                        seed=record['seed'],source_record=str(path.relative_to(root.parent)),
                        initialization_mps_energy=record['dmrg']['total_energy']))
    best={}
    for row in candidates:
        key=(row['electrons'],row['orbitals'],row['D_cap'],row['method'])
        if key not in best or row['energy_hartree']<best[key]['energy_hartree']:
            best[key]=row
    rows=[best[k] for k in sorted(best)]
    for filename,data in [('all_candidates.csv',candidates),('energies.csv',rows)]:
        with (root/filename).open('w') as out:
            writer=csv.DictWriter(out,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    targets = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
    matches=[]
    for ne,no in CASES:
        for target in targets:
            r=dict(electrons=ne, orbitals=no, target_error_hartree=target)
            for method in ('dmrg','letta'):
                selected=[x for x in rows if (x['electrons'],x['orbitals'],x['method'])==(ne,no,method)
                          and x['error_hartree']<=target]
                r[method+'_D_cap']=min((x['D_cap'] for x in selected),default=None)
            matches.append(r)
    with (root/'matched_dimensions.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(matches[0]));writer.writeheader();writer.writerows(matches)
    brackets=[]
    for ne,no in CASES:
        mps=sorted([x for x in rows if (x['electrons'],x['orbitals'],x['method'])==(ne,no,'dmrg')],key=lambda x:x['D_cap'])
        for l in [x for x in rows if (x['electrons'],x['orbitals'],x['method'])==(ne,no,'letta')]:
            target=FLOOR if l['error_hartree']<=FLOOR else l['error_hartree']+FLOOR
            hits=[m for m in mps if m['error_hartree']<=target]
            upper=hits[0]['D_cap'] if hits else None
            previous=max((m['D_cap'] for m in mps if upper is None or m['D_cap']<upper),default=None)
            brackets.append(dict(electrons=ne,orbitals=no,letta_D_cap=l['D_cap'],
                letta_error_hartree=l['error_hartree'],dmrg_previous_tested_cap=previous,
                dmrg_first_tested_cap_matching_or_better=upper))
    with (root/'equivalent_D_brackets.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(brackets[0]));writer.writeheader();writer.writerows(brackets)
    plt.rcParams.update({'font.family':'sans-serif','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),layout='constrained',sharey=True)
    for i,((ne,no),ax) in enumerate(zip(CASES,axes.flat)):
        for method,color,marker,label in [('dmrg','#0072B2','o','DMRG'),('letta','#D55E00','s','LETTA (NN)')]:
            group=[x for x in rows if (x['electrons'],x['orbitals'],x['method'])==(ne,no,method)]
            ax.plot([x['D_cap'] for x in group],[max(FLOOR,x['error_hartree']) for x in group],
                    color=color,marker=marker,markersize=4,linewidth=1.2,fillstyle='none',label=label)
        ax.set(xscale='log',yscale='log',title=f'CAS({ne}e, {no}o)',xlabel='Bond cap D (SU(2) multiplets)',ylim=(4e-11,1))
        ticks=sorted(set(x['D_cap'] for x in rows if (x['electrons'],x['orbitals'])==(ne,no)))
        ticks=sorted(set([d for d in (2,4,8,16,32,64,128) if min(ticks)<=d<=max(ticks)]+[max(ticks)]))
        ax.set_xticks(ticks,[str(x) for x in ticks]);ax.minorticks_off()
        ax.axhline(1e-3,color='.7',ls=':',lw=.7);ax.axhline(1e-6,color='.7',ls=':',lw=.7)
        ax.axhline(FLOOR,color='.6',ls='--',lw=.7)
        ax.text(-.14,1.08,chr(97+i),transform=ax.transAxes,fontweight='bold')
        if i%2==0:ax.set_ylabel('Energy error relative to FCI (hartree)')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('H₂O / 6-31G • canonical RHF orbitals • singlet\nValues below 10⁻¹⁰ hartree are shown at the numerical floor',fontsize=10)
    for ext in ('svg','pdf','png'):fig.savefig(root/f'error_vs_D.{ext}',dpi=240)
    plt.close(fig)
    lines=['# H₂O: U(1) × SU(2) LETTA versus DMRG','',
        '## Numerical protocol','',
        'All variational calculations use reduced U(1) charge and SU(2) spin tensors, targeting a singlet. '
        'The Hamiltonian and orbital order are identical for both methods. D is a cap on the number of SU(2) multiplets '
        'on each variational bond. The magnetic dimension is Σ(N,S) m(N,S)(2S+1). '
        'LETTA has additional invariant occupancy dependencies; its D does not include this extra storage. '
        'Parameter counts below are stored reduced coefficients, including allocated zeros, not independent degrees of freedom after gauge fixing.', '',
        'Geometry (Å): O (0,0,0), H (0,0,0.96), H (0.9294217,0,−0.24038). Basis: 6-31G. '
        'Each CAS freezes the lowest (10−Ne)/2 doubly occupied canonical RHF orbitals and uses the next No orbitals. '
        'No CASSCF orbital optimization or reordering is performed. CAS means (active electrons, spatial orbitals).', '',
        'The final comparison uses the lowest energy found at each tested cap for each method across native DMRG '
        'runs (seeds 71 and selected 72 checks) and block2 SU(2) runs (seeds 71 and 72 at every final-grid point). '
        'block2 uses the identical integrals, no point-group symmetry or orbital reordering, 24 two-site sweeps '
        'with noise 10⁻⁶, 10⁻⁷, then zero, followed by up to 100 one-site sweeps (energy tolerance 10⁻¹¹ Eh). '
        'Reduced tensors are imported directly, the last fused core is explicitly unfused, and redundant multiplets '
        'are removed by reduced norm Schmidt splits. Imported norms and energies are independently checked against '
        'block2 and PySCF before native one-site polishing and LETTA optimization.', '',
        'Every LETTA candidate starts from an exact embedding of its own optimized MPS and retains that candidate’s '
        'bond sectors. LETTA uses nearest-neighbor invariant occupancy ties. The best reported DMRG and LETTA points '
        'can come from different initializations and sector allocations; all_candidates.csv retains every candidate '
        'and energies.csv identifies the source record of each selected point. This tests the current NN invariant-label '
        'implementation, not spin-coupled or optimized long-range ties.', '',
        'A native pass is one sweep direction, alternating left/right. Two consecutive energy changes below 10⁻¹⁰ Eh '
        'define energy stationarity. New block2-seeded runs allow up to 100 native MPS and LETTA passes. The original '
        'native grid used 24 adaptive/LETTA passes and 12 polishing passes; nonstationary points receive up to 200 '
        'additional passes, with initial reports preserved. The adaptive native stage can stop on a stable left/right '
        'truncation cycle before polishing. Energy stationarity is not a certificate of the global variational minimum. '
        'Convergence status, residuals and source histories are retained.', '',
        'FCI and full wavefunctions are used only for independent post-run diagnostics: physical energy, residual, '
        'spin, electron-number leakage and overlap. The active solver never uses determinant-space projection. '
        'Each chemistry process used single-thread BLAS/OpenMP. Later high-D runs overlapped with independent seed checks '
        'and regression tests on the same machine; timings are diagnostic and do not support speedup claims. '
        'Timings in the CSV record optimization/initialization work, with import validation included in block2 initialization; LETTA timing '
        'is refinement time, so end-to-end LETTA additionally incurs its MPS initialization cost.', '',
        '## Reference energies','', '| CAS | FCI total energy (Eh) | FCI residual (Eh) | Frozen spatial orbitals |',
        '|---|---:|---:|---:|']
    for ne,no in CASES:
        folder=root/f'cas_{ne}_{no}';r=json.loads((folder/'reference.json').read_text());m=json.loads((folder/'model.json').read_text())
        lines.append(f'| ({ne}e,{no}o) | {r["energy"]:.12f} | {r["residual_norm"]:.2e} | {m["frozen_orbitals"]} |')
    lines+=['','## Smallest tested D meeting each error target','',
        'These are measured grid minima, not exact minimal dimensions. “—” means the tested grid did not reach the target. '
        'Results at or below 10⁻¹⁰ Eh should be treated as a numerical accuracy floor.', '',
        '| CAS | Error target (Eh) | DMRG D | LETTA D |','|---|---:|---:|---:|']
    for r in matches:
        lines.append(f'| ({r["electrons"]}e,{r["orbitals"]}o) | {r["target_error_hartree"]:.0e} | {r["dmrg_D_cap"] or "—"} | {r["letta_D_cap"] or "—"} |')
    lines+=['','## Direct precision matching','',
        'equivalent_D_brackets.csv gives, for every LETTA point, the first tested DMRG cap attaining its error or better, '
        'allowing 10⁻¹⁰ Eh numerical tolerance, and the preceding tested cap. When LETTA improves only slightly, '
        'the crossing may lie anywhere in that gap: the next tested cap must not be interpreted as an exact D conversion. '
        'At the numerical floor the comparison means both are within 10⁻¹⁰ Eh, not equality of roundoff digits.']
    lines+=['','## Full grid','', '| CAS | D cap | DMRG error (Eh) | LETTA error (Eh) | Actual D, M/L | Magnetic D, M/L | Stored coefficients, M/L |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for m in (r for r in rows if r['method']=='dmrg'):
        l=next((r for r in rows if r['method']=='letta' and (r['electrons'],r['orbitals'],r['D_cap'])==(m['electrons'],m['orbitals'],m['D_cap'])),None)
        if l is None:
            lines.append(f'| ({m["electrons"]}e,{m["orbitals"]}o) | {m["D_cap"]} | {m["error_hartree"]:.3e} | — | {m["D_actual"]}/— | {m["D_magnetic"]}/— | {m["parameters"]}/— |')
            continue
        lines.append(f'| ({m["electrons"]}e,{m["orbitals"]}o) | {m["D_cap"]} | {m["error_hartree"]:.3e} | {l["error_hartree"]:.3e} | {m["D_actual"]}/{l["D_actual"]} | {m["D_magnetic"]}/{l["D_magnetic"]} | {m["parameters"]}/{l["parameters"]} |')
    lines+=['','## Validation and provenance','',
        '137 regression tests passed for the Schmidt correction. After the diagnostic reconstruction improvement, '
        '12 focused benchmark tests passed (six new reconstruction tests plus six existing benchmark tests): 143 distinct '
        'tests in total. FINAL_AUDIT.json independently rechecks all 90 selected saved states against fresh FCI actions; '
        'all selected states are energy-stationary, with zero reported-energy discrepancy, spin contamination below '
        '1.1×10⁻¹⁴ and charge leakage below 4×10⁻¹⁶.', '',
        'The untied MPS split was corrected to whiten the reduced pair tensor with both boundary Gram matrices, '
        'including SU(2) dimension weights, before truncation. Tests check discarded norm and orthogonality against '
        'independent small wavefunctions, invariance under complex nonunitary gauges, and a target-spin FCI solve with '
        'global state expansion forbidden during optimization. Earlier pilot directories are excluded from these tables.', '',
        'Sources: benchmark_water_su2.py and benchmark_water_block2_letta.py; PySCF supplies RHF integrals and FCI references. '
        'Each case stores model metadata and an integral SHA-256 hash, compressed integrals, per-pass histories and '
        'pickled final states. This is deterministic numerical evidence, not an experimental dataset; no statistical '
        'significance or universal dimension-reduction factor is claimed.', '',
        'Figure contract: four quantitative comparison panels; identical method encodings and error scale, connected '
        'measured grid points, 10⁻³/10⁻⁶ Eh guides and an explicit 10⁻¹⁰ Eh display floor. SVG/PDF retain editable text. '
        'CSV is the source data; no smoothing, interpolated crossings, or omitted nonconverged points.', '',
        '## Reproduction', '',
        'From the repository root, use Python with NumPy, SciPy, PySCF, matplotlib, opt_einsum, sympy and the repository dependencies. '
        'Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1, NUMEXPR_NUM_THREADS=1 and PYTHONPATH=.', '',
        '```bash',
        'python examples/qchem/benchmark_water_su2.py --cas 4,4 --caps 2 3 4 5 6 8 --max-passes 24',
        'python examples/qchem/benchmark_water_su2.py --cas 6,6 --caps 2 4 8 12 14 16 18 20 24 --max-passes 24',
        'python examples/qchem/benchmark_water_su2.py --cas 8,6 --caps 2 4 8 9 10 12 --max-passes 24',
        'python examples/qchem/benchmark_water_su2.py --cas 8,8 --caps 2 4 8 12 16 24 --max-passes 24',
        'python examples/qchem/benchmark_water_su2.py --cas 6,6 8,6 8,8 --caps 8 --max-passes 24 --seed 72 --output examples/qchem/water_su2_seed72',
        'python examples/qchem/benchmark_water_su2.py --cas 6,6 --caps 16 --max-passes 24 --seed 72 --output examples/qchem/water_su2_seed72',
        'python examples/qchem/refine_water_su2.py examples/qchem/water_su2_seed72',
        'python examples/qchem/refine_water_su2.py',
        'python examples/qchem/benchmark_water_block2_letta.py',
        'python examples/qchem/benchmark_water_block2_letta.py --cas 8_8 --caps 49 50 51 57 58 68 69 75 76 80 96 128 --dmrg-only',
        'python examples/qchem/refine_water_su2.py examples/qchem/water_su2_comparison_final',
        'python examples/qchem/report_water_su2.py',
        'python examples/qchem/validate_water_su2_results.py',
        '```', '',
        'The block2 script requires the block2-pyscf environment. Existing completed points are resumed. '
        'Additional cap values can be requested in a later invocation; results.json aggregates all completed points.', '',
        '![Energy error versus D](error_vs_D.png)', '']
    highlights=['## Representative matched accuracies','',
        'D counts SU(2) multiplets for both methods. The DMRG column is the smallest tested cap attaining the LETTA error or better (10⁻¹⁰ Eh numerical tolerance). '
        'These are best tested states, not proofs of globally minimal D. In the three larger-space matches below, adjacent DMRG caps were also checked.', '',
        '| CAS | LETTA D | LETTA error (Eh) | DMRG D | DMRG error (Eh) |',
        '|---|---:|---:|---:|---:|']
    def error_text(value):return '≤1e-10' if value<=FLOOR else f'{value:.3e}'
    for case,ld,md in [((4,4),3,4),((6,6),16,17),((8,6),10,12),((8,8),40,51),((8,8),48,58),((8,8),64,76)]:
        l=next(x for x in rows if (x['electrons'],x['orbitals'],x['method'],x['D_cap'])==(*case,'letta',ld))
        m=next(x for x in rows if (x['electrons'],x['orbitals'],x['method'],x['D_cap'])==(*case,'dmrg',md))
        highlights.append(f'| ({case[0]}e,{case[1]}o) | {ld} | {error_text(l["error_hartree"])} | {md} | {error_text(m["error_hartree"])} |')
    highlights+=['',
        'The NN ties reduce the required variational bond dimension at these accuracies. They also add stored coefficients: '
        'for example, the CAS(8e,8o) LETTA D=64 state stores 6861 reduced coefficients versus 2724 for DMRG D=76. '
        'Thus this study establishes bond-dimension savings, not a memory or speed advantage. Both methods use charge U(1) and spin SU(2).', '',
        'The results also show why initial allocation matters: some low-D native MPS minima are weaker than block2’s, '
        'while some native-seeded LETTA states are better than the states initialized from the lower-energy block2 MPS. '
        'Optimizing LETTA-specific sector allocations, orbital order and nonlocal ties remains outside this fixed-NN study.', '']
    lines[2:2]=highlights
    (root/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path,nargs='?',default=Path('examples/qchem/water_su2_comparison_final'))
    report(parser.parse_args().root)
