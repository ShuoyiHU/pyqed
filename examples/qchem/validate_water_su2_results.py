"""Audit the selected table against saved states and fresh independent FCI actions."""
import csv
import json
import pickle
from pathlib import Path
import numpy as np
from benchmark_water_su2 import ElectronicProblem,reference,diagnostics,apply_cas_vector,write_json

root=Path('examples/qchem/water_su2_comparison_final')
rows=list(csv.DictReader((root/'energies.csv').open()))
problems={};checks=[]
for row in rows:
    case=f'cas_{row["electrons"]}_{row["orbitals"]}'
    if case not in problems:
        z=np.load(root/case/'integrals.npz')
        p=ElectronicProblem(z['h1'],z['eri'],tuple(z['nelec']),float(z['ecore']))
        exact,vector=reference(p)
        residual=float(np.linalg.norm(apply_cas_vector(p,vector)-exact*vector))
        assert residual<2e-10,residual
        problems[case]=(p,exact,vector,residual)
    p,exact,vector,residual=problems[case]
    source=root.parent/row['source_record']
    path=source.with_suffix('.pkl') if source.stem.startswith('dmrg_only_') else source.parent/f'D{row["D_cap"]}_{row["method"]}.pkl'
    with path.open('rb') as f:state=pickle.load(f)
    d=diagnostics(state,p,exact,vector)
    mismatch=abs(d['total_energy']-float(row['energy_hartree']))
    assert mismatch<1e-10,mismatch
    assert d['max_multiplets']<=int(row['D_cap'])
    checks.append(dict(case=case,method=row['method'],cap=int(row['D_cap']),source=str(path.relative_to(root.parent)),
        energy_discrepancy=mismatch,spin_squared=d['spin_squared'],sector_leakage=d['sector_leakage'],
        energy_error=d['energy_error'],residual_norm=d['residual_norm'],fci_overlap=d['fci_overlap'],
        energy_stationary=row['stationary']=='True',final_change=float(row['final_change'])))
result=dict(selected_states_checked=len(checks),max_energy_discrepancy=max(x['energy_discrepancy'] for x in checks),
    max_abs_spin_squared=max(abs(x['spin_squared']) for x in checks),max_sector_leakage=max(x['sector_leakage'] for x in checks),
    all_selected_energy_stationary=all(x['energy_stationary'] for x in checks),
    reference_residuals={k:v[3] for k,v in problems.items()},checks=checks)
write_json(root/'FINAL_AUDIT.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
