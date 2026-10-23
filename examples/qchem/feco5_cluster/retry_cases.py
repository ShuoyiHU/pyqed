"""Rerun selected cases with unchanged integrals and an explicit tolerance migration."""
import argparse,hashlib,json,os,shlex,shutil,subprocess
from pathlib import Path
from datetime import datetime


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def stage(root,old,destination,spaces):
    config=json.loads((root/'config.json').read_text())
    if config['energy_tolerance']!=1e-6:raise ValueError('Expected requested 1e-6 Ha energy tolerance')
    previous=dict(config,energy_tolerance=1e-11)
    # The previous configuration differs by this one literal only.
    old_bytes=(root/'config.json').read_bytes().replace(b'"energy_tolerance": 1e-6',b'"energy_tolerance": 1e-11')
    assert json.loads(old_bytes)==previous
    def signature(config_bytes):
        h=hashlib.sha256(config_bytes)
        for name in ['geometry.xyz','campaign.py','source_manifest.json']:h.update((root/name).read_bytes())
        return h.hexdigest()
    before,after=signature(old_bytes),signature((root/'config.json').read_bytes())
    cases=[]
    for b,basis in enumerate(config['bases']):
        for space in spaces:
            i=config['spaces'].index([space,space]);src=old/basis/f'cas_{space}_{space}'
            model=json.loads((src/'model.json').read_text())
            if model['signature']!=before:raise ValueError(f'Unexpected previous source/configuration: {src}')
            if digest(src/'integrals.npz')!=model['integral_sha256']:raise ValueError(f'Integral hash mismatch: {src}')
            cases.append((b,i,src,model))
    destination.mkdir()  # Refuse accidental duplicate run/output replacement.
    (destination/'logs').mkdir()
    for b,i,src,model in cases:
        dst=destination/src.parent.name/src.name;dst.mkdir(parents=True)
        shutil.copyfile(src/'integrals.npz',dst/'integrals.npz')
        assert digest(dst/'integrals.npz')==model['integral_sha256']
        model['signature']=after
        model['retry_provenance']={'source':str(src),'previous_signature':before,'change':'energy_tolerance 1e-11 -> 1e-6 Ha; fresh MPS initialization'}
        (dst/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    (destination/'retry.json').write_text(json.dumps(dict(source=str(old),spaces=spaces,old_signature=before,signature=after,energy_tolerance=1e-6),indent=2)+'\n')
    return config,cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--spaces',default='12')
    p.add_argument('--memory',default='12:48G,14:128G,16:320G,18:768G')
    p.add_argument('--submit',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;old=a.source.resolve()
    spaces=[int(s) for s in a.spaces.split(',')];memory=dict(x.split(':') for x in a.memory.split(','))
    for space in spaces:
        if str(space) not in memory:raise ValueError(f'Specify --memory for CAS({space},{space})')
    out=root/('results_retry_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    config,cases=stage(root,old,out,spaces)
    if a.submit:
        from campaign import verify_source
        from pyqed.mps.nonabelian.sweep_engine import load_engine
        verify_source();load_engine()
        # Remove only this source campaign's already-blocked LETTA jobs.
        selected={str(config['spaces'].index([space,space])) for space in spaces}
        records=[row for record in root.glob('results*/submitted.tsv')
            for row in record.read_text().splitlines()[1:] if row.strip()]
        blocked_candidates={row.split('\t')[3] for row in records
            if row.split('\t')[0]=='letta' and row.split('\t')[2] in selected}
        queue=subprocess.check_output(['squeue','--noheader','--user',os.environ['USER'],
            '--format=%i|%T|%R'],text=True)
        cancel=[fields[0] for row in queue.splitlines()
            if len(fields:=row.strip().split('|'))==3 and fields[0] in blocked_candidates
            and fields[1]=='PENDING' and fields[2].strip('()')=='DependencyNeverSatisfied']
        if cancel:subprocess.run(['scancel',*cancel],check=True)

    env=dict(os.environ,FECO_ROOT=str(root),OUTPUT_ROOT=str(out))
    manifest=out/'submitted.tsv';manifest.write_text('stage\tbasis\tspace\tjob_id\tdependency\n')
    common=['sbatch','--parsable','--partition='+os.environ.get('PARTITION','gubing,test,testf,test-intel'),'--qos='+os.environ.get('QOS','huge'),'--time=0','--nodes=1','--ntasks=1','--cpus-per-task=1','--export=ALL','--chdir='+str(root)]
    for b,i,src,model in cases:
        parent=None
        for method in ['mps','letta']:
            mem=memory[str(config['spaces'][i][0])]
            if method=='letta' and config['spaces'][i][0]==12:mem='64G'
            dep='-' if parent is None else 'afterok:'+parent+'_0'
            cmd=common+['--mem='+mem,'--job-name=qc_'+method+'_'+str(b)+'_'+str(i),'--output='+str(out/f'logs/{method}_{b}_{i}_%A_%a.out'),'--error='+str(out/f'logs/{method}_{b}_{i}_%A_%a.err')]
            cmd+=['--array=0-3'] if method=='mps' else ['--dependency='+dep]
            cmd += [str(root/'retry_worker.sh'),method,str(b),str(i)]
            if a.submit:
                reply=subprocess.check_output(cmd,env=env,text=True).strip();job=reply.split(';')[0]
                if not job.isdigit():raise RuntimeError('Invalid sbatch response: '+reply)
                with manifest.open('a') as f:f.write(f'{method}\t{b}\t{i}\t{job}\t{dep}\n')
                print(f'{method} {src.parent.name} {src.name}: job {job}, memory {mem}, dependency {dep}',flush=True)
            else:print(shlex.join(cmd));job=f'MPS_{b}_{i}'
            if method=='mps':parent=job
    print('Output:',out)
    print('Old results preserved; fresh MPS runs use saved, hash-verified integrals.')

if __name__=='__main__':main()
