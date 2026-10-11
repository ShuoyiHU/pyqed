from collections import Counter
import numpy as np
def initial_mps(n,electrons,spin,cap,convention,seed,allocation_only=False):
    from pyqed.mps.nonabelian.states import (build_random_reduced_spatial_mps,
                                            spatial_target_sector,_fuse_spatial_sectors)
    base=build_random_reduced_spatial_mps(n,target_sector=spatial_target_sector(electrons,spin),bond_multiplicity=1,seed=seed)
    physical=tuple(base[0].qns[1]);sectors=[tuple(base[0].qns[0])]+[tuple(s.qns[2]) for s in base]
    if max(map(len,sectors))>cap:
        # A union of complete fusion paths cannot create disconnected sectors.
        completions=[None]*(n+1);completions[n]=Counter({sectors[-1][0]:1})
        for i in reversed(range(n)):
            completions[i]=Counter({q:sum(completions[i+1][out] for p in physical
                for out in _fuse_spatial_sectors(q,p)) for q in sectors[i]})
        allowed=[set() for _ in range(n+1)];rng=np.random.default_rng(seed)
        for _ in range(cap):
            q=sectors[0][0];allowed[0].add(q)
            for i in range(n):
                choices=[out for p in physical for out in _fuse_spatial_sectors(q,p)
                    if completions[i+1][out]>0]
                weights=np.asarray([completions[i+1][out] for out in choices],dtype=float)
                q=choices[rng.choice(len(choices),p=weights/weights.sum())]
                allowed[i+1].add(q)
        sectors=[tuple(q for q in layer if q in allowed[i]) for i,layer in enumerate(sectors)]
    forward=[Counter({sectors[0][0]:1})]
    for i in range(n):
        counts=Counter()
        for q,r in forward[-1].items():
            for p in physical:
                for out in _fuse_spatial_sectors(q,p):
                    if out in sectors[i+1]:counts[out]+=r
        forward.append(counts)
    backward=[None]*(n+1);backward[n]=Counter({sectors[-1][0]:1})
    for i in reversed(range(n)):
        backward[i]=Counter({q:sum(backward[i+1][out] for p in physical for out in _fuse_spatial_sectors(q,p)) for q in sectors[i]})
    layouts=[Counter({sectors[0][0]:1})]
    for cut in range(1,n):
        counts=Counter({q:1 for q in sectors[cut]});weight=lambda q:1 if convention=='multiplets' else q.irrep.dim
        used=sum(weight(q) for q in counts)
        if used>cap:raise ValueError(f'Cap {cap} cannot retain one copy of each reachable sector at cut {cut}; adaptive sector selection is needed')
        while True:
            changed=False
            for q in sectors[cut]:
                if counts[q]<min(forward[cut][q],backward[cut][q]) and used+weight(q)<=cap:
                    counts[q]+=1;used+=weight(q);changed=True
            if not changed:break
        layouts.append(counts)
    layouts.append(Counter({sectors[-1][0]:1}))
    dimensions=[dict(multiplets=sum(x.values()),magnetic=sum(q.irrep.dim*d for q,d in x.items())) for x in layouts[1:-1]]
    if allocation_only:return dimensions
    rng=np.random.default_rng(seed);sites=[]
    for i in range(n):
        left,right=layouts[i:i+2];data={}
        for ql,dl in left.items():
            for qp in physical:
                for qr in _fuse_spatial_sectors(ql,qp):
                    if qr in right:data[ql,qp,qr]=rng.normal(size=(dl,1,right[qr]))/np.sqrt(max(1,dl*right[qr]))
        qns=[[q for q,d in left.items() for _ in range(d)],list(physical),[q for q,d in right.items() for _ in range(d)]]
        sites.append(type(base[0])(data=data,qns=qns,dirs=[-1,1,1],metadata={'physical_basis':'fully_reduced_su2'}))
    dimensions=[dict(multiplets=sum(x.values()),magnetic=sum(q.irrep.dim*d for q,d in x.items())) for x in layouts[1:-1]]
    return sites,dimensions


def lift_mps(parent,neighborhoods):
    from pyqed._letta_one_site_opt import ReducedLatticeLETTA
    tensors=[]
    for i,blocks in enumerate(parent.tensors):
        ties=len(neighborhoods[i])-1
        tensors.append({key:np.broadcast_to(value.reshape(value.shape[:2]+(1,)*ties+value.shape[2:]),
            value.shape[:2]+(parent.physical_dim,)*ties+value.shape[2:]).copy() for key,value in blocks.items()})
    return ReducedLatticeLETTA(parent.lattice_shape,parent.symmetry,tensors,bond_sectors=parent.bond_sectors,
        coordinates=parent.coordinates,neighborhoods=neighborhoods,normalize=False)

from pathlib import Path
import time,json
def start_memory_log(out):
    """Record process memory without retaining numerical objects."""
    import atexit, resource, sys, threading
    stop=threading.Event();start=time.monotonic()
    def sample():
        row={'elapsed_seconds':time.monotonic()-start,'unix_time':time.time(),
             'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
        status=Path('/proc/self/status')
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith('VmRSS:'):row['rss_bytes']=int(line.split()[1])*1024
        with (out/'memory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def monitor():
        while not stop.wait(30):sample()
    sample();thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def finish():
        stop.set();thread.join();sample()
    atexit.register(finish)
