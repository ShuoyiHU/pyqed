"""Optional benchmark phase timing; never changes solver tolerances or arrays."""
from functools import wraps
from pathlib import Path
import faulthandler,json,resource,sys,time


def install(out):
    import pyqed._letta_one_site_opt.reduced_solver as solver
    import pyqed._letta_one_site_opt.reduced_cbe as cbe
    from pyqed._letta_one_site_opt.reduced_coordinates import ReducedLocalCoordinates
    start=time.monotonic();path=Path(out)/'phases.jsonl'
    stack=(Path(out)/'stacks.log').open('a',buffering=1)
    faulthandler.dump_traceback_later(600,repeat=True,file=stack)
    import atexit
    def close_trace():
        faulthandler.cancel_dump_traceback_later()
        stack.close()
    atexit.register(close_trace)
    def emit(phase,event,**fields):
        row=dict(phase=phase,event=event,elapsed_seconds=time.monotonic()-start,
                 unix_time=time.time(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),**fields)
        with path.open('a') as f:f.write(json.dumps(row)+'\n')
    def wrap(owner,name,phase,describe):
        original=getattr(owner,name)
        @wraps(original)
        def traced(*args,**kwargs):
            fields=describe(args,kwargs)
            emit(phase,'begin',**fields);t=time.monotonic()
            try:value=original(*args,**kwargs)
            except BaseException as error:
                emit(phase,'error',seconds=time.monotonic()-t,error=type(error).__name__,**fields);raise
            emit(phase,'end',seconds=time.monotonic()-t,**fields)
            if phase=='cbe_site':
                names=('energy','cbe_baseline_energy','cbe_expansion_dimension','cbe_recovery_reason',
                       'cbe_energy_refinement_iterations','cbe_projection_iterations','local_converged')
                row={name:getattr(value,name,None) for name in names}
                row.update(site=args[2],direction=args[3],elapsed_seconds=time.monotonic()-start,
                           seconds=time.monotonic()-t)
                with (Path(out)/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            return value
        setattr(owner,name,traced)
    wrap(solver,'optimize_reduced_site','one_site',lambda a,k:{'site':a[2]})
    wrap(cbe,'reduced_cbe_site','cbe_site',lambda a,k:{'site':a[2]})
    wrap(solver,'_davidson_in_metric_coordinates','davidson',lambda a,k:{'dimension':a[0].local_dimension,'tolerance':a[1].eigensolver_tolerance,'max_iterations':a[1].eigensolver_max_iterations})
    original=ReducedLocalCoordinates.prepare
    @wraps(original)
    def prepare(self,tolerance,**kwargs):
        if getattr(self,'tolerance',None)==(tolerance,kwargs.get('singular_tolerance'),kwargs.get('max_workspace_mb')):return original(self,tolerance,**kwargs)
        emit('overlap_svd','begin',site=self.site,blocks=len(self.blocks),dimension=self.dimension);t=time.monotonic()
        try:value=original(self,tolerance,**kwargs)
        except BaseException as error:
            emit('overlap_svd','error',site=self.site,error=type(error).__name__);raise
        emit('overlap_svd','end',site=self.site,seconds=time.monotonic()-t,rank=self.rank)
        return value
    ReducedLocalCoordinates.prepare=prepare
    original_apply=ReducedLocalCoordinates.apply
    @wraps(original_apply)
    def apply(self,vector):
        value=original_apply(self,vector)
        self._phase_matvec_count=getattr(self,'_phase_matvec_count',0)+1
        if self._phase_matvec_count%50==0:emit('hamiltonian','progress',site=self.site,calls=self._phase_matvec_count)
        return value
    ReducedLocalCoordinates.apply=apply
    emit('worker','diagnostics_installed')
