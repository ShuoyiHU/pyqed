"""Run a worker and report child peak RSS without an external time executable."""
import json
import resource
import subprocess
import sys
import time

if __name__ == '__main__':
    start=time.monotonic()
    result=subprocess.run(sys.argv[1:])
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    rss_bytes=usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    print(json.dumps(dict(worker_elapsed_seconds=time.monotonic()-start,
        child_peak_rss_bytes=rss_bytes,returncode=result.returncode)),file=sys.stderr,flush=True)
    sys.exit(result.returncode if result.returncode>=0 else 128-result.returncode)
