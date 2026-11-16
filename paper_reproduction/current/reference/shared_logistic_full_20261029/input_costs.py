"""Common input-cache loading and public-design construction, charged to every plan."""
from study import *
import platform,sys,scipy


def main():
    rows={}
    for c in CASES:
        tm=time.perf_counter();a,o,y,pools=load_domain(c['domain']);elapsed=time.perf_counter()-tm
        rows[c['id']]=dict(seconds=elapsed,training=len(pools['training']),comparison=len(pools['comparison']),labels=y.shape[1],dimensions=a.shape[-1],
                          max_row_norm=max(float(np.linalg.norm(a,axis=-1).max()),float(np.linalg.norm(o,axis=-1).max())))
        print(c['id'],elapsed,flush=True)
        del a,o,y,pools
    save(ROOT/'input_costs.json',dict(cases=rows,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        convention='Same cache-loading and fixed-design construction charge for every method; original encoder extraction is outside the online comparison.'))


if __name__=='__main__':main()
