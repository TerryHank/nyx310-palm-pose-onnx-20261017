"""Compare sparse projection with the original dense path on planes and real raw data."""
import ctypes as c
import json
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=c.c_void_p
def ptr(a):return a.ctypes.data_as(P)


def main():
    lib=c.CDLL(str(ROOT/'build-host/libreplay.so'))
    lib.core_measure_raw_sampled.argtypes=[P,P,P,c.c_int,c.c_int,c.c_int,c.c_int,P,P,P,P,c.c_int,P]
    rays=np.fromfile(ROOT.parent/'lowmem/depth_rays_xy_float32.bin',np.float32).reshape(480,640,2)
    calibration=json.loads((ROOT.parent/'lowmem/calibration.json').read_text());color=calibration['color']
    K=np.array([[color['fx']*.4,0,color['cx']*.4],[0,color['fy']*.4,color['cy']*.4],[0,0,1]],np.float64)
    D=np.array([color[n] for n in ('k1','k2','p1','p2','k3','k4','k5','k6')],np.float64)
    R=np.array(calibration['rotation'],np.float64).reshape(3,3);t=np.array(calibration['translation_mm'],np.float64)
    points=np.zeros((21,3),np.float64)
    for i,p in {0:(320,300),1:(285,280),5:(280,230),9:(320,220),13:(350,230),17:(365,255)}.items():points[i,:2]=p
    cases=[]
    for z in (200,600,800,1500,3500):cases.append((f'plane_{z}',np.full((480,640),z,np.uint16)))
    real=np.load(ROOT.parent/'lowmem/depth_ray_verification.npz')
    for key in real.files:
        if real[key].shape==(480,640) and real[key].dtype==np.uint16:
            cases.append(('recorded_'+key,np.ascontiguousarray(real[key])))
    records=[]
    def run(depth,step):
        out=np.zeros(3,np.float64)
        ok=lib.core_measure_raw_sampled(ptr(points),ptr(depth),ptr(rays),640,480,640,480,ptr(K),ptr(D),ptr(R),ptr(t),step,ptr(out))
        return bool(ok),out
    for name,depth in cases:
        dense,a=run(depth,1);sampled,b=run(depth,2)
        difference=float(np.max(np.abs(a-b))) if dense and sampled else None
        if name.startswith('plane'):
            assert dense and sampled,(name,dense,sampled)
            assert difference<2,(name,difference)
        records.append(dict(case=name,dense_valid=dense,sampled_valid=sampled,max_xyz_difference_mm=difference))
    depth=cases[2][1]
    durations={}
    for step in (1,2):
        start=time.perf_counter()
        for _ in range(50):run(depth,step)
        durations[str(step)]=(time.perf_counter()-start)*1000/50
    report=dict(passed=True,cases=records,host_dense_ms=durations['1'],host_sampled_ms=durations['2'],
                host_speedup=durations['1']/durations['2'],camera_benchmark=False,
                note='Planes are synthetic; recorded raw sample is not a hand truth measurement. Sparse failure triggers dense fallback in production.')
    (ROOT/'sampled_depth_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
