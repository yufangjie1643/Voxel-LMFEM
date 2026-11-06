"""Re-voxelize one fixed analytic weave; no tiling or nearest-neighbor upsample.

Geometry follows the pre-existing semianalytic_yarn_20260922/woven_anisotropic.py
Weave/YARNS/material_grid definitions, restated here without its independent solver.
Yarns are vertical elliptical sections, not normal-plane swept tubes.
"""
import argparse,os,gc
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--grid',type=int,required=True);p.add_argument('--config',choices=['U','A11'],required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--repeat',type=int,required=True);a=p.parse_args()
if a.config=='U':os.environ['FEAD_SOURCE']=str(Path(__file__).parent/'dense_control')
from material_control import *

def geometry(N):
    width,thickness,amplitude,height=.18,.035,.09,.4
    assert 2*amplitude*np.cos(2*np.pi*width)-2*thickness>0
    x=(np.arange(N)+.5)/N;y=x.copy();z=(x-.5)*height
    X,Y=np.meshgrid(x,y,indexing='xy')
    ids=np.full((N,N,N),-1,dtype=np.int8)
    o1=np.zeros((N,N,N,3),dtype=np.float64);o2=np.zeros_like(o1)
    for k,zz in enumerate(z):
        for i,(axis,center,sign) in enumerate([(0,0.,1),(0,.5,-1),(1,0.,-1),(1,.5,1)]):
            s=X if axis==0 else Y;t=Y if axis==0 else X
            d=(t-center+.5)%1-.5;zc=sign*amplitude*np.cos(2*np.pi*s)
            dz=(zz-zc+height/2)%height-height/2
            inside=(d/width)**2+(dz/thickness)**2<1
            if np.any(inside&(ids[k]>=0)):raise ValueError('Overlapping yarn voxels')
            ids[k][inside]=i;slope=-sign*amplitude*2*np.pi*np.sin(2*np.pi*s[inside]);norm=np.sqrt(1+slope*slope)
            v=np.zeros((len(slope),3));v[:,axis]=1/norm;v[:,2]=slope/norm
            w=np.zeros_like(v);w[:,1-axis]=1 if axis==0 else -1
            o1[k][inside]=v;o2[k][inside]=w
    return dict(grid_shape=(N,N,N),domain_size=np.array([[1.,1.,height]]),yarn_id=ids[None],orient1=o1[None],orient2=o2[None],indices=np.array([0])),dict(width=width,thickness=thickness,amplitude=amplitude,height=height,exact_yarn_fraction=4*np.pi*width*thickness/height,voxel_yarn_fraction=float((ids>=0).mean()))

if a.out.exists():raise FileExistsError(a.out)
gpu_guard();torch.set_num_threads(1);data,geom=geometry(a.grid)
mode=a.config=='A11';record=dict(config=a.config,grid=a.grid,repeat=a.repeat,geometry=geom,material=STRONG,provenance=provenance(),
    protocol='fixed physical geometry reclassified at voxel centers on each grid; six loads; true residual 1e-10; two fresh-process repetitions; CPU geometry creation and JIT excluded')
try:
    # Small same-path warmup only. Full-size runs test capacity, not timing statistics.
    warm,_=geometry(8);s=setup(warm,0,1,STRONG,mode,mode,2 if mode else 0);elastic_basis(s.op)
    del s,warm;gc.collect();torch.cuda.empty_cache();synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    solver=setup(data,0,1,STRONG,mode,mode,2 if mode else 0);C,_,reports=elastic_basis(solver.op)
    synchronize();record.update(status='passed',seconds=time.perf_counter()-start,C_H_raw=C.tolist(),reports=reports,
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
except torch.OutOfMemoryError as e:
    record.update(status='OOM',error=str(e),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
save_json(a.out,record);print('RESOLUTION',a.config,a.grid,record['status'],flush=True)
