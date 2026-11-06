"""Paired precision checks of local fields, thermal expansion and first failure."""
import argparse, gc
from physics import *
from mixed_preconditioner import enable_reference
from failure import FAILURE, local_stress_basis, envelope, directions

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    gpu_guard();torch.set_num_threads(1)
    data=read_dataset(a.data,[0,42,63]);results={}
    for kind,material in [('thermal',THERMAL),('failure',FAILURE)]:
        pairs={}
        for precision in ['fp64','mixed']:
            gpu_guard();solver=build_batch(data,0,3,material);op=solver.op
            enable_reference(op,2,precision)
            R,mask=rotation_fields(data,0,3,op.device)
            if kind=='thermal':
                ch,S,loads=elastic_basis(op,retain='stress')
                alpha=expansion_field(R,mask,material)
                ah,u,avg,report=thermal_response(op,ch,alpha)
                outputs=dict(C_H=ch,stress_basis=S.cpu().numpy(),alpha_H=ah)
                loads=loads+[report];del alpha,u
            else:
                ch,S,loads=local_stress_basis(op,R)
                onset,modes,_=envelope(S,mask,directions(360))
                outputs=dict(C_H=ch,stress_localization=S.cpu().numpy(),onset=onset)
            pairs[precision]=dict(outputs=outputs,reports=loads,
                material_dtype=str(op.C.dtype),symbol_dtype=str(op._fft_symbol_inv.dtype))
            del solver,op,R,mask,S;gc.collect();torch.cuda.empty_cache()
        metrics={}
        for key,base in pairs['fp64']['outputs'].items():
            delta=pairs['mixed']['outputs'][key]-base
            percell=np.linalg.norm(delta.reshape(3,-1),axis=1)/np.linalg.norm(base.reshape(3,-1),axis=1)
            metrics[key+'_max_relative_norm']=float(percell.max())
            assert percell.max()<1e-8,(kind,key,percell)
        if kind=='failure':
            metrics['onset_max_pointwise_relative']=float(np.max(np.abs(pairs['mixed']['outputs']['onset']/pairs['fp64']['outputs']['onset']-1)))
            assert metrics['onset_max_pointwise_relative']<1e-8
        for v in pairs.values():
            # Keep reproducible small outputs; full field metrics are above.
            v['outputs']={k:x.tolist() for k,x in v['outputs'].items() if k in ['C_H','alpha_H','onset']}
        results[kind]=dict(material=material,metrics=metrics,runs=pairs)
        print(kind,metrics,flush=True)
    save_json(a.out,dict(status='passed',indices=[0,42,63],dataset_sha256=data['sha256'],
        results=results,provenance=provenance(),
        source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
        protocol='three-cell paired output checks; FP64 physics and residual 1e-10 for both; no performance claim'))

if __name__=='__main__':main()
