"""CPU-only equivalence to the existing analytic geometry oracle."""
import ast,importlib.util,sys,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
original=P/'geometry_reference/woven_anisotropic.py'
sys.path.insert(0,str(original.parent))
spec=importlib.util.spec_from_file_location('analytic_weave_reference',original)
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
tree=ast.parse((P/'run_resolution.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='geometry')
scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(P/'run_resolution.py'),'exec'),scope)
rows=[]
for N in [8,16,32,64]:
    data,g=scope['geometry'](N);ids,_,vf=ref.material_grid((N,N,N),ref.Weave(),'ti40')
    assert np.array_equal(data['yarn_id'][0]>=0,(ids!=0).transpose(2,1,0))
    rows.append(dict(N=N,voxel_fraction=vf,exact_fraction=g['exact_yarn_fraction'],phase_match=True))
out=P.parents[1]/'results/geometry_validation.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(status='passed',geometry=ref.Weave().validate(),checks=rows),indent=2))
print('GEOMETRY VALIDATED',rows)
