"""Keep the verified original modal HG algebra, replace only C:e action."""
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
source=(HERE.parent/"pipeline_gpu_20261020/modal_functions.cuh").read_text()
source=source[:source.index("template<bool energy>\n__device__ __forceinline__ double fi_eval")]
source=source.replace("hg_eval(const double* u, const double* c, double scale, const double* g, double* f)",
                      "structural_modal_eval(const double* u, const double* frame, const double* params, bool yarn, const double* g, double* f)")
source=source.replace("  double value = 0.;", "  double value = 0.;\n  const double scale=params[yarn?9:8];",1)
needle="  const double s0_0 ="
position=source.index(needle)
source=source[:position]+"  const double strain_value[6]={e0_0,e0_1,e0_2,e0_3,e0_4,e0_5};\n  double stress_value[6];\n  structural_stress(strain_value,frame,params,yarn,stress_value);\n"+source[position:]
source,count=re.subn(r"  const double s0_([0-5]) = [^\n]+;",lambda m:f"  const double s0_{m[1]} = stress_value[{m[1]}]*g[3];",source)
assert count==6 and "c[" not in source
(HERE/"structural_modal.cuh").write_text(source)
