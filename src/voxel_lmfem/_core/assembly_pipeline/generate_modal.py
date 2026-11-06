"""Generate force-plus-energy CUDA arithmetic from the verified CPU formulas."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "modal_reference/generated_kernels.py"


def generate():
    tree = ast.parse(SOURCE.read_text())
    parts = ["// Generated from the CPU-verified scalar modal formulas."]
    for kind in ("hg", "fi"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == kind + "_modal_local")
        lines = ["template<bool energy>",
                 f"__device__ __forceinline__ double {kind}_eval(const double* u, const double* c, double scale, const double* g, double* f) {{",
                 "  double value = 0.;"]
        for node in fn.body:
            if isinstance(node, ast.Assign):
                name = node.targets[0].id
                lines.append(f"  const double {name} = {ast.unparse(node.value)};")
                if name.startswith("s") and name[1].isdigit():
                    lines.append(f"  if constexpr(energy) value += e{name[1:]} * {name};")
            elif isinstance(node, ast.Return):
                lines.extend(f"  f[{i}] = {ast.unparse(v)};" for i, v in enumerate(node.value.elts))
            else:
                raise TypeError(ast.dump(node))
        if kind == "hg":
            terms = " + ".join(f"q2_{m}_{d} * q2_{m}_{d}" for m in (3, 5, 6, 7) for d in range(3))
            lines.append(f"  if constexpr(energy) value += (scale * 0.125) * ({terms});")
        lines.extend(["  return value;", "}"])
        parts.append("\n".join(lines))
    (HERE / "modal_functions.cuh").write_text("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    generate()
