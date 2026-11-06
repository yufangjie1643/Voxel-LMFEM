"""Generate exact isotropic specialization of the frozen FI modal algebra.

python generate_isotropic.py
Only explicitly declared isotropic phase rows may use this specialization.
"""
import ast
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent


class Isotropic(ast.NodeTransformer):
    def visit_Subscript(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=="c":
            k=ast.literal_eval(node.slice)
            name="d" if k in (0,6,11) else "l" if k in (1,2,7) else "m" if k in (15,18,20) else None
            return ast.Name(id=name,ctx=ast.Load()) if name else ast.Constant(value=0.)
        return self.generic_visit(node)

    def visit_BinOp(self,node):
        node=self.generic_visit(node)
        def zero(n):return isinstance(n,ast.Constant) and n.value==0
        if isinstance(node.op,ast.Mult) and (zero(node.left) or zero(node.right)):return ast.Constant(value=0.)
        if isinstance(node.op,(ast.Add,ast.Sub)) and zero(node.right):return node.left
        if isinstance(node.op,ast.Add) and zero(node.left):return node.right
        return node


def main():
    text=(HERE/"fi_modal.cuh").read_text()
    text=text.replace("void fi_modal(","void fi_isotropic(")
    text=text.replace("{\n","{\n  const double d=c[0], l=c[1], m=c[15];\n",1)
    pattern=r"(  const double [A-Za-z0-9_]+ = )([^;]+);"
    count=0
    def substitute(match):
        nonlocal count
        expr=ast.parse(match[2],mode="eval")
        if "c[" in match[2]:count+=1
        expr=ast.fix_missing_locations(Isotropic().visit(expr))
        return match[1]+ast.unparse(expr.body)+";"
    text=re.sub(pattern,substitute,text)
    (HERE/"fi_isotropic.cuh").write_text(text,encoding="utf-8")
    print("Specialized constitutive expressions",count)


if __name__=="__main__":main()
