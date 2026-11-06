"""Portable access to the exact prior stopping routine and frozen inputs."""
from pathlib import Path
from types import SimpleNamespace
import ast,os,subprocess
import torch
from femlib.materials import isotropic_to_voigt
from femlib.batched.pcg import solve_pcg_batched
ROOT=Path(__file__).resolve().parent
PAIRS=[(i,j) for i in range(6) for j in range(i,6)]
def function_source(file,name):
    source=file.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(source,node)
certification_source=function_source(ROOT/'source_support/physics.py.txt','solve_checked')
exec(compile(certification_source,'<frozen-solve-checked>','exec'),globals())
exec(function_source(ROOT/'source_support/previous_benchmark.py.txt','guard'),globals())
def prepare(case,n):
    p=ROOT/'inputs'/f'{case}_{n}.npy'
    if not p.exists():raise FileNotFoundError(p)
    return p
previous=SimpleNamespace(guard=guard,prepare=prepare)
