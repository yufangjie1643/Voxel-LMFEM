"""Full 27-tap FI assembly paired with the existing 14-tap implementation.

Both use the same packed C21 contraction, slab size and CUDA stencil kernel.
The three source substitutions only select all taps instead of the lower half.
No expanded 14-tap cache is created on the 27-tap path.
"""
import inspect


def full_assembly(spatial):
    source = inspect.getsource(spatial.slab_assembly)
    changes = [
        ('def slab_assembly(', 'def slab_assembly27('),
        ('torch.zeros((B,14,9,Nz,Ny,Nx)', 'torch.zeros((B,27,9,Nz,Ny,Nx)'),
        ('selected=np.flatnonzero(taps<=13)', 'selected=np.arange(8)'),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise RuntimeError('Assembly source changed: ' + old)
        source = source.replace(old, new)
    namespace = dict(vars(spatial))
    exec(compile(source, __file__ + ':full27', 'exec'), namespace)
    return namespace['slab_assembly27'], source


def install():
    import spatial_adapter
    assembly, source = full_assembly(spatial_adapter)
    spatial_adapter.slab_assembly = assembly
    return source
