"""HG stencil assembly definitions used inside the frozen runtime namespace."""
STENCIL=ASSEMBLY=None
ROUTE='matrix_free';CALLS=[];CONTEXT={}

def extensions():
    global STENCIL,ASSEMBLY
    STENCIL=load(name='hg_stencil_apply_20260930',sources=[str(OUT/'c3d8.cpp'),str(OUT/'c3d8_cuda.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],verbose=False)
    ASSEMBLY=load(name='hg_stencil_assembly_20260930',sources=[str(OUT/'hg_assembly.cpp'),str(OUT/'hg_assembly.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],verbose=False)

def assemble(op,taps):
    # Templates are tiny geometry-only arrays, rebuilt within measured setup.
    B=op.B_center.detach().cpu().numpy()
    H=hourglass_basis().reshape(8,3,12)
    T=np.zeros((taps,8,9,21));HH=np.zeros((taps,8,9));valid=np.zeros((taps,8),dtype=np.int32)
    pairs=list(zip(*np.triu_indices(6)))
    for a in range(8):
        for b in range(8):
            d=LOCAL_OFFSET[b]-LOCAL_OFFSET[a]
            tap=int((d[2]+1)*9+(d[1]+1)*3+d[0]+1)
            if tap>=taps: continue
            valid[tap,a]=1
            for pq in range(9):
                p,q=divmod(pq,3)
                for c,(i,j) in enumerate(pairs):
                    T[tap,a,pq,c]=op.mesh.V_elem*(B[i,a,p]*B[j,b,q]+(B[j,a,p]*B[i,b,q] if i!=j else 0))
                HH[tap,a,pq]=H[a,p]@H[b,q]
    device=op.device
    return ASSEMBLY.assemble(op.C,op.hg_scale,torch.as_tensor(T,device=device),torch.as_tensor(HH,device=device),
                            torch.as_tensor(valid,device=device),torch.as_tensor(LOCAL_OFFSET,dtype=torch.int32,device=device),
                            op.mesh.Nz,op.mesh.Ny,op.mesh.Nx)

class HGStencil(HG):
    def apply(self,u):
        self._validate_apply_input(u)
        return STENCIL.apply(u,self.S_hg,self.mesh.Nz,self.mesh.Ny,self.mesh.Nx,128)

