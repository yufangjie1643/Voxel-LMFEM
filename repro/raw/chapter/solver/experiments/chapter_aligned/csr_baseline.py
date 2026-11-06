"""Assembled FP64 HEX8 CSR baseline; CuPy SpMV and library CG.

Eight-point element integration, periodic identification, scalar Jacobi on
the zero-mean displacement subspace. Two bounded assembly passes remove
exact zeros without retaining a full COO or stencil cache. No modal,
background, mixed-precision or custom SpMV implementation is used here.
"""
from functools import lru_cache
from types import SimpleNamespace
import itertools
import time
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import cg, LinearOperator

OFF = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0),
                (0,0,1),(1,0,1),(1,1,1),(0,1,1)], dtype=np.int32)
PAIRS = [(i,j) for i in range(6) for j in range(i,6)]


def geometry(voxel):
    sign = 2*OFF-1
    def at(p):
        grad = np.empty((8,3))
        for d in range(3):
            others = [j for j in range(3) if j != d]
            grad[:,d] = sign[:,d]*np.prod(1+sign[:,others]*np.asarray(p)[others],axis=1)/(4*voxel[d])
        B = np.zeros((6,8,3))
        B[0,:,0],B[1,:,1],B[2,:,2] = grad[:,0],grad[:,1],grad[:,2]
        B[3,:,1],B[3,:,2] = grad[:,2],grad[:,1]
        B[4,:,0],B[4,:,2] = grad[:,2],grad[:,0]
        B[5,:,0],B[5,:,1] = grad[:,1],grad[:,0]
        return B
    B = np.array([at(p) for p in itertools.product((-1/np.sqrt(3),1/np.sqrt(3)),repeat=3)])
    T = np.einsum('gIar,gJbc->abrcIJ',B,B)*np.prod(voxel)/8
    packed = np.stack([T[...,i,j] if i==j else T[...,i,j]+T[...,j,i] for i,j in PAIRS],axis=-1)
    W = np.zeros((27,8,3,3,21))
    for a in range(8):
        for b in range(8):
            dx,dy,dz = OFF[b]-OFF[a]
            W[(dz+1)*9+(dy+1)*3+dx+1,a] = packed[a,b]
    return B, at((0,0,0)), W


CUDA = r'''
__device__ int wrap(int x,int n){return (x+n)%n;}
__device__ int pk(int i,int j){if(i>j){int t=i;i=j;j=t;}return i*6-i*(i-1)/2+j-i;}
__device__ int sorted_neighbour(int x,int rank,int n){
  int a=wrap(x-1,n),b=x,c=wrap(x+1,n),t;
  if(a>b){t=a;a=b;b=t;}if(b>c){t=b;b=c;c=t;}if(a>b){t=a;a=b;b=t;}
  return rank==0?a:rank==1?b:c;
}
extern "C" __global__ void entries(const double* C,const double* W,const int* off,
 double* v,int nx,int ny,int nz,long long start,long long nodes){
 long long q=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(q>=nodes*243)return;
 int k=q%81,r=(q/81)%3,c=k%3,s=k/3;
 long long node=start+q/243;
 int x=node%nx,y=(node/nx)%ny,z=node/(nx*ny);
 int xx=sorted_neighbour(x,s%3,nx),yy=sorted_neighbour(y,(s/3)%3,ny),zz=sorted_neighbour(z,s/9,nz);
 int dx=xx==x?0:xx==wrap(x-1,nx)?-1:1;
 int dy=yy==y?0:yy==wrap(y-1,ny)?-1:1;
 int dz=zz==z?0:zz==wrap(z-1,nz)?-1:1;
 int tap=(dz+1)*9+(dy+1)*3+dx+1;
 double value=0.;
 for(int a=0;a<8;++a){
   int ox=off[3*a],oy=off[3*a+1],oz=off[3*a+2];
   if(ox+dx<0||ox+dx>1||oy+dy<0||oy+dy>1||oz+dz<0||oz+dz>1)continue;
   long long e=((long long)wrap(z-oz,nz)*ny+wrap(y-oy,ny))*nx+wrap(x-ox,nx);
   const double* w=W+(((tap*8+a)*3+r)*3+c)*21;
   for(int j=0;j<21;++j)value+=w[j]*C[e*21+j];
 }
 v[q]=value;
}
extern "C" __global__ void columns(const long long* row,const long long* col,int* out,
 long long size,long long start,int nx,int ny,int nz){
 long long q=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(q>=size)return;
 long long node=start+row[q]/3;int s=col[q]/3;
 int x=node%nx,y=(node/nx)%ny,z=node/(nx*ny);
 int xx=sorted_neighbour(x,s%3,nx),yy=sorted_neighbour(y,(s/3)%3,ny),zz=sorted_neighbour(z,s/9,nz);
 out[q]=(((zz*ny)+yy)*nx+xx)*3+col[q]%3;
}
extern "C" __global__ void diagonal(const double* values,const int* indices,const int* ptr,double* out,int rows){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;
 long long lo=ptr[r],hi=(long long)ptr[r+1]-1;double v=0;
 while(lo<=hi){long long m=lo+(hi-lo)/2;int c=indices[m];
   if(c==r){v=values[m];break;}if(c<r)lo=m+1;else hi=m-1;
 }out[r]=v;
}
extern "C" __global__ void rhs(const double* C,const double* B,const int* off,const double* E,
 double* f,int nx,int ny,int nz,double vol){
 long long q=(long long)blockIdx.x*blockDim.x+threadIdx.x,N=(long long)nx*ny*nz;
 if(q>=3*N)return;long long node=q/3;int r=q%3;
 int x=node%nx,y=(node/nx)%ny,z=node/(nx*ny);double v=0;
 for(int a=0;a<8;++a){
  long long e=((long long)wrap(z-off[3*a+2],nz)*ny+wrap(y-off[3*a+1],ny))*nx+wrap(x-off[3*a],nx);
  for(int i=0;i<6;++i){double s=0;for(int j=0;j<6;++j)s+=C[e*21+pk(i,j)]*E[j];v-=B[(i*8+a)*3+r]*s*vol;}
 }f[q]=v;
}
extern "C" __global__ void stress(const double* C,const double* B,const int* off,const double* E,
 const double* u,double* out,int nx,int ny,int nz){
 long long e=(long long)blockIdx.x*blockDim.x+threadIdx.x,N=(long long)nx*ny*nz;if(e>=N)return;
 int x=e%nx,y=(e/nx)%ny,z=e/(nx*ny);double strain[6];
 for(int i=0;i<6;++i)strain[i]=E[i];
 for(int a=0;a<8;++a){
   long long n=((long long)wrap(z+off[3*a+2],nz)*ny+wrap(y+off[3*a+1],ny))*nx+wrap(x+off[3*a],nx);
   for(int i=0;i<6;++i)for(int c=0;c<3;++c)strain[i]+=B[(i*8+a)*3+c]*u[n*3+c];
 }
 for(int i=0;i<6;++i){double s=0;for(int j=0;j<6;++j)s+=C[e*21+pk(i,j)]*strain[j];out[e*6+i]=s;}
}
'''


@lru_cache(None)
def kernels():
    module = cp.RawModule(code=CUDA, options=('--std=c++11','--fmad=false'))
    return {name:module.get_function(name) for name in ('entries','columns','diagonal','rhs','stress')}


class AssembledCSR:
    def __init__(self, coefficients, grid, voxel, chunk_nodes=32768):
        self.C = coefficients.reshape(-1,21)
        self.grid,self.voxel = tuple(map(int,grid)),tuple(map(float,voxel))
        nz,ny,nx = self.grid
        if min(grid)<3:raise ValueError('CSR neighbour layout requires at least three cells per axis')
        self.n = nx*ny*nz
        self.mesh = SimpleNamespace(Nx=nx,Ny=ny,Nz=nz,n_dofs=3*self.n,n_elem=self.n)
        self.B,self.B0,W = geometry(self.voxel)
        self.w,self.b0,self.off = cp.asarray(W),cp.asarray(self.B0),cp.asarray(OFF)
        self.chunk = chunk_nodes
        self.assemble()

    def tile(self,start,stop):
        v=cp.empty((3*(stop-start),81),dtype=cp.float64)
        kernels()['entries'](((v.size+127)//128,),(128,),
            (self.C,self.w,self.off,v,*map(np.int32,self.grid[::-1]),np.int64(start),np.int64(stop-start)))
        return v

    def assemble(self):
        lengths=cp.empty(3*self.n,dtype=cp.int32)
        chunks=list(range(0,self.n,self.chunk))
        for start in chunks:
            stop=min(self.n,start+self.chunk);v=self.tile(start,stop)
            lengths[3*start:3*stop]=cp.count_nonzero(v,axis=1)
            del v
        ptr64=cp.empty(3*self.n+1,dtype=cp.int64);ptr64[0]=0
        cp.cumsum(lengths,out=ptr64[1:]);nnz=int(ptr64[-1])
        if nnz>=2**31:raise ValueError('This campaign uses 32-bit CSR indices; nnz exceeds that range')
        ptr=ptr64.astype(cp.int32);del ptr64,lengths
        values=cp.empty(nnz,dtype=cp.float64);indices=cp.empty(nnz,dtype=cp.int32)
        for start in chunks:
            stop=min(self.n,start+self.chunk);v=self.tile(start,stop)
            row,col=(cp.ascontiguousarray(a) for a in cp.nonzero(v))
            lo,hi=int(ptr[3*start]),int(ptr[3*stop])
            if row.size!=hi-lo:raise RuntimeError('Assembly passes disagree')
            values[lo:hi]=v[row,col]
            kernels()['columns'](((row.size+255)//256,),(256,),
                (row,col,indices[lo:hi],np.int64(row.size),np.int64(start),*map(np.int32,self.grid[::-1])) )
            del v,row,col
        self.matrix=csr_matrix((values,indices,ptr),shape=(3*self.n,3*self.n),copy=False)
        self.matrix.has_sorted_indices=True
        self.matrix.has_canonical_format=True
        # CuPy 14.2 diagonal() adds two int32 nnz offsets before dividing by
        # two; their sum can overflow even when each CSR offset is valid.
        self.diag=cp.empty(3*self.n,dtype=cp.float64)
        kernels()['diagonal'](((3*self.n+255)//256,),(256,),
            (values,indices,ptr,self.diag,np.int32(3*self.n)))
        if not bool(cp.all(self.diag>0)):raise ValueError('Nonpositive Jacobi diagonal')
        self.inverse_diag=1/self.diag
        self.preconditioner=LinearOperator(self.matrix.shape,matvec=self.apply_precond,dtype=cp.float64)
        self.metadata=dict(format='CuPy CSR',nnz=nnz,index_dtype=str(indices.dtype),
            csr_bytes=values.nbytes+indices.nbytes+ptr.nbytes,exact_zeros_removed=True,
            assembly='Two-pass bounded-row element assembly from eight Gauss points',
            solver='cupyx.scipy.sparse.linalg.cg',preconditioner='scalar Jacobi with periodic zero-mean projection')
        cp.get_default_memory_pool().free_all_blocks()

    def project_zero_mean(self,u):
        v=u.reshape(-1,3)
        return (v-v.mean(axis=0,keepdims=True)).ravel()

    def apply(self,u):return self.matrix@u.ravel()

    def apply_precond(self,r):return self.project_zero_mean(self.inverse_diag*r)

    def compute_rhs(self,macro):
        e=cp.asarray(macro,dtype=cp.float64);f=cp.empty(3*self.n,dtype=cp.float64)
        kernels()['rhs'](((f.size+127)//128,),(128,),
            (self.C,self.b0,self.off,e,f,*map(np.int32,self.grid[::-1]),np.float64(np.prod(self.voxel))))
        return f

    def compute_avg_stress(self,u,macro):
        e=cp.asarray(macro,dtype=cp.float64);s=cp.empty((self.n,6),dtype=cp.float64)
        kernels()['stress'](((self.n+127)//128,),(128,),
            (self.C,self.b0,self.off,e,u,s,*map(np.int32,self.grid[::-1])))
        return cp.asnumpy(s.mean(axis=0))[None]

    def solve(self,rhs,tolerance=1e-8):
        norm=float(cp.linalg.norm(rhs));count=0;restarts=0;u=None
        if norm==0:return cp.zeros_like(rhs),dict(iterations=0,restarts=0,true_residuals=[0.])
        def callback(_):
            nonlocal count
            count+=1
        while count<10000:
            u,info=cg(self.matrix,rhs,x0=u,rtol=tolerance,atol=0.,maxiter=10000-count,
                      M=self.preconditioner,callback=callback)
            u=self.project_zero_mean(u)
            residual=float(cp.linalg.norm(rhs-self.apply(u)))/norm
            if not np.isfinite(residual):raise RuntimeError('Nonfinite true residual')
            if residual<=tolerance:return u,dict(iterations=count,restarts=restarts,true_residuals=[residual],cg_info=int(info))
            if info:break
            restarts+=1
        raise RuntimeError(f'CSR failed after {count} iterations: {residual}')


def validate():
    rng=np.random.default_rng(20261024);grid=(4,5,6);voxel=(.3,.5,.7);n=int(np.prod(grid))
    q=rng.normal(size=(n,6,6));C=q@q.transpose(0,2,1)+np.eye(6)*3
    op=AssembledCSR(cp.asarray(np.stack([C[:,i,j] for i,j in PAIRS],-1)),grid,voxel,chunk_nodes=17)
    B,B0,_=geometry(voxel);K=np.zeros((3*n,3*n));f=np.zeros((3*n,6));nz,ny,nx=grid
    for e in range(n):
        x,y,z=e%nx,(e//nx)%ny,e//(nx*ny)
        nodes=((z+OFF[:,2])%nz*ny+(y+OFF[:,1])%ny)*nx+(x+OFF[:,0])%nx
        dofs=(nodes[:,None]*3+np.arange(3)).ravel()
        ke=sum(bg.reshape(6,24).T@C[e]@bg.reshape(6,24) for bg in B)*np.prod(voxel)/8
        K[np.ix_(dofs,dofs)]+=ke
        f[dofs]-=B0.reshape(6,24).T@C[e]*np.prod(voxel)
    matrix=op.matrix.get().toarray()
    relative=lambda a,b:float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
    errors=dict(matrix=relative(matrix,K),symmetry=relative(matrix,matrix.T))
    translation=np.tile(np.eye(3),(n,1));errors['rigid_translation']=float(np.linalg.norm(matrix@translation)/np.linalg.norm(matrix))
    calls=[]
    for j in range(6):
        macro=np.eye(6)[j];rhs=op.project_zero_mean(op.compute_rhs(macro));u,report=op.solve(rhs,1e-10)
        reference=np.zeros(3*n);reference[3:]=np.linalg.solve(K[3:,3:],f[3:,j]);reference-=np.tile(reference.reshape(n,3).mean(0),n)
        s=np.zeros(6)
        for e in range(n):
            x,y,z=e%nx,(e//nx)%ny,e//(nx*ny)
            nodes=((z+OFF[:,2])%nz*ny+(y+OFF[:,1])%ny)*nx+(x+OFF[:,0])%nx
            s+=C[e]@(macro+B0.reshape(6,24)@reference.reshape(n,3)[nodes].ravel())/n
        calls.append(dict(rhs=relative(cp.asnumpy(rhs),f[:,j]),displacement=relative(cp.asnumpy(u),reference),
                          effective_stress=relative(op.compute_avg_stress(u,macro)[0],s),**report))
    assert max(errors.values())<1e-12,errors
    assert max(max(r['rhs'],r['displacement'],r['effective_stress']) for r in calls)<1e-8,calls
    return dict(passed=True,grid=grid,voxel=voxel,material='random fully anisotropic SPD',errors=errors,calls=calls,
                metadata=op.metadata,cupy=cp.__version__)
