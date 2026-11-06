"""Group equal isotropic FI stencil coefficients on rectangular voxels.

python generate_background.py
Diagonal blocks have seven sign-invariant groups. Each off-diagonal block
has a signed face sum and a signed corner sum, whose weights have ratio 4.
"""
from pathlib import Path
from background import OFFSETS

HERE=Path(__file__).resolve().parent


def main():
    lines=["// Generated exact isotropic rectangular-voxel stencil grouping.",
      "__global__ void background_grouped(const double* u,const double* S,double* out,int nx,int ny,int nz){",
      "  int64_t node=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;",
      "  if(node>=int64_t(nx)*ny*nz)return;",
      "  int x=node%nx,y=(node/nx)%ny,z=node/(int64_t(nx)*ny);",
      "  const double c0=u[3*node],c1=u[3*node+1],c2=u[3*node+2];"]
    for j,(dx,dy,dz) in enumerate(OFFSETS):
        lines += [f"  const int64_t p{j}=(int64_t(wrap(z+({dz}),nz))*ny+wrap(y+({dy}),ny))*nx+wrap(x+({dx}),nx);",
                  f"  const int64_t m{j}=(int64_t(wrap(z-({dz}),nz))*ny+wrap(y-({dy}),ny))*nx+wrap(x-({dx}),nx);"]
        for d in range(3):lines.append(f"  const double q{j}_{d}=(u[3*p{j}+{d}]-c{d})+(u[3*m{j}+{d}]-c{d});")
    groups=[([9],9),([6],6),([5],5),([2,11],11),([1,10],10),([4,7],7),([0,3,8,12],12)]
    for d in range(3):
        terms=[f"S[{9*j+3*d+d}]*({'+'.join(f'q{k}_{d}' for k in group)})" for group,j in groups]
        lines.append(f"  double f{d}="+"+".join(terms)+";")
    for a,b,minus,plus,signs in ((0,1,2,11,[1,-1,-1,1]),(0,2,1,10,[-1,-1,1,1]),(1,2,4,7,[-1,1,-1,1])):
        for out,input_ in ((a,b),(b,a)):
            corner="".join(("+" if sign>0 else "-")+f"q{k}_{input_}" for k,sign in zip((0,3,8,12),signs))
            lines.append(f"  f{out}+=S[{12*9+3*a+b}]*(4.*(q{plus}_{input_}-q{minus}_{input_}){corner});")
    lines += ["  out[3*node]=f0;out[3*node+1]=f1;out[3*node+2]=f2;","}",""]
    (HERE/"background_grouped.cuh").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":main()
