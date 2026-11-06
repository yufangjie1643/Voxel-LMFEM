"""Independent laminate stiffness from continuity of in-plane strain and traction.

No FE, modal, FFT or GPU routines are used. Engineering order: 11,22,33,23,13,12.
"""
import numpy as np


def constituent_matrices(material):
    matrix, yarn = material['matrix'], material['yarn']
    E, nu = matrix['E'], matrix['Nu']
    mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
    C0 = np.zeros((6,6)); C0[:3,:3] = lam
    C0[:3,:3] += 2*mu*np.eye(3); C0[3:,3:] = mu*np.eye(3)
    S = np.diag([1/yarn[k] for k in ('E11','E22','E33','G23','G13','G12')])
    for i,j,key,e in [(0,1,'Nu12','E11'),(0,2,'Nu13','E11'),(1,2,'Nu23','E22')]:
        S[i,j] = S[j,i] = -yarn[key]/yarn[e]
    return C0, np.linalg.inv(S)


def rotate(C, R):
    pairs = ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))
    tensor = np.zeros((3,3,3,3))
    for a,(i,j) in enumerate(pairs):
        for b,(k,l) in enumerate(pairs):
            for ii,jj in ((i,j),(j,i)):
                for kk,ll in ((k,l),(l,k)):
                    tensor[ii,jj,kk,ll] = C[a,b]
    rotated = np.einsum('ia,jb,kc,ld,abcd->ijkl',R,R,R,R,tensor)
    return np.array([[rotated[i,j,k,l] for k,l in pairs] for i,j in pairs])


def effective(matrices, fractions):
    # a: common in-plane engineering strains; b: common out-of-plane tractions.
    a, b = [0,1,5], [2,3,4]
    A, B, D = [np.zeros((3,3)) for _ in range(3)]
    for C, f in zip(matrices, fractions):
        inv = np.linalg.inv(C[np.ix_(b,b)])
        Cba = C[np.ix_(b,a)]
        A += f*inv
        B += f*inv@Cba
        D += f*(C[np.ix_(a,a)]-C[np.ix_(a,b)]@inv@Cba)
    H = np.linalg.inv(A)
    out = np.zeros((6,6))
    out[np.ix_(b,b)] = H
    out[np.ix_(b,a)] = H@B
    out[np.ix_(a,b)] = B.T@H
    out[np.ix_(a,a)] = D+B.T@H@B
    return out


def from_record(record):
    meta = record['input']
    C0, C1 = constituent_matrices(meta['material'])
    C1 = rotate(C1,np.array(meta['orientation']))
    f = meta['active_fraction']
    return effective([C0,C1],[1-f,f])
