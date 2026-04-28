import torch 
import agsutil
from jackpy.jack import ZonalPol
import sympy
from sympy.combinatorics.partitions import IntegerPartition
import numpy as np

def compute_tau_int_traceXY_topow_r(lam, r):
    r""" 
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> tau = compute_tau_int_traceXY_topow_r(lam,2)
        >>> tau 
        tensor([1.0000, 0.8571, 0.7388])

        >>> N = lam.size(-1) 
        >>> T1 = lam.sum()**2
        >>> T2 = (lam**2).sum()
        >>> tauref = torch.tensor([1,T1/N,1/((N-1)*N*(N+2))*((N+1)*T1**2+2*N*T2**2-4*T1*T2)]) 
        >>> tauref
        tensor([1.0000, 0.8571, 0.7388])
        >>> torch.allclose(tau,tauref)
        True

        >>> compute_tau_int_traceXY_topow_r(lam,3)
        tensor([1.0000, 0.8571, 0.7388, 0.6402])
    """
    N = lam.size(-1)
    assert lam.shape==(N,)
    device = lam.device
    tau = torch.zeros(r+1,device=device)
    tau[0] = 1
    lamlist = lam.cpu().tolist()
    onesN = [1.]*N
    for t in range(1,r+1):
        for m in agsutil.enumerate_partitions(t):
            if len(m)>N: continue
            mpart = sympy.combinatorics.partitions.IntegerPartition(m)
            zp = ZonalPol(N,mpart)
            exprtm = (zp(*lamlist)**2)/zp(*onesN)
            tau[t] += float(exprtm)
    return tau

class KernelPolyFlag(object):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> r = 3
        >>> c = 0
        >>> lam = torch.arange(1,8,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> kernel = KernelPolyFlag(lam,r,c)
        >>> kernel.tau 
        tensor([1.0000, 0.8000, 0.6415, 0.5156])
        >>> kernel.double_integral
        tensor(0.5156)
        >>> N = lam.size(-1)
        >>> qx = tff_qr(torch.rand((2**3,N**2),generator=rng))
        >>> x = torch.einsum("...ij,...j,...kj->...ik",qx,lam,qx)
        >>> qy = tff_qr(torch.rand((2**18,N**2),generator=rng))
        >>> y = torch.einsum("...ij,...j,...kj->...ik",qy,lam,qy)
        >>> x.shape
        torch.Size([8, 7, 7])
        >>> y.shape
        torch.Size([262144, 7, 7])
        >>> kvals = kernel(x[:,None,:,:],y[None,:,:,:])
        >>> kvals.shape
        torch.Size([8, 262144])
        >>> kmeans = kvals.mean(-1)
        >>> kmeans.shape
        torch.Size([8])
        >>> kmeans
        tensor([1.0005, 1.0000, 1.0005, 1.0001, 0.9998, 1.0002, 1.0003, 1.0002])
        >>> kmeans.mean()-1
        tensor(0.0002)
    """
    def __init__(self, lam, r, c):
        self.lam = lam 
        self.r = r
        self.c = c
        self.tau = compute_tau_int_traceXY_topow_r(self.lam,self.r)
        jvec = torch.arange(self.r+1,device=self.lam.device)
        rt = jvec[-1]
        self.double_integral = (agsutil.comb(rt,jvec)*self.c**(rt-jvec)*self.tau).sum()
    def __call__(self, x, y):
        v = (self.c+(x*y).sum((-2,-1)))**self.r/self.double_integral 
        return v

def tff_qr(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,5,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tff_qr(torch.rand((2,N**2),generator=rng))
        >>> q.shape
        torch.Size([2, 4, 4])
        >>> q
        tensor([[[-0.2996,  0.0109,  0.5127, -0.8045],
                 [ 0.7289, -0.5403,  0.4203, -0.0110],
                 [ 0.3670, -0.0490, -0.7150, -0.5930],
                 [-0.4942, -0.8400, -0.2220,  0.0312]],
        <BLANKLINE>
                [[-0.7117,  0.3685, -0.1934,  0.5659],
                 [ 0.4597,  0.6282, -0.6261, -0.0449],
                 [ 0.4017,  0.4095,  0.6721,  0.4682],
                 [-0.3476,  0.5494,  0.3447, -0.6771]]])
    """ 
    N = int(np.round(np.sqrt(u.size(-1))))
    assert u.size(-1)==(N**2)
    assert (0<=u).all()
    assert (u<=1).all()
    u = u.reshape((*u.shape[:-1],N,N))
    v = agsutil.icdf_std_normal(u)
    q,r = torch.linalg.qr(v)
    d = torch.diagonal(r,dim1=-2,dim2=-1).sign()
    q = q*d[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...jk->...ik",q,r*d[...,:,None]),v)
    return q

def tff_eig(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,5,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tff_eig(torch.rand((2,N*(N+1)//2+N),generator=rng))
        >>> q.shape
        torch.Size([2, 4, 4])
        >>> q
        tensor([[[-0.6609, -0.6009, -0.2925,  0.3413],
                 [ 0.6756, -0.3757, -0.6245,  0.1115],
                 [ 0.2735, -0.0652,  0.4832,  0.8291],
                 [-0.1787,  0.7025, -0.5394,  0.4285]],
        <BLANKLINE>
                [[ 0.5858, -0.7351, -0.2483,  0.2341],
                 [ 0.2611, -0.0613,  0.9497,  0.1614],
                 [-0.7540, -0.6354,  0.1565,  0.0576],
                 [-0.1419,  0.2284, -0.1089,  0.9570]]])
    """ 
    N = int(np.round((-3+np.sqrt(9+8*u.size(-1)))/2))
    assert u.size(-1)==(N*(N+1)//2+N)
    alpha = agsutil.icdf_std_normal(u[...,:N])
    beta = agsutil.icdf_std_normal(u[...,N:(-N)])/np.sqrt(2)
    signs = torch.where(u[...,(-N):]>1/2,1.,-1.).to(u.device)
    il0,il1 = torch.tril_indices(N,N,offset=-1,device=u.device)
    v = torch.eye(N,device=u.device)*alpha[...,None]
    v[...,il0,il1] = beta
    v += v.tril(-1).transpose(dim0=-2,dim1=-1)
    # assert torch.allclose(v[...,torch.arange(N,device=u.device),torch.arange(N,device=u.device)],alpha)
    gamma,q = torch.linalg.eigh(v)
    q = q*signs[...,None,:] # this is required because torch.eigh pins the sign to make the largest entry of each eigenvector positive
    # assert torch.allclose(torch.einsum("...ij,...kj->...ik",q,q),torch.eye(N,device=u.device))
    # assert torch.allclose(torch.einsum("...ij,...j,...kj->...ik",q,gamma,q),v)
    return q
