import agsutil
import torch 
import numpy as np

from .rand import (
    rand_flag_real,
    rand_flag_complex,
    tf_coe_qr,
)

def compute_tau_int_traceXY_topow_r(lam, r):
    r""" 
    Compute the expected value of the trace of the product of two random flag matrices raised to the power `r`.

    Args:
        lam (torch.Tensor): Eigenvalues of the flag matrices.
        r (int): The power to which the trace is raised.

    Returns:
        torch.Tensor: A tensor of shape `(r+1,)` containing the expected values for powers `0` to `r`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> tau = compute_tau_int_traceXY_topow_r(lam,2)
        >>> tau 
        tensor([1.0000, 0.8571, 0.7388])

        >>> n = lam.size(-1) 
        >>> T1 = lam.sum()**2
        >>> T2 = (lam**2).sum()
        >>> tauref = torch.tensor([1,T1/n,1/((n-1)*n*(n+2))*((n+1)*T1**2+2*n*T2**2-4*T1*T2)]) 
        >>> tauref
        tensor([1.0000, 0.8571, 0.7388])
        >>> torch.allclose(tau,tauref)
        True

        >>> compute_tau_int_traceXY_topow_r(lam,3)
        tensor([1.0000, 0.8571, 0.7388, 0.6402])
    """
    import sympy
    try:
        from jackpy.jack import ZonalPol
    except ImportError:
        raise ImportError("jackpy not found, try pip install jackpolynomials or pip install --ignore-requires-python jackpolynomials")
    n = lam.size(-1)
    assert lam.shape==(n,)
    device = lam.device
    tau = torch.zeros(r+1,device=device)
    tau[0] = 1
    lamlist = lam.cpu().tolist()
    onesN = [1.]*n
    for t in range(1,r+1):
        for m in agsutil.enumerate_partitions(t):
            if len(m)>n: continue
            mpart = sympy.combinatorics.partitions.IntegerPartition(m)
            zp = ZonalPol(n,mpart)
            exprtm = (zp(*lamlist)**2)/zp(*onesN)
            tau[t] += float(exprtm)
    return tau

class KernelPolyFlag(object):
    r"""
    Polynomial kernel for flag matrices.

    Args:
        lam (torch.Tensor): Eigenvalues of the flag matrices.
        r (int): Degree of the polynomial.
        c (float): Constant offset.

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
        >>> n = lam.size(-1)
        >>> qx = tf_coe_qr(torch.rand((2**3,n**2),generator=rng))
        >>> x = torch.einsum("...ij,...j,...kj->...ik",qx,lam,qx)
        >>> qy = tf_coe_qr(torch.rand((2**18,n**2),generator=rng))
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

class KernelMaternChordal(object):
    r"""
    Matern chordal kernel for flag matrices.

    Args:
        lam (torch.Tensor): Eigenvalues of the flag matrices.
        nu (float): Smoothness parameter. Must be in `[1/2, 3/2, 5/2, 7/2, 9/2, 11/2, 13/2, np.inf]`.
        sigma2 (float, optional): Variance parameter.
        rho (float, optional): Length-scale parameter.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,8,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> kernel = KernelMaternChordal(lam,nu=1/2)

        >>> x = rand_flag_real(3,lam,seed=7)
        >>> x.shape 
        torch.Size([3, 7, 7])
        >>> k = kernel(x[:,None,:,:],x[None,:,:,:])
        >>> k.shape 
        torch.Size([3, 3])
        >>> k
        tensor([[1.0000, 0.5458, 0.5388],
                [0.5458, 1.0000, 0.5787],
                [0.5388, 0.5787, 1.0000]])
        >>> for nu in [1/2,3/2,5/2,7/2,9/2,11/2,13/2,np.inf]:
        ...     kernel = KernelMaternChordal(lam,nu=1/2)
        ...     k = kernel(x[:,None,:,:],x[None,:,:,:])
        ...     assert k.shape==(3,3)
        ...     assert not k.isnan().any()
        ...     assert k.isfinite().all()
        
        >>> x = rand_flag_complex(3,lam,seed=7)
        >>> x.shape 
        torch.Size([3, 7, 7])
        >>> k = kernel(x[:,None,:,:],x[None,:,:,:])
        >>> k.shape 
        torch.Size([3, 3])
        >>> k
        tensor([[1.0000, 0.5225, 0.5189],
                [0.5225, 1.0000, 0.5716],
                [0.5189, 0.5716, 1.0000]])
        >>> for nu in [1/2,3/2,5/2,7/2,9/2,11/2,13/2,np.inf]:
        ...     kernel = KernelMaternChordal(lam,nu=1/2)
        ...     k = kernel(x[:,None,:,:],x[None,:,:,:])
        ...     assert k.shape==(3,3)
        ...     assert not k.isnan().any()
        ...     assert k.isfinite().all()
    """
    SUPPORTED_NU = [1/2,3/2,5/2,7/2,9/2,11/2,13/2,np.inf]
    def __init__(self, lam, nu, sigma2=1, rho=1):
        self.lam = lam 
        self.nu = nu
        self.sigma2 = sigma2 
        self.rho = rho 
        self.tr_lam2 = (self.lam**2).sum()
        assert self.nu in self.SUPPORTED_NU, "nu should be in %s"%str(self.SUPPORTED_NU)
        if self.nu==1/2: 
            self.c = torch.tensor([1],dtype=self.lam.real.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 1/2$"
        elif self.nu==3/2:
            self.c = torch.tensor([1,np.sqrt(3)],dtype=self.lam.real.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 3/2$"
        elif self.nu==5/2:
            self.c = torch.tensor([1,np.sqrt(5),5/3],dtype=self.lam.real.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 5/2$"
        elif self.nu==7/2:
            self.c = torch.tensor([1,np.sqrt(7),14/5,7*np.sqrt(7)/15],dtype=self.lam.real.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 7/2$"
        elif self.nu==9/2:
            self.c = torch.tensor([1,3,27/7,18/7,27/35],dtype=self.lam.real.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 9/2$"
        elif self.nu==11/2:
            self.c = torch.tensor([1,np.sqrt(11),44/9,11*np.sqrt(11)/9,121/63,121*np.sqrt(11)/945],dtype=self.lam.real.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 11/2$"
        elif self.nu==13/2:
            self.c = torch.tensor([1,np.sqrt(13),65/11,52*np.sqrt(13)/33,338/99,169*np.sqrt(13)/495,2197/10395],dtype=self.lam.real.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 13/2$"
        elif self.nu==np.inf:
            self.nu_str = r"$\nu = \infty$"
    def chordal_dist(self, x, y):
        return torch.linalg.norm(x-y,dim=(-2,-1)).real
    def __call__(self, x, y):
        r = self.chordal_dist(x,y) 
        rscaled = r/self.rho
        if self.nu==np.inf:
            k = torch.exp(-(rscaled)**2/2)
        else:
            exp_term = torch.exp(-np.sqrt(2*self.nu)*rscaled) 
            poly_term = (self.c*rscaled[...,None]**torch.arange(len(self.c),dtype=self.lam.real.dtype)).sum(-1)
            k = exp_term*poly_term
        return self.sigma2*k