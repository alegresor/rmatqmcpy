import torch 
import agsutil
from jackpy.jack import ZonalPol
import sympy
import numpy as np
import qmcpy

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

class KernelMaternChordal(object):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,8,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x = genflag_x_qr_iid(4,lam,seed=7)
        >>> x.shape 
        torch.Size([4, 7, 7])

        >>> kernel = KernelMaternChordal(lam,nu=1/2)
        >>> k = kernel(x[:,None,:,:],x[None,:,:,:])
        >>> k.shape 
        torch.Size([4, 4])
        >>> k 
        tensor([[1.0000, 0.5492, 0.6187, 0.5031],
                [0.5492, 1.0000, 0.5205, 0.4770],
                [0.6187, 0.5205, 1.0000, 0.4945],
                [0.5031, 0.4770, 0.4945, 1.0000]])

        >>> for nu in [1/2,3/2,5/2,7/2,9/2,11/2,13/2,np.inf]:
        ...     kernel = KernelMaternChordal(lam,nu=1/2)
        ...     k = kernel(x[:,None,:,:],x[None,:,:,:])
        ...     assert k.shape==(4,4)
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
            self.c = torch.tensor([1],dtype=self.lam.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 1/2$"
        elif self.nu==3/2:
            self.c = torch.tensor([1,np.sqrt(3)],dtype=self.lam.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 3/2$"
        elif self.nu==5/2:
            self.c = torch.tensor([1,np.sqrt(5),5/3],dtype=self.lam.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 5/2$"
        elif self.nu==7/2:
            self.c = torch.tensor([1,np.sqrt(7),14/5,7*np.sqrt(7)/15],dtype=self.lam.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 7/2$"
        elif self.nu==9/2:
            self.c = torch.tensor([1,3,27/7,18/7,27/35],dtype=self.lam.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 9/2$"
        elif self.nu==11/2:
            self.c = torch.tensor([1,np.sqrt(11),44/9,11*np.sqrt(11)/9,121/63,121*np.sqrt(11)/945],dtype=self.lam.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 11/2$"
        elif self.nu==13/2:
            self.c = torch.tensor([1,np.sqrt(13),65/11,52*np.sqrt(13)/33,338/99,169*np.sqrt(13)/495,2197/10395],dtype=self.lam.dtype,device=self.lam.device)
            self.nu_str = r"$\nu = 13/2$"
        elif self.nu==np.inf:
            self.nu_str = r"$\nu = \infty$"
    def chordal_dist(self, x, y):
        return torch.linalg.norm(x-y,dim=(-2,-1))
    def __call__(self, x, y):
        r = self.chordal_dist(x,y) 
        rscaled = r/self.rho
        if self.nu==np.inf:
            k = torch.exp(-(rscaled)**2/2)
        else:
            exp_term = torch.exp(-np.sqrt(2*self.nu)*rscaled) 
            poly_term = (self.c*rscaled[...,None]**torch.arange(len(self.c),dtype=self.lam.dtype)).sum(-1)
            k = exp_term*poly_term
        return self.sigma2*k

def tff_qr(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tff_qr(torch.rand((2,N**2),generator=rng))
        >>> q.shape
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.6554,  0.0144,  0.7552],
                 [ 0.4522,  0.8083,  0.3771],
                 [-0.6049,  0.5886, -0.5363]],
        <BLANKLINE>
                [[ 0.4473, -0.3480, -0.8239],
                 [-0.6125, -0.7904,  0.0013],
                 [ 0.6517, -0.5041,  0.5667]]])
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

def tff_qr_complex(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tff_qr_complex(torch.rand((2,2*N**2),generator=rng))
        >>> q.shape
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.2586-0.2661j,  0.8447-0.1177j,  0.3613+0.0667j],
                 [-0.2387+0.0676j,  0.2890+0.2913j, -0.5804-0.6582j],
                 [-0.4265-0.7867j, -0.2891-0.1438j, -0.2795+0.1297j]],
        <BLANKLINE>
                [[-0.3620+0.7348j,  0.3715+0.0140j,  0.3549-0.2545j],
                 [ 0.3698+0.1818j,  0.4820+0.5669j, -0.1691+0.4979j],
                 [ 0.0229-0.3983j,  0.3680+0.4155j, -0.0743-0.7261j]]])
    """ 
    N = int(np.round(np.sqrt(u.size(-1)/2)))
    assert u.size(-1)==(2*N**2)
    assert (0<=u).all()
    assert (u<=1).all()
    u = u.reshape((*u.shape[:-1],N,N,2))
    v = agsutil.icdf_std_normal(u)
    v_complex = torch.complex(v[...,0],v[...,1])
    q,r = torch.linalg.qr(v_complex)
    d = torch.diagonal(r,dim1=-2,dim2=-1)
    ph = d/torch.abs(d)
    q = q*ph[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...jk->...ik",q,r*ph[..., :, None].conj()),v_complex)
    return q

def tff_svd_quaternionic(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tff_svd_quaternionic(torch.rand((2,4*N**2),generator=rng))
        >>> q.shape
        torch.Size([2, 6, 6])
        >>> q
        tensor([[[-0.3855-0.0066j,  0.2846+0.1340j,  0.0531+0.4453j, -0.0713-0.0289j,
                  -0.0862-0.1470j, -0.5170-0.4990j],
                 [-0.2564-0.3464j, -0.4935+0.2373j,  0.3129-0.0210j, -0.1699+0.2938j,
                  -0.3639+0.4087j, -0.0355-0.0141j],
                 [ 0.4523+0.2013j, -0.3856+0.2435j,  0.2998+0.1332j,  0.5145+0.1862j,
                   0.0854-0.2397j, -0.1482-0.2306j],
                 [ 0.0713-0.0289j,  0.0862-0.1470j,  0.5170-0.4990j, -0.3855+0.0066j,
                   0.2846-0.1340j,  0.0531-0.4453j],
                 [ 0.1699+0.2938j,  0.3639+0.4087j,  0.0355-0.0141j, -0.2564+0.3464j,
                  -0.4935-0.2373j,  0.3129+0.0210j],
                 [-0.5145+0.1862j, -0.0854-0.2397j,  0.1482-0.2306j,  0.4523-0.2013j,
                  -0.3856-0.2435j,  0.2998-0.1332j]],
        <BLANKLINE>
                [[-0.4396-0.3859j, -0.1858+0.4638j, -0.1706-0.4717j,  0.1578-0.1175j,
                  -0.1740+0.0735j,  0.2780-0.0707j],
                 [ 0.0092+0.2218j, -0.1126+0.4596j,  0.3860+0.1825j, -0.4997-0.0507j,
                   0.0025+0.3514j,  0.2753+0.3050j],
                 [ 0.3743+0.2976j,  0.2388+0.3247j,  0.0849-0.3571j,  0.2850-0.0874j,
                  -0.4425+0.0953j, -0.3876+0.1736j],
                 [-0.1578-0.1175j,  0.1740+0.0735j, -0.2780-0.0707j, -0.4396+0.3859j,
                  -0.1858-0.4638j, -0.1706+0.4717j],
                 [ 0.4997-0.0507j, -0.0025+0.3514j, -0.2753+0.3050j,  0.0092-0.2218j,
                  -0.1126-0.4596j,  0.3860-0.1825j],
                 [-0.2850-0.0874j,  0.4425+0.0953j,  0.3876+0.1736j,  0.3743-0.2976j,
                   0.2388-0.3247j,  0.0849+0.3571j]]])
    """ 
    N = int(np.round(np.sqrt(u.size(-1)/4)))
    assert u.size(-1)==(4*N**2)
    assert (0<=u).all()
    assert (u<=1).all()
    u = u.reshape((*u.shape[:-1],N,N,4))
    v = agsutil.icdf_std_normal(u)
    B = torch.complex(v[..., 0],v[..., 1])
    C = torch.complex(v[..., 2],v[..., 3])
    v_symplectic = torch.cat([
        torch.cat([B,              C            ], dim=-1),
        torch.cat([-torch.conj(C), torch.conj(B)], dim=-1)], dim=-2)
    U,_,Vh = torch.linalg.svd(v_symplectic)
    q = torch.einsum("...ij,...jk->...ik",U,Vh)
    # assert torch.allclose(torch.einsum("...ij,...kj->...ik",q,torch.conj(q)),torch.eye(2*N,dtype=q.dtype).expand_as(q))
    return q

def tff_eig(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,5,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tff_eig(torch.rand((2,N*(N+1)//2+2*N),generator=rng))
        >>> q.shape
        torch.Size([2, 4, 4])
        >>> q
        tensor([[[-0.2925, -0.6609, -0.6009,  0.3413],
                 [-0.6245,  0.6756, -0.3757,  0.1115],
                 [ 0.4832,  0.2735, -0.0652,  0.8291],
                 [-0.5394, -0.1787,  0.7025,  0.4285]],
        <BLANKLINE>
                [[ 0.0554,  0.2047,  0.9616,  0.1740],
                 [ 0.7688,  0.4471, -0.0574, -0.4536],
                 [ 0.4338, -0.8694,  0.1864, -0.1453],
                 [ 0.4666,  0.0474, -0.1929,  0.8619]]])
    """
    N = int(np.round((-5+np.sqrt(25+8*u.size(-1)))/2))
    assert u.size(-1)==(N*(N+1)//2+2*N)
    assert (0<=u).all()
    assert (u<=1).all()
    num_tril = N*(N-1)//2
    alpha = agsutil.icdf_std_normal(u[...,:N])
    beta = agsutil.icdf_std_normal(u[...,N:N+num_tril])/np.sqrt(2)
    signs = torch.where(u[...,N+num_tril:N+num_tril+N]>1/2,1.,-1.).to(u.device)
    perms = u[...,N+num_tril+N:].argsort(-1)
    il0,il1 = torch.tril_indices(N,N,offset=-1,device=u.device)
    v = torch.eye(N,device=u.device)*alpha[...,None]
    v[...,il0,il1] = beta
    v += v.tril(-1).transpose(dim0=-2,dim1=-1)
    # assert torch.allclose(v[...,torch.arange(N,device=u.device),torch.arange(N,device=u.device)],alpha)
    gamma,q = torch.linalg.eigh(v)
    q = q*signs[...,None,:] # this is required because torch.eigh pins the sign to make the largest entry of each eigenvector positive
    # assert torch.allclose(torch.einsum("...ij,...kj->...ik",q,q),torch.eye(N,device=u.device))
    # assert torch.allclose(torch.einsum("...ij,...j,...kj->...ik",q,gamma,q),v)
    q = torch.gather(q,dim=-1,index=perms[...,None,:].expand_as(q))
    return q

def genflag_q_qr_iid(n, N, seed=None, device="cpu"):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = genflag_q_qr_iid(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.6554,  0.0144,  0.7552],
                 [ 0.4522,  0.8083,  0.3771],
                 [-0.6049,  0.5886, -0.5363]],
        <BLANKLINE>
                [[ 0.4473, -0.3480, -0.8239],
                 [-0.6125, -0.7904,  0.0013],
                 [ 0.6517, -0.5041,  0.5667]]])
    """
    rng = agsutil.get_torch_rng(seed,device=device)
    u = torch.rand((n,N**2),generator=rng,device=device)
    q = tff_qr(u)
    return q

def genflag_q_qr_ld(n, N, seed=None, device="cpu"):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = genflag_q_qr_ld(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.1323, -0.4299,  0.8931],
                 [-0.6249,  0.7356,  0.2615],
                 [-0.7694, -0.5236, -0.3660]],
        <BLANKLINE>
                [[ 0.8600, -0.4448, -0.2502],
                 [ 0.4598,  0.8880,  0.0014],
                 [ 0.2215, -0.1162,  0.9682]]])
    """
    rng = agsutil.get_torch_rng(seed,device=device)
    u = torch.from_numpy(qmcpy.DigitalNetB2(N**2,seed=seed,randomize="LMS_DS",order="GRAY")(n)).to(device)
    q = tff_qr(u)
    return q

def genflag_q_eig_iid(n, N, seed=None, device="cpu"):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = genflag_q_eig_iid(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.4010,  0.2775, -0.8731],
                 [-0.3006, -0.9401, -0.1607],
                 [-0.8654,  0.1980,  0.4604]],
        <BLANKLINE>
                [[-0.6765,  0.5182, -0.5232],
                 [ 0.1492,  0.7922,  0.5917],
                 [ 0.7211,  0.3223, -0.6133]]])
    """
    rng = agsutil.get_torch_rng(seed,device=device)
    u = torch.rand((n,N*(N+1)//2+2*N),generator=rng,device=device)
    q = tff_eig(u)
    return q

def genflag_q_eig_ld(n, N, seed=None, device="cpu"):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = genflag_q_eig_ld(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[ 0.2150, -0.7427,  0.6341],
                 [-0.9324,  0.0370,  0.3595],
                 [ 0.2905,  0.6685,  0.6846]],
        <BLANKLINE>
                [[ 0.9117, -0.0349,  0.4094],
                 [ 0.2656,  0.8102, -0.5226],
                 [ 0.3134, -0.5851, -0.7479]]])
    """
    u = torch.from_numpy(qmcpy.DigitalNetB2(N*(N+1)//2+2*N,seed=seed,randomize="LMS_DS",order="GRAY")(n)).to(device)
    q = tff_eig(u)
    return q

def genflag_q_qr_iid_equal_w(n, N, seed=None, device="cpu"):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q,w = genflag_q_qr_iid_equal_w(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.6554,  0.0144,  0.7552],
                 [ 0.4522,  0.8083,  0.3771],
                 [-0.6049,  0.5886, -0.5363]],
        <BLANKLINE>
                [[ 0.4473, -0.3480, -0.8239],
                 [-0.6125, -0.7904,  0.0013],
                 [ 0.6517, -0.5041,  0.5667]]])
        >>> w.shape 
        torch.Size([2])
        >>> w
        tensor([0.5000, 0.5000])
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,q),torch.eye(3))
        True
    """
    q = genflag_q_qr_iid(n,N,seed,device)
    w = torch.ones(n,device=device)/n
    return q,w

def genflag_q_qr_ld_equal_w(n, N, seed=None, device="cpu"):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q,w = genflag_q_qr_ld_equal_w(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.1323, -0.4299,  0.8931],
                 [-0.6249,  0.7356,  0.2615],
                 [-0.7694, -0.5236, -0.3660]],
        <BLANKLINE>
                [[ 0.8600, -0.4448, -0.2502],
                 [ 0.4598,  0.8880,  0.0014],
                 [ 0.2215, -0.1162,  0.9682]]])
        >>> w.shape 
        torch.Size([2])
        >>> w
        tensor([0.5000, 0.5000])
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,q),torch.eye(3))
        True
    """
    q = genflag_q_qr_ld(n,N,seed,device)
    w = torch.ones(n,device=device)/n
    return q,w

def genflag_q_eig_iid_equal_w(n, N, seed=None, device="cpu"):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q,w = genflag_q_eig_iid_equal_w(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.4010,  0.2775, -0.8731],
                 [-0.3006, -0.9401, -0.1607],
                 [-0.8654,  0.1980,  0.4604]],
        <BLANKLINE>
                [[-0.6765,  0.5182, -0.5232],
                 [ 0.1492,  0.7922,  0.5917],
                 [ 0.7211,  0.3223, -0.6133]]])
        >>> w.shape 
        torch.Size([2])
        >>> w
        tensor([0.5000, 0.5000])
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,q),torch.eye(3))
        True
    """
    q = genflag_q_eig_iid(n,N,seed,device)
    w = torch.ones(n,device=device)/n
    return q,w

def genflag_q_eig_ld_equal_w(n, N, seed=None, device="cpu"):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q,w = genflag_q_eig_ld_equal_w(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[ 0.2150, -0.7427,  0.6341],
                 [-0.9324,  0.0370,  0.3595],
                 [ 0.2905,  0.6685,  0.6846]],
        <BLANKLINE>
                [[ 0.9117, -0.0349,  0.4094],
                 [ 0.2656,  0.8102, -0.5226],
                 [ 0.3134, -0.5851, -0.7479]]])
        >>> w.shape 
        torch.Size([2])
        >>> w
        tensor([0.5000, 0.5000])
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,q),torch.eye(3))
        True
    """
    q = genflag_q_eig_ld(n,N,seed,device)
    w = torch.ones(n,device=device)/n
    return q,w

def genflag_x_qr_iid(n, lam, seed=None):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x = genflag_x_qr_iid(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.5721,  0.1553, -0.2142],
                 [ 0.1553,  0.5179,  0.0191],
                 [-0.2142,  0.0191,  0.5136]],
        <BLANKLINE>
                [[ 0.6625,  0.0730, -0.2027],
                 [ 0.0730,  0.4342,  0.1069],
                 [-0.2027,  0.1069,  0.5069]]])
    """
    q = genflag_q_qr_iid(n,lam.size(-1),seed,lam.device)
    x = torch.einsum("...ij,...j,...kj->...ik",q,lam,q)
    return x

def genflag_x_qr_ld(n, lam, seed=None):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x = genflag_x_qr_ld(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.7430,  0.0403, -0.1146],
                 [ 0.0403,  0.4484, -0.1541],
                 [-0.1146, -0.1541,  0.4121]],
        <BLANKLINE>
                [[ 0.3536, -0.1058, -0.1156],
                 [-0.1058,  0.4780, -0.0269],
                 [-0.1156, -0.0269,  0.7719]]])
    """
    q = genflag_q_qr_ld(n,lam.size(-1),seed,lam.device)
    x = torch.einsum("...ij,...j,...kj->...ik",q,lam,q)
    return x

def genflag_x_eig_iid(n, lam, seed=None):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x = genflag_x_eig_iid(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.6953,  0.0053, -0.2002],
                 [ 0.0053,  0.5173, -0.0893],
                 [-0.2002, -0.0893,  0.3910]],
        <BLANKLINE>
                [[ 0.4854, -0.0558,  0.2161],
                 [-0.0558,  0.6222, -0.1257],
                 [ 0.2161, -0.1257,  0.4961]]])
    """
    q = genflag_q_eig_iid(n,lam.size(-1),seed,lam.device)
    x = torch.einsum("...ij,...j,...kj->...ik",q,lam,q)
    return x

def genflag_x_eig_ld(n, lam, seed=None):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x = genflag_x_eig_ld(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.6296,  0.1145,  0.0993],
                 [ 0.1145,  0.3367,  0.1382],
                 [ 0.0993,  0.1382,  0.6372]],
        <BLANKLINE>
                [[ 0.3572, -0.1219, -0.1582],
                 [-0.1219,  0.5886,  0.0822],
                 [-0.1582,  0.0822,  0.6578]]])
    """
    q = genflag_q_eig_ld(n,lam.size(-1),seed,lam.device)
    x = torch.einsum("...ij,...j,...kj->...ik",q,lam,q)
    return x

def genflag_x_qr_iid_equal_w(n, lam, seed=None):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x,w = genflag_x_qr_iid_equal_w(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.5721,  0.1553, -0.2142],
                 [ 0.1553,  0.5179,  0.0191],
                 [-0.2142,  0.0191,  0.5136]],
        <BLANKLINE>
                [[ 0.6625,  0.0730, -0.2027],
                 [ 0.0730,  0.4342,  0.1069],
                 [-0.2027,  0.1069,  0.5069]]])
        >>> w.shape 
        torch.Size([2])
        >>> w
        tensor([0.5000, 0.5000])
    """
    x = genflag_x_qr_iid(n,lam,seed)
    w = torch.ones(n,device=lam.device)/n
    return x,w

def genflag_x_qr_ld_equal_w(n, lam, seed=None):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x,w = genflag_x_qr_ld_equal_w(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.7430,  0.0403, -0.1146],
                 [ 0.0403,  0.4484, -0.1541],
                 [-0.1146, -0.1541,  0.4121]],
        <BLANKLINE>
                [[ 0.3536, -0.1058, -0.1156],
                 [-0.1058,  0.4780, -0.0269],
                 [-0.1156, -0.0269,  0.7719]]])
        >>> w.shape 
        torch.Size([2])
        >>> w
        tensor([0.5000, 0.5000])
    """
    x = genflag_x_qr_ld(n,lam,seed)
    w = torch.ones(n,device=lam.device)/n
    return x,w

def genflag_x_eig_iid_equal_w(n, lam, seed=None):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x,w = genflag_x_eig_iid_equal_w(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.6953,  0.0053, -0.2002],
                 [ 0.0053,  0.5173, -0.0893],
                 [-0.2002, -0.0893,  0.3910]],
        <BLANKLINE>
                [[ 0.4854, -0.0558,  0.2161],
                 [-0.0558,  0.6222, -0.1257],
                 [ 0.2161, -0.1257,  0.4961]]])
        >>> w.shape 
        torch.Size([2])
        >>> w
        tensor([0.5000, 0.5000])
    """
    x = genflag_x_eig_iid(n,lam,seed)
    w = torch.ones(n,device=lam.device)/n
    return x,w

def genflag_x_eig_ld_equal_w(n, lam, seed=None):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x,w = genflag_x_eig_ld_equal_w(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.6296,  0.1145,  0.0993],
                 [ 0.1145,  0.3367,  0.1382],
                 [ 0.0993,  0.1382,  0.6372]],
        <BLANKLINE>
                [[ 0.3572, -0.1219, -0.1582],
                 [-0.1219,  0.5886,  0.0822],
                 [-0.1582,  0.0822,  0.6578]]])
        >>> w.shape 
        torch.Size([2])
        >>> w
        tensor([0.5000, 0.5000])
    """
    x = genflag_x_eig_ld(n,lam,seed)
    w = torch.ones(n,device=lam.device)/n
    return x,w
