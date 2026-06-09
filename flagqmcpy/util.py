import torch 
import agsutil
from jackpy.jack import ZonalPol
import sympy
import numpy as np
import qmcpy as qp

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
        >>> qx = tf_coe_qr(torch.rand((2**3,N**2),generator=rng))
        >>> x = torch.einsum("...ij,...j,...kj->...ik",qx,lam,qx)
        >>> qy = tf_coe_qr(torch.rand((2**18,N**2),generator=rng))
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

def tf_coe_qr(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tf_coe_qr(torch.rand((2,N**2),generator=rng))
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

def tf_cue_qr(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tf_cue_qr(torch.rand((2,2*N**2),generator=rng))
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

def tf_coe_eig(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tf_coe_eig(torch.rand((2,N*(N+1)//2+2*N),generator=rng))
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
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,q),torch.eye(N))
        True
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
    q = q*signs[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...j,...kj->...ik",q,gamma,q),v)
    q = torch.gather(q,dim=-1,index=perms[...,None,:].expand_as(q))
    return q

def tf_cue_eig(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tf_cue_eig(torch.rand((2,N**2+2*N),generator=rng))
        >>> q.shape
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[ 0.4933+0.2062j, -0.3761-0.1423j,  0.0437-0.7420j],
                 [-0.1455-0.7220j, -0.3441+0.1009j,  0.5631-0.1091j],
                 [-0.3802+0.1649j, -0.7717-0.3380j, -0.2524+0.2341j]],
        <BLANKLINE>
                [[-0.0814-0.6795j, -0.2821+0.2191j, -0.6032+0.2004j],
                 [ 0.0860+0.4265j, -0.3979+0.7882j, -0.0729-0.1606j],
                 [-0.1979+0.5507j,  0.1467-0.2669j, -0.7512+0.0238j]]])
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,q.conj()),torch.eye(N,dtype=q.dtype))
        True
    """
    N = int(np.round((-1+np.sqrt(1+u.size(-1)))))
    assert u.size(-1)==(N**2+2*N)
    assert (0<=u).all()
    assert (u<=1).all()
    num_tril = N*(N-1)//2
    alpha = agsutil.icdf_std_normal(u[...,:N])
    beta = torch.complex(
        agsutil.icdf_std_normal(u[...,N:N+num_tril])/np.sqrt(2),
        agsutil.icdf_std_normal(u[...,N+num_tril:N+2*num_tril])/np.sqrt(2))
    theta = 2*np.pi*u[...,N+2*num_tril:N+2*num_tril+N]
    perms = u[...,N+2*num_tril+N:].argsort(-1)
    il0,il1 = torch.tril_indices(N,N,offset=-1,device=u.device)
    v = torch.eye(N,device=u.device,dtype=beta.dtype)*alpha[...,None].to(beta.dtype)
    v[...,il0,il1] = beta
    v += v.tril(-1).conj().transpose(dim0=-2,dim1=-1)
    # assert torch.allclose(v[...,torch.arange(N,device=u.device),torch.arange(N,device=u.device)],alpha.to(v.dtype))
    gamma,q = torch.linalg.eigh(v)
    phases = torch.complex(torch.cos(theta),torch.sin(theta))
    q = q*phases[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...j,...kj->...ik",q,gamma,q.conj()),v)
    q = torch.gather(q,dim=-1,index=perms[...,None,:].expand_as(q))
    return q

def tf_cqe_svd(u):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> N = lam.size(-1)
        >>> q = tf_cqe_svd(torch.rand((2,4*N**2),generator=rng))
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
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,torch.conj(q)),torch.eye(2*N,dtype=q.dtype).expand_as(q))
        True
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
    return q

def rand_coe_qr(n, N, seed=None, device="cpu", qp_unif_gen=qp.IIDStdUniform):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        
        >>> q = rand_coe_qr(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.9577, -0.0582, -0.2820],
                 [ 0.2193,  0.4869, -0.8455],
                 [ 0.1865, -0.8715, -0.4535]],
        <BLANKLINE>
                [[-0.8182, -0.3566, -0.4510],
                 [-0.5690,  0.3904,  0.7237],
                 [ 0.0820, -0.8488,  0.5224]]])
        
        >>> q = rand_coe_qr(2,3,seed=7,qp_unif_gen=qp.Net)
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
    u = torch.from_numpy(qp_unif_gen(dimension=N**2,seed=seed)(n)).to(device)
    q = tf_coe_qr(u)
    return q

def rand_cue_qr(n, N, seed=None, device="cpu", qp_unif_gen=qp.IIDStdUniform):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = rand_cue_qr(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.7005+0.0905j,  0.0129-0.0897j,  0.5259-0.4652j],
                 [ 0.1364-0.4500j,  0.2766-0.6973j, -0.2259-0.4062j],
                 [-0.4493+0.2795j,  0.6275-0.1878j, -0.2937+0.4525j]],
        <BLANKLINE>
                [[ 0.5150+0.6182j, -0.1275-0.0258j, -0.5254-0.2442j],
                 [ 0.1241+0.1499j, -0.6573-0.5715j,  0.3584+0.2739j],
                 [ 0.1027-0.5515j, -0.0820-0.4666j, -0.2046-0.6473j]]])
        
        >>> q = rand_cue_qr(2,3,seed=7,qp_unif_gen=qp.Net)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[ 0.1884+0.1952j,  0.2403+0.7558j,  0.1686-0.5187j],
                 [-0.0775+0.3625j, -0.0203-0.5851j,  0.2945-0.6581j],
                 [-0.8250-0.3293j, -0.0804+0.1475j, -0.2682-0.3329j]],
        <BLANKLINE>
                [[-0.6121-0.3989j, -0.2673-0.1013j, -0.6161-0.0701j],
                 [ 0.4590-0.0060j,  0.0882-0.5607j, -0.4561+0.5091j],
                 [ 0.1663+0.4774j, -0.7245-0.2670j, -0.0730-0.3782j]]])
    """
    rng = agsutil.get_torch_rng(seed,device=device)
    u = torch.from_numpy(qp_unif_gen(dimension=2*N**2,seed=seed)(n)).to(device)
    q = tf_cue_qr(u)
    return q

def rand_coe_eig(n, N, seed=None, device="cpu", qp_unif_gen=qp.IIDStdUniform):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = rand_coe_eig(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.9005, -0.4348, -0.0030],
                 [ 0.2592, -0.5312, -0.8066],
                 [ 0.3491, -0.7271,  0.5911]],
        <BLANKLINE>
                [[ 0.2322,  0.3624, -0.9026],
                 [ 0.9409, -0.3190,  0.1140],
                 [-0.2467, -0.8757, -0.4150]]])
        
        >>> q = rand_coe_eig(2,3,seed=7,qp_unif_gen=qp.Net)
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
    u = torch.from_numpy(qp_unif_gen(dimension=N*(N+1)//2+2*N,seed=seed)(n)).to(device)
    q = tf_coe_eig(u)
    return q

def rand_cue_eig(n, N, seed=None, device="cpu", qp_unif_gen=qp.IIDStdUniform):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = rand_cue_eig(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.7999-0.3030j, -0.1347-0.4416j, -0.0308-0.2330j],
                 [ 0.2180+0.0408j, -0.5586-0.2853j,  0.7457-0.0362j],
                 [ 0.3826-0.2699j, -0.5249-0.3433j, -0.6181+0.0734j]],
        <BLANKLINE>
                [[-0.1474-0.0856j,  0.3710-0.4748j, -0.6285-0.4614j],
                 [ 0.2958+0.8971j,  0.2207-0.2072j,  0.1219-0.0362j],
                 [-0.2346-0.1540j, -0.0932-0.7325j,  0.6111-0.0500j]]])
        
        >>> q = rand_cue_eig(2,3,seed=7,qp_unif_gen=qp.Net)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.4215-0.7588j,  0.4406+0.0016j, -0.2176+0.0711j],
                 [-0.3701+0.2593j,  0.2614+0.6915j,  0.1970-0.4589j],
                 [-0.0046+0.2058j,  0.0371+0.5080j, -0.3877+0.7402j]],
        <BLANKLINE>
                [[ 0.5386-0.4166j, -0.1877+0.1013j, -0.5136-0.4766j],
                 [ 0.4655+0.4044j,  0.6083+0.2491j, -0.3016+0.3109j],
                 [-0.0103+0.3949j, -0.3730-0.6191j, -0.5221+0.2211j]]])
    """
    u = torch.from_numpy(qp_unif_gen(dimension=N**2+2*N,seed=seed)(n)).to(device)
    q = tf_cue_eig(u)
    return q

def rand_cqe_svd(n, N, seed=None, device="cpu", qp_unif_gen=qp.IIDStdUniform):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = rand_cqe_svd(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 6, 6])
        >>> q
        tensor([[[-0.5379+0.0018j,  0.1665-0.3441j,  0.1465-0.4183j, -0.0826+0.2936j,
                   0.0846-0.3585j, -0.3592-0.1015j],
                 [-0.2501+0.3772j, -0.4710+0.0421j,  0.1512+0.1012j,  0.1678-0.1855j,
                   0.3810+0.2695j, -0.1729-0.4776j],
                 [ 0.0476+0.3640j,  0.4540-0.1432j, -0.1454-0.3261j,  0.1149-0.4497j,
                   0.1691-0.1465j,  0.4496-0.2084j],
                 [ 0.0826+0.2936j, -0.0846-0.3585j,  0.3592-0.1015j, -0.5379-0.0018j,
                   0.1665+0.3441j,  0.1465+0.4183j],
                 [-0.1678-0.1855j, -0.3810+0.2695j,  0.1729-0.4776j, -0.2501-0.3772j,
                  -0.4710-0.0421j,  0.1512-0.1012j],
                 [-0.1149-0.4497j, -0.1691-0.1465j, -0.4496-0.2084j,  0.0476-0.3640j,
                   0.4540+0.1432j, -0.1454+0.3261j]],
        <BLANKLINE>
                [[-0.0798-0.5452j, -0.1923+0.2617j,  0.0815+0.2038j, -0.0359-0.4116j,
                   0.2102-0.3235j,  0.4459+0.1562j],
                 [-0.3699+0.0262j, -0.2731-0.5066j, -0.1090-0.0730j, -0.2111-0.4440j,
                  -0.1848-0.1637j, -0.3600+0.2859j],
                 [ 0.1009+0.1924j,  0.1127-0.3529j,  0.5547+0.1064j,  0.2102+0.2346j,
                  -0.2066-0.4166j,  0.2735+0.3259j],
                 [ 0.0359-0.4116j, -0.2102-0.3235j, -0.4459+0.1562j, -0.0798+0.5452j,
                  -0.1923-0.2617j,  0.0815-0.2038j],
                 [ 0.2111-0.4440j,  0.1848-0.1637j,  0.3600+0.2859j, -0.3699-0.0262j,
                  -0.2731+0.5066j, -0.1090+0.0730j],
                 [-0.2102+0.2346j,  0.2066-0.4166j, -0.2735+0.3259j,  0.1009-0.1924j,
                   0.1127+0.3529j,  0.5547-0.1064j]]])
        
        >>> q = rand_cqe_svd(2,3,seed=7,qp_unif_gen=qp.Net)
        >>> q.shape 
        torch.Size([2, 6, 6])
        >>> q
        tensor([[[ 0.0024+0.0842j,  0.6149-0.0634j, -0.3828-0.0823j, -0.2519-0.3091j,
                   0.3628+0.2602j, -0.1177+0.2920j],
                 [ 0.3461-0.6176j,  0.0761+0.1610j, -0.5182+0.0585j,  0.2130+0.0398j,
                  -0.1930-0.0418j,  0.3297+0.0230j],
                 [ 0.0650-0.1395j,  0.2626-0.2246j,  0.3886-0.2880j,  0.4926+0.1394j,
                   0.4424+0.1808j,  0.3377-0.1361j],
                 [ 0.2519-0.3091j, -0.3628+0.2602j,  0.1177+0.2920j,  0.0024-0.0842j,
                   0.6149+0.0634j, -0.3828+0.0823j],
                 [-0.2130+0.0398j,  0.1930-0.0418j, -0.3297+0.0230j,  0.3461+0.6176j,
                   0.0761-0.1610j, -0.5182-0.0585j],
                 [-0.4926+0.1394j, -0.4424+0.1808j, -0.3377-0.1361j,  0.0650+0.1395j,
                   0.2626+0.2246j,  0.3886+0.2880j]],
        <BLANKLINE>
                [[ 0.2988+0.4254j, -0.3412+0.0841j,  0.2553+0.2561j,  0.0461+0.1016j,
                  -0.1127-0.1724j,  0.5693-0.3106j],
                 [-0.1268+0.1878j,  0.2846+0.1790j,  0.2376+0.0494j, -0.3145-0.5206j,
                   0.0981-0.6136j, -0.1342-0.0512j],
                 [ 0.4454-0.0571j, -0.0687+0.2563j, -0.5844+0.0523j, -0.0528+0.3026j,
                   0.2235-0.4632j, -0.1256+0.0951j],
                 [-0.0461+0.1016j,  0.1127-0.1724j, -0.5693-0.3106j,  0.2988-0.4254j,
                  -0.3412-0.0841j,  0.2553-0.2561j],
                 [ 0.3145-0.5206j, -0.0981-0.6136j,  0.1342-0.0512j, -0.1268-0.1878j,
                   0.2846-0.1790j,  0.2376-0.0494j],
                 [ 0.0528+0.3026j, -0.2235-0.4632j,  0.1256+0.0951j,  0.4454+0.0571j,
                  -0.0687-0.2563j, -0.5844-0.0523j]]])
    """
    u = torch.from_numpy(qp_unif_gen(dimension=4*N**2,seed=seed)(n)).to(device)
    q = tf_cqe_svd(u)
    return q

def rand_flag_real(n, lam, seed=None, rand_coe=rand_coe_qr, qp_unif_gen=qp.IIDStdUniform):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x = rand_flag_real(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.3107,  0.1198,  0.0819],
                 [ 0.1198,  0.7127,  0.0915],
                 [ 0.0819,  0.0915,  0.5802]],
        <BLANKLINE>
                [[ 0.4100, -0.2117, -0.0450],
                 [-0.2117,  0.5880,  0.1135],
                 [-0.0450,  0.1135,  0.6057]]])
    """
    assert rand_coe in [rand_coe_qr,rand_coe_eig]
    assert not torch.is_complex(lam)
    q = rand_coe(n=n,N=lam.size(-1),seed=seed,device=lam.device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...j,...kj->...ik",q,lam.to(q.dtype),q)
    return x

def rand_flag_complex(n, lam, seed=None, rand_cue=rand_cue_qr, qp_unif_gen=qp.IIDStdUniform):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x = rand_flag_complex(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.5330-2.6890e-17j,  0.0552+1.6613e-01j, -0.1884-6.8538e-02j],
                 [ 0.0552-1.6613e-01j,  0.5332+1.8188e-18j,  0.0186+1.5354e-02j],
                 [-0.1884+6.8538e-02j,  0.0186-1.5354e-02j,  0.5374+1.4321e-17j]],
        <BLANKLINE>
                [[ 0.4512+1.7336e-17j, -0.1101+1.5204e-02j,  0.1480-1.7041e-01j],
                 [-0.1101-1.5204e-02j,  0.5788-7.8507e-18j, -0.0483+2.4620e-02j],
                 [ 0.1480+1.7041e-01j, -0.0483-2.4620e-02j,  0.5736-3.6136e-18j]]])
    """
    assert rand_cue in [rand_cue_qr,rand_cue_eig]
    assert not torch.is_complex(lam)
    q = rand_cue(n=n,N=lam.size(-1),seed=seed,device=lam.device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...j,...kj->...ik",q,lam.to(q.dtype),q.conj())
    return x

def rand_flag_quaternionic(n, lam, seed=None, rand_cqe=rand_cqe_svd, qp_unif_gen=qp.IIDStdUniform):
    r"""
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> lam = torch.arange(1,4,dtype=float)
        >>> lam = lam/torch.linalg.norm(lam)
        >>> x = rand_flag_quaternionic(2,lam,seed=7)
        >>> x.shape 
        torch.Size([2, 6, 6])
        >>> x
        tensor([[[ 0.1039+2.2207e-18j, -0.0657+2.7406e-01j,  0.2670+1.9196e-01j,
                  -0.2458+3.7671e-02j, -0.1937+5.2998e-03j, -0.0665-1.4676e-01j],
                 [-0.0657-2.7406e-01j, -0.1392+1.2335e-18j, -0.1863+1.6344e-01j,
                  -0.1937+5.2998e-03j, -0.1535-2.0375e-01j,  0.0983+2.4468e-01j],
                 [ 0.2670-1.9196e-01j, -0.1863-1.6344e-01j, -0.0219+5.7349e-18j,
                  -0.0665-1.4676e-01j,  0.0983+2.4468e-01j, -0.0638-2.7260e-01j],
                 [ 0.2458+3.7671e-02j,  0.1937+5.2998e-03j,  0.0665-1.4676e-01j,
                   0.1039+7.0945e-18j, -0.0657-2.7406e-01j,  0.2670-1.9196e-01j],
                 [ 0.1937+5.2998e-03j,  0.1535-2.0375e-01j, -0.0983+2.4468e-01j,
                  -0.0657+2.7406e-01j, -0.1392-5.8086e-18j, -0.1863-1.6344e-01j],
                 [ 0.0665-1.4676e-01j, -0.0983+2.4468e-01j,  0.0638-2.7260e-01j,
                   0.2670+1.9196e-01j, -0.1863+1.6344e-01j, -0.0219+1.4636e-17j]],
        <BLANKLINE>
                [[-0.1280+7.3419e-18j, -0.0233+2.9112e-02j, -0.1971+7.3167e-02j,
                  -0.0638+3.1947e-01j, -0.2302-1.7927e-02j,  0.2307+7.9706e-02j],
                 [-0.0233-2.9112e-02j, -0.0390-1.5836e-18j,  0.0028-2.0747e-01j,
                  -0.2302-1.7927e-02j,  0.1097+2.2488e-01j, -0.3193+1.4914e-01j],
                 [-0.1971-7.3167e-02j,  0.0028+2.0747e-01j,  0.0545-3.4540e-18j,
                   0.2307+7.9706e-02j, -0.3193+1.4914e-01j, -0.0072+3.9862e-01j],
                 [ 0.0638+3.1947e-01j,  0.2302-1.7927e-02j, -0.2307+7.9706e-02j,
                  -0.1280-1.0567e-18j, -0.0233-2.9112e-02j, -0.1971-7.3167e-02j],
                 [ 0.2302-1.7927e-02j, -0.1097+2.2488e-01j,  0.3193+1.4914e-01j,
                  -0.0233+2.9112e-02j, -0.0390-3.4049e-19j,  0.0028+2.0747e-01j],
                 [-0.2307+7.9706e-02j,  0.3193+1.4914e-01j,  0.0072+3.9862e-01j,
                  -0.1971+7.3167e-02j,  0.0028-2.0747e-01j,  0.0545-1.0322e-17j]]])
    """
    assert rand_cqe in [rand_cqe_svd]
    assert not torch.is_complex(lam)
    N = lam.size(-1)
    q = rand_cqe(n=n,N=N,seed=seed,device=lam.device,qp_unif_gen=qp_unif_gen)
    I_nn = torch.diag(torch.cat([torch.ones(N,device=lam.device),-torch.ones(N,device=lam.device)])).to(q.dtype)
    q_H = q.conj().transpose(-2, -1)
    q_left_corner = torch.einsum("ij,...jk,kl->...il",I_nn,q_H,I_nn)
    lam_paired = torch.cat([lam,lam],dim=-1)
    x = torch.einsum("...ij,...j,...jk->...ik",q,lam_paired.to(q.dtype),q_left_corner)
    return x
