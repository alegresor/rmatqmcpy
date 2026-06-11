import agsutil
import torch 
import numpy as np

def tf_coe_qr(u):
    r"""
    Transform uniform samples to random orthogonal matrices (COE) using QR decomposition.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., n**2)`.

    Returns:
        x (torch.Tensor): A batch of random orthogonal matrices of shape `(..., n, n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_coe_qr(torch.rand((2,n**2),generator=rng))
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
    n = int(np.round(np.sqrt(u.size(-1))))
    assert u.size(-1)==(n**2)
    assert (0<=u).all()
    assert (u<=1).all()
    u = u.reshape((*u.shape[:-1],n,n))
    v = agsutil.icdf_std_normal(u)
    q,r = torch.linalg.qr(v)
    d = torch.diagonal(r,dim1=-2,dim2=-1).sign()
    q = q*d[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...jk->...ik",q,r*d[...,:,None]),v)
    return q

def tf_cue_qr(u):
    r"""
    Transform uniform samples to random unitary matrices (CUE) using QR decomposition.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., 2*n**2)`.

    Returns:
        x (torch.Tensor): A batch of random unitary matrices of shape `(..., n, n)`.
        
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_cue_qr(torch.rand((2,2*n**2),generator=rng))
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
    n = int(np.round(np.sqrt(u.size(-1)/2)))
    assert u.size(-1)==(2*n**2)
    assert (0<=u).all()
    assert (u<=1).all()
    u = u.reshape((*u.shape[:-1],n,n,2))
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
    Transform uniform samples to random orthogonal matrices (COE) using eigenvalue decomposition.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., n*(n+1)//2 + 2*n)`.

    Returns:
        x (torch.Tensor): A batch of random orthogonal matrices of shape `(..., n, n)`.
        
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_coe_eig(torch.rand((2,n*(n+1)//2+2*n),generator=rng))
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
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,q),torch.eye(n))
        True
    """
    n = int(np.round((-5+np.sqrt(25+8*u.size(-1)))/2))
    assert u.size(-1)==(n*(n+1)//2+2*n)
    assert (0<=u).all()
    assert (u<=1).all()
    num_tril = n*(n-1)//2
    alpha = agsutil.icdf_std_normal(u[...,:n])
    beta = agsutil.icdf_std_normal(u[...,n:n+num_tril])/np.sqrt(2)
    signs = torch.where(u[...,n+num_tril:n+num_tril+n]>1/2,1.,-1.).to(u.device)
    perms = u[...,n+num_tril+n:].argsort(-1)
    il0,il1 = torch.tril_indices(n,n,offset=-1,device=u.device)
    v = torch.eye(n,device=u.device)*alpha[...,None]
    v[...,il0,il1] = beta
    v += v.tril(-1).transpose(dim0=-2,dim1=-1)
    # assert torch.allclose(v[...,torch.arange(n,device=u.device),torch.arange(n,device=u.device)],alpha)
    gamma,q = torch.linalg.eigh(v)
    q = q*signs[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...j,...kj->...ik",q,gamma,q),v)
    q = torch.gather(q,dim=-1,index=perms[...,None,:].expand_as(q))
    return q

def tf_cue_eig(u):
    r"""
    Transform uniform samples to random unitary matrices (CUE) using eigenvalue decomposition.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., n**2 + 2*n)`.

    Returns:
        x (torch.Tensor): A batch of random unitary matrices of shape `(..., n, n)`.
        
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_cue_eig(torch.rand((2,n**2+2*n),generator=rng))
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
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,q.conj()),torch.eye(n,dtype=q.dtype))
        True
    """
    n = int(np.round((-1+np.sqrt(1+u.size(-1)))))
    assert u.size(-1)==(n**2+2*n)
    assert (0<=u).all()
    assert (u<=1).all()
    num_tril = n*(n-1)//2
    alpha = agsutil.icdf_std_normal(u[...,:n])
    beta = torch.complex(
        agsutil.icdf_std_normal(u[...,n:n+num_tril])/np.sqrt(2),
        agsutil.icdf_std_normal(u[...,n+num_tril:n+2*num_tril])/np.sqrt(2))
    theta = 2*np.pi*u[...,n+2*num_tril:n+2*num_tril+n]
    perms = u[...,n+2*num_tril+n:].argsort(-1)
    il0,il1 = torch.tril_indices(n,n,offset=-1,device=u.device)
    v = torch.eye(n,device=u.device,dtype=beta.dtype)*alpha[...,None].to(beta.dtype)
    v[...,il0,il1] = beta
    v += v.tril(-1).conj().transpose(dim0=-2,dim1=-1)
    # assert torch.allclose(v[...,torch.arange(n,device=u.device),torch.arange(n,device=u.device)],alpha.to(v.dtype))
    gamma,q = torch.linalg.eigh(v)
    phases = torch.complex(torch.cos(theta),torch.sin(theta))
    q = q*phases[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...j,...kj->...ik",q,gamma,q.conj()),v)
    q = torch.gather(q,dim=-1,index=perms[...,None,:].expand_as(q))
    return q

def tf_cqe_svd(u):
    r"""
    Transform uniform samples to random symplectic unitary matrices (CQE) using SVD.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., 4*n**2)`.

    Returns:
        x (torch.Tensor): A batch of random symplectic unitary matrices of shape `(..., 2n, 2n)`.
        
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_cqe_svd(torch.rand((2,4*n**2),generator=rng))
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
        >>> torch.allclose(torch.einsum("...ij,...kj->...ik",q,torch.conj(q)),torch.eye(2*n,dtype=q.dtype).expand_as(q))
        True
    """ 
    n = int(np.round(np.sqrt(u.size(-1)/4)))
    assert u.size(-1)==(4*n**2)
    assert (0<=u).all()
    assert (u<=1).all()
    u = u.reshape((*u.shape[:-1],n,n,4))
    v = agsutil.icdf_std_normal(u)
    B = torch.complex(v[..., 0],v[..., 1])
    C = torch.complex(v[..., 2],v[..., 3])
    v_symplectic = torch.cat([
        torch.cat([B,              C            ], dim=-1),
        torch.cat([-torch.conj(C), torch.conj(B)], dim=-1)], dim=-2)
    U,_,Vh = torch.linalg.svd(v_symplectic)
    q = torch.einsum("...ij,...jk->...ik",U,Vh)
    return q