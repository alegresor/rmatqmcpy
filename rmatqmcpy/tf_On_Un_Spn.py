import agsutil
import torch 
import numpy as np

def tf_On_QR(u, n, k):
    r"""
    Transform uniform samples to random orthogonal matrices using QR decomposition.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., n*k)`.
        n (int): Number of rows.
        k (int): Number of columns with `k<=n`.

    Returns:
        x (torch.Tensor): A batch of random orthogonal matrices of shape `(..., n, k)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_On_QR(torch.rand((2,n**2),generator=rng),n,n)
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
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q),torch.eye(n))
        True
        
        >>> k = 2
        >>> q = tf_On_QR(torch.rand((4,n*k),generator=rng),n,k)
        >>> q.shape
        torch.Size([4, 3, 2])
        >>> q
        tensor([[[-0.6351,  0.7368],
                 [ 0.7422,  0.6652],
                 [ 0.2139, -0.1206]],
        <BLANKLINE>
                [[ 0.5383,  0.2390],
                 [ 0.7023,  0.4028],
                 [-0.4658,  0.8835]],
        <BLANKLINE>
                [[ 0.1407, -0.9139],
                 [ 0.6684, -0.1959],
                 [-0.7303, -0.3554]],
        <BLANKLINE>
                [[-0.7980, -0.4883],
                 [ 0.5758, -0.4442],
                 [-0.1782,  0.7512]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q),torch.eye(k))
        True
    """ 
    assert k<=n
    assert u.size(-1)==(n*k)
    assert (0<=u).all()
    assert (u<=1).all()
    u = u.reshape((*u.shape[:-1],n,k))
    v = agsutil.icdf_std_normal(u)
    q,r = torch.linalg.qr(v,mode='reduced')
    d = torch.diagonal(r,dim1=-2,dim2=-1).sign()
    q = q*d[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...jk->...ik",q,r*d[...,:,None]),v)
    return q

def tf_Un_QR(u, n, k):
    r"""
    Transform uniform samples to random unitary matrices using QR decomposition.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., 2*n*k)`.
        n (int): Number of rows.
        k (int): Number of columns with `k<=n`.

    Returns:
        x (torch.Tensor): A batch of random unitary matrices of shape `(..., n, k)`.
        
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_Un_QR(torch.rand((2,2*n**2),generator=rng),n,n)
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
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q.conj()),torch.eye(n,dtype=q.dtype))
        True

        >>> k = 2
        >>> q = tf_Un_QR(torch.rand((4,2*n*k),generator=rng),n,k)
        >>> q.shape
        torch.Size([4, 3, 2])
        >>> q
        tensor([[[-0.3500-0.6164j, -0.2414-0.1013j],
                 [-0.0782+0.5104j,  0.4290+0.1885j],
                 [-0.3047-0.3715j,  0.8351-0.1207j]],
        <BLANKLINE>
                [[ 0.0614+0.3318j, -0.8324-0.2894j],
                 [-0.3029+0.5080j,  0.3375+0.3046j],
                 [ 0.6301+0.3731j,  0.1075+0.0721j]],
        <BLANKLINE>
                [[ 0.7912+0.2576j,  0.4185+0.2187j],
                 [ 0.3229+0.0629j, -0.7973-0.0245j],
                 [ 0.0592-0.4425j, -0.2767+0.2533j]],
        <BLANKLINE>
                [[-0.2778-0.3864j, -0.2751+0.8253j],
                 [-0.2274+0.0779j,  0.0250-0.1292j],
                 [ 0.7963+0.2856j,  0.1643+0.4459j]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q.conj()),torch.eye(k,dtype=q.dtype))
        True
    """ 
    assert k<=n
    assert u.size(-1)==(2*n*k)
    assert (0<=u).all()
    assert (u<=1).all()
    u = u.reshape((*u.shape[:-1],n,k,2))
    v = agsutil.icdf_std_normal(u)
    v_complex = torch.complex(v[...,0],v[...,1])
    q,r = torch.linalg.qr(v_complex,mode='reduced')
    d = torch.diagonal(r,dim1=-2,dim2=-1)
    ph = torch.sgn(d)
    q = q*ph[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...jk->...ik",q,r*ph[..., :, None].conj()),v_complex)
    return q

def tf_On_eig(u, n, k):
    r"""
    Transform uniform samples to random orthogonal matrices  using eigenvalue decomposition.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., n*(n+1)//2 + 2*n)`.
        n (int): Number of rows.
        k (int): Number of columns with `k<=n`.

    Returns:
        x (torch.Tensor): A batch of random orthogonal matrices of shape `(..., n, k)`.
        
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_On_eig(torch.rand((2,2*n+n*(n-1)//2),generator=rng),n,n)
        >>> q.shape
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.8410,  0.5062, -0.1912],
                 [-0.4034, -0.3511,  0.8450],
                 [ 0.3606,  0.7877,  0.4995]],
        <BLANKLINE>
                [[ 0.5232, -0.5182,  0.6765],
                 [-0.5917, -0.7922, -0.1492],
                 [ 0.6133, -0.3223, -0.7211]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q),torch.eye(n))
        True

        >>> k = 2
        >>> q = tf_On_eig(torch.rand((4,2*n+n*(n-1)//2),generator=rng),n,k)
        >>> q.shape
        torch.Size([4, 3, 2])
        >>> q
        tensor([[[ 0.0680,  0.7201],
                 [-0.5824, -0.5334],
                 [ 0.8101, -0.4439]],
        <BLANKLINE>
                [[ 0.1218, -0.9108],
                 [-0.8979,  0.0683],
                 [-0.4230, -0.4071]],
        <BLANKLINE>
                [[-0.7189,  0.6946],
                 [-0.6655, -0.6771],
                 [-0.2007, -0.2431]],
        <BLANKLINE>
                [[ 0.1882, -0.9249],
                 [-0.2634, -0.3716],
                 [ 0.9461,  0.0805]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q),torch.eye(k))
        True
    """
    assert k<=n
    num_tril = n*(n-1)//2
    assert u.size(-1)==(2*n+num_tril)
    assert (0<=u).all()
    assert (u<=1).all()
    signs = torch.where(u[...,:n]>1/2,1.,-1.).to(u.device)
    alpha = agsutil.icdf_std_normal(u[...,n:2*n])*np.sqrt(2)
    beta = agsutil.icdf_std_normal(u[...,2*n:2*n+num_tril])
    il0,il1 = torch.tril_indices(n,n,offset=-1,device=u.device)
    v = torch.zeros((*u.shape[:-1],n,n),dtype=u.dtype,device=u.device)
    diag_indices = torch.arange(n,device=u.device)
    v[...,diag_indices,diag_indices] = alpha
    v[...,il0,il1] = beta
    v += v.tril(-1).transpose(dim0=-2,dim1=-1)
    assert torch.allclose(v[...,torch.arange(n,device=u.device),torch.arange(n,device=u.device)],alpha)
    gamma,q = torch.linalg.eigh(v)
    q = q*signs[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...j,...kj->...ik",q,gamma,q),v)
    return q[...,:k]

def tf_Un_eig(u, n, k):
    r"""
    Transform uniform samples to random unitary matrices using eigenvalue decomposition.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., n**2 + 2*n)`.
        n (int): Number of rows.
        k (int): Number of columns with `k<=n`.

    Returns:
        x (torch.Tensor): A batch of random unitary matrices of shape `(..., n, k)`.
        
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_Un_eig(torch.rand((2,n**2+n),generator=rng),n,n)
        >>> q.shape
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[ 0.1184-0.6343j, -0.1075+0.7167j,  0.1566-0.1844j],
                 [-0.1909-0.4149j, -0.0457-0.5509j,  0.6462-0.2611j],
                 [ 0.6050+0.0953j,  0.3971-0.1076j, -0.0938-0.6685j]],
        <BLANKLINE>
                [[-0.1771-0.3102j,  0.6653+0.1603j, -0.4573-0.4415j],
                 [-0.7242-0.5052j, -0.4297-0.0679j,  0.1084-0.1391j],
                 [ 0.2435+0.1829j, -0.4473-0.3773j, -0.3695-0.6544j]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q.conj()),torch.eye(n,dtype=q.dtype))
        True

        >>> k = 2
        >>> q = tf_Un_eig(torch.rand((4,n**2+n),generator=rng),n,k)
        >>> q.shape
        torch.Size([4, 3, 2])
        >>> q
        tensor([[[-0.2158+0.1676j, -0.0992-0.8279j],
                 [ 0.5745+0.6733j,  0.4101-0.0443j],
                 [-0.3384+0.1658j,  0.0866-0.3565j]],
        <BLANKLINE>
                [[ 0.2821+0.7138j, -0.4911-0.1984j],
                 [-0.4658+0.4386j, -0.0056+0.6666j],
                 [ 0.0355-0.0141j, -0.2300+0.4714j]],
        <BLANKLINE>
                [[-0.4660-0.2014j, -0.2798+0.6882j],
                 [-0.4888+0.5990j, -0.2285-0.4499j],
                 [ 0.3796+0.0216j,  0.4329+0.0782j]],
        <BLANKLINE>
                [[ 0.2325-0.7290j,  0.1024+0.0821j],
                 [ 0.4156-0.3091j,  0.0759+0.4502j],
                 [ 0.3812+0.0299j,  0.4367-0.7640j]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q.conj()),torch.eye(k,dtype=q.dtype))
        True
    """
    assert k<=n
    assert u.size(-1)==(n**2+n)
    assert (0<=u).all()
    assert (u<=1).all()
    num_tril = n*(n-1)//2
    alpha = agsutil.icdf_std_normal(u[...,n:2*n])*np.sqrt(2)
    beta = torch.complex(
        agsutil.icdf_std_normal(u[...,2*n:2*n+num_tril]),
        agsutil.icdf_std_normal(u[...,2*n+num_tril:]))
    theta = 2*np.pi*u[...,:n]
    il0,il1 = torch.tril_indices(n,n,offset=-1,device=u.device)
    v = torch.zeros((*u.shape[:-1],n,n),dtype=beta.dtype,device=u.device)
    diag_indices = torch.arange(n,device=u.device)
    v[...,diag_indices,diag_indices] = alpha.to(beta.dtype)
    v[...,il0,il1] = beta
    v += v.tril(-1).conj().transpose(dim0=-2,dim1=-1)
    # assert torch.allclose(v[...,torch.arange(n,device=u.device),torch.arange(n,device=u.device)],alpha.to(v.dtype))
    gamma,q = torch.linalg.eigh(v)
    phases = torch.complex(torch.cos(theta),torch.sin(theta))
    q = q*phases[...,None,:]
    # assert torch.allclose(torch.einsum("...ij,...j,...kj->...ik",q,gamma,q.conj()),v)
    return q[...,:k]

def tf_Spn_SVD(u, n, k):
    r"""
    Transform uniform samples to random symplectic unitary matrices using SVD.

    Args:
        u (torch.Tensor): Uniformly distributed samples in [0, 1] of shape `(..., 4*n**2)`.
        n (int): Half the number of rows.
        k (int): Half the number of columns with `k<=n`.

    Returns:
        x (torch.Tensor): A batch of random symplectic unitary matrices of shape `(..., 2n, 2k)`.
        
    Examples:
        >>> torch.set_default_dtype(torch.float64) 
        >>> rng = torch.Generator().manual_seed(7)

        >>> delta = torch.arange(1,4,dtype=float)
        >>> delta = delta/torch.linalg.norm(delta)
        >>> n = delta.size(-1)
        >>> q = tf_Spn_SVD(torch.rand((2,4*n**2),generator=rng),n,n)
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
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q.conj()),torch.eye(2*n,dtype=q.dtype))
        True

        >>> k = 2 
        >>> q = tf_Spn_SVD(torch.rand((5,4*n**2),generator=rng),n,k)
        >>> q.shape
        torch.Size([5, 6, 4])
        >>> q
        tensor([[[-0.0633-0.1036j, -0.2454-0.0317j, -0.1075+0.2173j,  0.0357-0.2214j],
                 [-0.7398+0.2957j, -0.4981-0.1796j, -0.1062+0.0984j,  0.1297+0.0526j],
                 [ 0.0691+0.3965j,  0.2245-0.3309j,  0.0134-0.3295j,  0.4265+0.4968j],
                 [ 0.1075+0.2173j, -0.0357-0.2214j, -0.0633+0.1036j, -0.2454+0.0317j],
                 [ 0.1062+0.0984j, -0.1297+0.0526j, -0.7398-0.2957j, -0.4981+0.1796j],
                 [-0.0134-0.3295j, -0.4265+0.4968j,  0.0691-0.3965j,  0.2245+0.3309j]],
        <BLANKLINE>
                [[-0.0486-0.0043j,  0.1458+0.0475j,  0.2022-0.0570j,  0.0165+0.1342j],
                 [ 0.2863-0.1145j, -0.3776+0.5283j, -0.0927-0.4884j,  0.3623-0.2821j],
                 [-0.5816+0.0233j, -0.3505+0.1184j, -0.1007+0.5121j,  0.3969-0.1769j],
                 [-0.2022-0.0570j, -0.0165+0.1342j, -0.0486+0.0043j,  0.1458-0.0475j],
                 [ 0.0927-0.4884j, -0.3623-0.2821j,  0.2863+0.1145j, -0.3776-0.5283j],
                 [ 0.1007+0.5121j, -0.3969-0.1769j, -0.5816-0.0233j, -0.3505-0.1184j]],
        <BLANKLINE>
                [[ 0.1763-0.0398j, -0.5018-0.4715j,  0.1128+0.1180j, -0.0184+0.4873j],
                 [-0.6141-0.1638j,  0.0427-0.1202j,  0.1371+0.1876j,  0.1736-0.2845j],
                 [ 0.4854-0.2710j, -0.1455-0.1455j,  0.1575-0.3858j, -0.1879-0.2882j],
                 [-0.1128+0.1180j,  0.0184+0.4873j,  0.1763+0.0398j, -0.5018+0.4715j],
                 [-0.1371+0.1876j, -0.1736-0.2845j, -0.6141+0.1638j,  0.0427+0.1202j],
                 [-0.1575-0.3858j,  0.1879-0.2882j,  0.4854+0.2710j, -0.1455+0.1455j]],
        <BLANKLINE>
                [[ 0.0197-0.1907j, -0.1932-0.7409j, -0.1938-0.1155j,  0.4894-0.0170j],
                 [ 0.1002+0.0349j, -0.2719+0.1094j,  0.1073+0.0866j,  0.0376-0.0229j],
                 [-0.2872+0.4903j, -0.0111+0.1788j, -0.7311-0.1571j,  0.1123+0.2034j],
                 [ 0.1938-0.1155j, -0.4894-0.0170j,  0.0197+0.1907j, -0.1932+0.7409j],
                 [-0.1073+0.0866j, -0.0376-0.0229j,  0.1002-0.0349j, -0.2719-0.1094j],
                 [ 0.7311-0.1571j, -0.1123+0.2034j, -0.2872-0.4903j, -0.0111-0.1788j]],
        <BLANKLINE>
                [[ 0.1997+0.0214j, -0.4774+0.0789j,  0.3295-0.2111j,  0.1172-0.4257j],
                 [ 0.0496+0.7202j, -0.1458-0.1148j, -0.1592+0.0259j, -0.4487-0.0895j],
                 [-0.1664+0.2085j, -0.1348+0.0409j, -0.0706-0.4281j,  0.5542+0.0138j],
                 [-0.3295-0.2111j, -0.1172-0.4257j,  0.1997-0.0214j, -0.4774-0.0789j],
                 [ 0.1592+0.0259j,  0.4487-0.0895j,  0.0496-0.7202j, -0.1458+0.1148j],
                 [ 0.0706-0.4281j, -0.5542+0.0138j, -0.1664-0.2085j, -0.1348-0.0409j]]])
        >>> torch.allclose(torch.einsum("...ji,...jk->...ik",q,q.conj()),torch.eye(2*k,dtype=q.dtype))
        True

    """ 
    assert k<=n
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
    ivec = torch.cat([torch.arange(k),torch.arange(n,n+k)],dim=0)
    return q[...,ivec]
