import agsutil
import torch 
import numpy as np
import qmcpy as qp

from .rand_On_Un_Spn import (
  rand_On_QR,
  rand_Un_QR,
  rand_On_eig,
  rand_Un_eig,
  rand_Spn_SVD,
) 

def rand_flag_R(N, n, delta=None, seed=None, rand_On=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random real flag matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        delta (torch.Tensor): Eigenvalues of the flag matrices, defaults to `torch.arange(1,n+1)/torch.linalg.norm(torch.arange(1,n+1))`.
        seed (int): Random seed for reproducibility.
        rand_On (callable): Function to generate random orthogonal matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.
    
    Returns:
        x (torch.Tensor): A batch of `N` random real flag matrices of size `(N, n, n)`.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_flag_R(2,3,seed=7)
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
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if delta is None: 
        delta = torch.arange(1,n+1,device=device,dtype=torch.get_default_dtype())
        delta = delta/torch.linalg.norm(delta)
    assert delta.shape==(n,)
    assert not delta.is_complex()
    if rand_On is None: rand_On = rand_On_QR
    assert rand_On in [rand_On_QR,rand_On_eig]
    assert not torch.is_complex(delta)
    n = delta.size(-1)
    q = rand_On(N=N,n=n,k=n,seed=seed,device=delta.device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...j,...kj->...ik",q,delta.to(q.dtype),q)
    return x

def rand_flag_C(N, n, delta=None, seed=None, rand_Un=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random complex flag matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        delta (torch.Tensor): Eigenvalues of the flag matrices, defaults to `torch.arange(1,n+1)/torch.linalg.norm(torch.arange(1,n+1))`.
        seed (int): Random seed for reproducibility.
        rand_Un (callable): Function to generate random unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random complex flag matrices of size `(N, n, n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_flag_C(2,3,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> torch.complex(x.real.round(decimals=4)+0.,x.imag.round(decimals=4)+0.)
        tensor([[[ 0.5330+0.0000j,  0.0552+0.1661j, -0.1884-0.0685j],
                 [ 0.0552-0.1661j,  0.5332+0.0000j,  0.0186+0.0154j],
                 [-0.1884+0.0685j,  0.0186-0.0154j,  0.5374+0.0000j]],
        <BLANKLINE>
                [[ 0.4512+0.0000j, -0.1101+0.0152j,  0.1480-0.1704j],
                 [-0.1101-0.0152j,  0.5788+0.0000j, -0.0483+0.0246j],
                 [ 0.1480+0.1704j, -0.0483-0.0246j,  0.5736+0.0000j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if delta is None: 
        delta = torch.arange(1,n+1,device=device,dtype=torch.get_default_dtype())
        delta = delta/torch.linalg.norm(delta)
    assert delta.shape==(n,)
    assert not delta.is_complex()
    if rand_Un is None: rand_Un = rand_Un_QR
    assert rand_Un in [rand_Un_QR,rand_Un_eig]
    assert not torch.is_complex(delta)
    n = delta.size(-1)
    q = rand_Un(N=N,n=n,k=n,seed=seed,device=delta.device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...j,...kj->...ik",q,delta.to(q.dtype),q.conj())
    return x

def rand_flag_H(N, n, delta=None, seed=None, rand_Spn=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random quaternionic flag matrices of size `(N, 2n, 2n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        delta (torch.Tensor): Eigenvalues of the flag matrices, defaults to `torch.arange(1,n+1)/torch.linalg.norm(torch.arange(1,n+1))`.
        seed (int): Random seed for reproducibility.
        rand_Spn (callable): Function to generate random symplectic unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random quaternionic flag matrices of size `(N, 2n, 2n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_flag_H(2,3,seed=7)
        >>> x.shape 
        torch.Size([2, 6, 6])
        >>> torch.complex(x.real.round(decimals=4)+0.,x.imag.round(decimals=4)+0.)
        tensor([[[ 0.5221+0.0000e+00j,  0.0063-1.2520e-01j,  0.0377-5.4600e-02j,
                   0.0000+0.0000e+00j,  0.0484+4.0300e-02j,  0.0241+1.6270e-01j],
                 [ 0.0063+1.2520e-01j,  0.5409+0.0000e+00j, -0.0698-1.0140e-01j,
                  -0.0484-4.0300e-02j,  0.0000+0.0000e+00j, -0.0412+5.7600e-02j],
                 [ 0.0377+5.4600e-02j, -0.0698+1.0140e-01j,  0.5406+0.0000e+00j,
                  -0.0241-1.6270e-01j,  0.0412-5.7600e-02j,  0.0000+0.0000e+00j],
                 [ 0.0000+0.0000e+00j, -0.0484+4.0300e-02j, -0.0241+1.6270e-01j,
                   0.5221+0.0000e+00j,  0.0063+1.2520e-01j,  0.0377+5.4600e-02j],
                 [ 0.0484-4.0300e-02j,  0.0000+0.0000e+00j,  0.0412+5.7600e-02j,
                   0.0063-1.2520e-01j,  0.5409+0.0000e+00j, -0.0698+1.0140e-01j],
                 [ 0.0241-1.6270e-01j, -0.0412-5.7600e-02j,  0.0000+0.0000e+00j,
                   0.0377-5.4600e-02j, -0.0698-1.0140e-01j,  0.5406+0.0000e+00j]],
        <BLANKLINE>
                [[ 0.4803+0.0000e+00j, -0.0923-1.2690e-01j,  0.1221+3.1900e-02j,
                   0.0000+0.0000e+00j, -0.0532-1.0000e-04j,  0.0830-8.8000e-03j],
                 [-0.0923+1.2690e-01j,  0.4942+0.0000e+00j,  0.0287+3.6500e-02j,
                   0.0532+1.0000e-04j,  0.0000+0.0000e+00j, -0.0995+4.8100e-02j],
                 [ 0.1221-3.1900e-02j,  0.0287-3.6500e-02j,  0.6290+0.0000e+00j,
                  -0.0830+8.8000e-03j,  0.0995-4.8100e-02j,  0.0000+0.0000e+00j],
                 [ 0.0000+0.0000e+00j,  0.0532-1.0000e-04j, -0.0830-8.8000e-03j,
                   0.4803+0.0000e+00j, -0.0923+1.2690e-01j,  0.1221-3.1900e-02j],
                 [-0.0532+1.0000e-04j,  0.0000+0.0000e+00j,  0.0995+4.8100e-02j,
                  -0.0923-1.2690e-01j,  0.4942+0.0000e+00j,  0.0287-3.6500e-02j],
                 [ 0.0830+8.8000e-03j, -0.0995-4.8100e-02j,  0.0000+0.0000e+00j,
                   0.1221+3.1900e-02j,  0.0287+3.6500e-02j,  0.6290+0.0000e+00j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if delta is None: 
        delta = torch.arange(1,n+1,device=device,dtype=torch.get_default_dtype())
        delta = delta/torch.linalg.norm(delta)
    assert delta.shape==(n,)
    assert not delta.is_complex()
    if rand_Spn is None: rand_Spn = rand_Spn_SVD
    assert rand_Spn in [rand_Spn_SVD]
    assert not torch.is_complex(delta)
    q = rand_Spn(N=N,n=n,k=n,seed=seed,device=delta.device,qp_unif_gen=qp_unif_gen)
    lam_paired = torch.cat([delta, delta], dim=-1)
    x = torch.einsum("...ij,...j,...kj->...ik", q,lam_paired.to(q.dtype),q.conj())
    return x

def rand_LGr_R(N, n, seed=None, rand_Un=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random real Lagrangian Grassmannian matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the Lagrangian subspace.
        seed (int): Random seed for reproducibility.
        rand_Un (callable): Function to generate random unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random real Lagrangian Grassmannian matrices of size `(N, n, n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_LGr_R(2,3,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.5347-0.6183j, -0.4216+0.1853j,  0.3367+0.0794j],
                 [-0.4216+0.1853j, -0.7076-0.3250j,  0.3573-0.2321j],
                 [ 0.3367+0.0794j,  0.3573-0.2321j,  0.3638-0.7526j]],
        <BLANKLINE>
                [[ 0.1151+0.8999j, -0.0811+0.0124j,  0.3417+0.2311j],
                 [-0.0811+0.0124j,  0.1517+0.9848j, -0.0134+0.0125j],
                 [ 0.3417+0.2311j, -0.0134+0.0125j, -0.8817+0.2282j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_Un is None: rand_Un = rand_Un_QR
    assert rand_Un in [rand_Un_QR,rand_Un_eig]
    q = rand_Un(N=N,n=n,k=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...kj->...ik",q,q)
    return x

def rand_LGr_C(N, n, seed=None, rand_Spn=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random complex Lagrangian Grassmannian matrices of size `(N, 2n, 2n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Half the dimension of the square matrix.
        seed (int): Random seed for reproducibility.
        rand_Spn (callable): Function to generate random symplectic unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random complex Lagrangian Grassmannian matrices of size `(N, 2n, 2n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_LGr_C(2,3,seed=7)
        >>> x.shape 
        torch.Size([2, 6, 6])
        >>> torch.complex(x.real.round(decimals=4)+0.,x.imag.round(decimals=4)+0.)
        tensor([[[ 0.2639+0.0000j,  0.0442+0.5590j,  0.4301+0.3442j, -0.3209-0.2230j,
                  -0.3178+0.0323j, -0.2092-0.1182j],
                 [ 0.0442-0.5590j, -0.0769+0.0000j, -0.2988+0.1905j, -0.3178+0.0323j,
                  -0.2813-0.1818j,  0.3129+0.4919j],
                 [ 0.4301-0.3442j, -0.2988-0.1905j, -0.0221+0.0000j, -0.2092-0.1182j,
                   0.3129+0.4919j,  0.1832-0.3732j],
                 [ 0.3209-0.2230j,  0.3178+0.0323j,  0.2092-0.1182j,  0.2639+0.0000j,
                   0.0442-0.5590j,  0.4301-0.3442j],
                 [ 0.3178+0.0323j,  0.2813-0.1818j, -0.3129+0.4919j,  0.0442+0.5590j,
                  -0.0769+0.0000j, -0.2988-0.1905j],
                 [ 0.2092-0.1182j, -0.3129+0.4919j, -0.1832-0.3732j,  0.4301+0.3442j,
                  -0.2988+0.1905j, -0.0221+0.0000j]],
        <BLANKLINE>
                [[-0.0856+0.0000j, -0.1772+0.0371j, -0.3202+0.0526j, -0.3455+0.5465j,
                  -0.4689+0.1672j,  0.4316-0.0500j],
                 [-0.1772-0.0371j, -0.0281+0.0000j,  0.0949-0.2171j, -0.4689+0.1672j,
                   0.2348+0.5843j, -0.4891+0.1633j],
                 [-0.3202-0.0526j,  0.0949+0.2171j,  0.0069+0.0000j,  0.4316-0.0500j,
                  -0.4891+0.1633j, -0.1544+0.6000j],
                 [ 0.3455+0.5465j,  0.4689+0.1672j, -0.4316-0.0500j, -0.0856+0.0000j,
                  -0.1772-0.0371j, -0.3202-0.0526j],
                 [ 0.4689+0.1672j, -0.2348+0.5843j,  0.4891+0.1633j, -0.1772+0.0371j,
                  -0.0281+0.0000j,  0.0949+0.2171j],
                 [-0.4316-0.0500j,  0.4891+0.1633j,  0.1544+0.6000j, -0.3202+0.0526j,
                   0.0949-0.2171j,  0.0069+0.0000j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_Spn is None: rand_Spn = rand_Spn_SVD
    assert rand_Spn in [rand_Spn_SVD]
    q = rand_Spn(N=N,n=n,k=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    I_nn = torch.diag(torch.cat([
        torch.ones(n,device=device), 
        -torch.ones(n,device=device)
    ])).to(q.dtype)
    q_left_corner = torch.einsum("ij,...kj,kl->...il",I_nn,q.conj(),I_nn)
    x = torch.einsum("...ij,...jk->...ik",q,q_left_corner)
    return x

def rand_LGr_H(N, n, seed=None, rand_Un=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random quaternionic Lagrangian Grassmannian matrices of size `(N, 2n, 2n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Half the dimension of the square matrix.
        seed (int): Random seed for reproducibility.
        rand_Un (callable): Function to generate random unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random quaternionic Lagrangian Grassmannian matrices of size `(N, 2n, 2n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_LGr_H(2,3,seed=7)
        >>> x.shape 
        torch.Size([2, 6, 6])
        >>> torch.complex(x.real.round(decimals=4)+0.,x.imag.round(decimals=4)+0.)
        tensor([[[ 0.2525+0.4835j,  0.3498+0.2373j,  0.0075+0.2605j,  0.0000+0.0000j,
                  -0.0722+0.0447j, -0.0564+0.6675j],
                 [-0.1068+0.5171j, -0.1195+0.0116j,  0.6605+0.3107j,  0.0722-0.0447j,
                   0.0000+0.0000j,  0.1007-0.3958j],
                 [-0.0877-0.3649j,  0.1217+0.4282j,  0.1840+0.1076j,  0.0564-0.6675j,
                  -0.1007+0.3958j,  0.0000+0.0000j],
                 [ 0.0000+0.0000j,  0.5042+0.1211j, -0.1175+0.0137j,  0.2525+0.4835j,
                  -0.1068+0.5171j, -0.0877-0.3649j],
                 [-0.5042-0.1211j,  0.0000+0.0000j,  0.4195-0.4047j,  0.3498+0.2373j,
                  -0.1195+0.0116j,  0.1217+0.4282j],
                 [ 0.1175-0.0137j, -0.4195+0.4047j,  0.0000+0.0000j,  0.0075+0.2605j,
                   0.6605+0.3107j,  0.1840+0.1076j]],
        <BLANKLINE>
                [[ 0.0364+0.3162j, -0.0919+0.2657j, -0.3184+0.4507j,  0.0000+0.0000j,
                  -0.0340-0.2485j, -0.1330+0.6592j],
                 [-0.2394+0.0866j,  0.0133-0.2684j, -0.6471+0.0804j,  0.0340+0.2485j,
                   0.0000+0.0000j,  0.5890-0.1670j],
                 [ 0.0843-0.2270j, -0.1381-0.0200j, -0.3063-0.0327j,  0.1330-0.6592j,
                  -0.5890+0.1670j,  0.0000+0.0000j],
                 [ 0.0000+0.0000j, -0.1605+0.8300j, -0.1159-0.2173j,  0.0364+0.3162j,
                  -0.2394+0.0866j,  0.0843-0.2270j],
                 [ 0.1605-0.8300j,  0.0000+0.0000j, -0.1564+0.3004j, -0.0919+0.2657j,
                   0.0133-0.2684j, -0.1381-0.0200j],
                 [ 0.1159+0.2173j,  0.1564-0.3004j,  0.0000+0.0000j, -0.3184+0.4507j,
                  -0.6471+0.0804j, -0.3063-0.0327j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_Un is None: rand_Un = rand_Un_QR
    assert rand_Un in [rand_Un_QR,rand_Un_eig]
    q = rand_Un(N=N,n=2*n,k=2*n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    I_N = torch.eye(n,device=device,dtype=q.dtype)
    O_N = torch.zeros(n,n,device=device,dtype=q.dtype)
    J = torch.cat([
        torch.cat([O_N,-I_N],dim=-1),
        torch.cat([I_N, O_N],dim=-1)],dim=-2)
    q_S = -torch.einsum("ij,...kj,kl->...il",J,q,J)
    x = torch.einsum("...ij,...jk->...ik",q,q_S)
    return x

def rand_Stiefel_R(N, n, k=1, seed=None, rand_On=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random real Stiefel matrices of size `(N, n, k)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Number of rows.
        k (int): Number of columns.
        seed (int): Random seed for reproducibility.
        rand_On (callable): Function to generate random orthogonal matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random real Stiefel matrices of size `(N, n, k)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_Stiefel_R(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 4, 3])
        >>> x
        tensor([[[-0.7177,  0.3121,  0.1379],
                 [ 0.1644,  0.3183, -0.9033],
                 [ 0.1398, -0.7875, -0.0974],
                 [-0.6620, -0.4256, -0.3944]],
        <BLANKLINE>
                [[-0.7589,  0.1929,  0.3816],
                 [ 0.1093, -0.6896,  0.7074],
                 [ 0.6078,  0.5222,  0.4722],
                 [ 0.2064, -0.4631, -0.3620]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_On is None: rand_On = rand_On_QR
    assert rand_On in [rand_On_QR,rand_On_eig]
    assert 1<=k<=n
    x = rand_On(N=N,n=n,k=k,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    return x

def rand_Stiefel_C(N, n, k=1, seed=None, rand_Un=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random complex Stiefel matrices of size `(N, n, k)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Number of rows.
        k (int): Number of columns.
        seed (int): Random seed for reproducibility.
        rand_Un (callable): Function to generate random unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random complex Stiefel matrices of size `(N, n, k)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_Stiefel_C(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 4, 3])
        >>> x
        tensor([[[-0.6106+0.0789j,  0.0853-0.1095j,  0.2571-0.3622j],
                 [ 0.1189-0.3923j,  0.2634-0.6302j, -0.1178-0.5404j],
                 [-0.3917+0.2437j,  0.6549-0.2208j, -0.2803+0.4433j],
                 [ 0.3137+0.3765j,  0.1450+0.1248j, -0.3241-0.3415j]],
        <BLANKLINE>
                [[ 0.0893+0.1079j, -0.1481-0.1997j,  0.8927-0.1751j],
                 [ 0.0739-0.3969j, -0.1797-0.3939j,  0.0373-0.1982j],
                 [-0.0191-0.4580j,  0.0510-0.6749j, -0.1948+0.0873j],
                 [ 0.3224-0.7095j, -0.1439+0.5215j,  0.1805+0.2314j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_Un is None: rand_Un = rand_Un_QR
    assert rand_Un in [rand_Un_QR,rand_Un_eig]
    assert 1<=k<=n
    x = rand_Un(N=N,n=n,k=k,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    return x

def rand_Stiefel_H(N, n, k=1, seed=None, rand_Spn=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random quaternionic Stiefel matrices of size `(N, 2n, 2k)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Half the number of rows.
        k (int): Half the number of columns.
        seed (int): Random seed for reproducibility.
        rand_Spn (callable): Function to generate random symplectic unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random quaternionic Stiefel matrices of size `(N, 2n, 2k)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_Stiefel_H(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 8, 6])
        >>> x
        tensor([[[-0.4140+0.3030j,  0.0601+0.0313j,  0.2725-0.3912j,  0.0845+0.0474j,
                   0.0467-0.2241j, -0.2175-0.1204j],
                 [-0.5555+0.0748j,  0.3572+0.0449j, -0.0927+0.1001j,  0.1535+0.3413j,
                  -0.2548-0.3059j,  0.1301+0.1163j],
                 [-0.0531-0.1221j, -0.0396-0.3151j, -0.1012+0.2916j,  0.1274-0.2141j,
                  -0.0340-0.2394j,  0.5060-0.4238j],
                 [-0.3121-0.2754j, -0.4287-0.3945j, -0.0973-0.0722j, -0.0424-0.1354j,
                  -0.3801-0.1076j, -0.1164+0.3205j],
                 [-0.0845+0.0474j, -0.0467-0.2241j,  0.2175-0.1204j, -0.4140-0.3030j,
                   0.0601-0.0313j,  0.2725+0.3912j],
                 [-0.1535+0.3413j,  0.2548-0.3059j, -0.1301+0.1163j, -0.5555-0.0748j,
                   0.3572-0.0449j, -0.0927-0.1001j],
                 [-0.1274-0.2141j,  0.0340-0.2394j, -0.5060-0.4238j, -0.0531+0.1221j,
                  -0.0396+0.3151j, -0.1012-0.2916j],
                 [ 0.0424-0.1354j,  0.3801-0.1076j,  0.1164+0.3205j, -0.3121+0.2754j,
                  -0.4287+0.3945j, -0.0973+0.0722j]],
        <BLANKLINE>
                [[ 0.2331-0.3342j,  0.4754+0.1704j, -0.0965+0.1623j, -0.3163-0.3840j,
                   0.2357+0.3150j, -0.2709+0.1039j],
                 [ 0.1156-0.2160j,  0.0750-0.2084j, -0.0375+0.0934j, -0.0851-0.1242j,
                  -0.1064-0.1801j,  0.3919+0.0141j],
                 [ 0.3396+0.1853j,  0.3284-0.2578j, -0.0459+0.3762j,  0.2561+0.2939j,
                  -0.3221-0.2715j, -0.3632+0.0764j],
                 [ 0.2941-0.0642j,  0.1781+0.0226j,  0.4171+0.1087j, -0.3197-0.0961j,
                  -0.2625-0.2109j,  0.1682-0.4698j],
                 [ 0.3163-0.3840j, -0.2357+0.3150j,  0.2709+0.1039j,  0.2331+0.3342j,
                   0.4754-0.1704j, -0.0965-0.1623j],
                 [ 0.0851-0.1242j,  0.1064-0.1801j, -0.3919+0.0141j,  0.1156+0.2160j,
                   0.0750+0.2084j, -0.0375-0.0934j],
                 [-0.2561+0.2939j,  0.3221-0.2715j,  0.3632+0.0764j,  0.3396-0.1853j,
                   0.3284+0.2578j, -0.0459-0.3762j],
                 [ 0.3197-0.0961j,  0.2625-0.2109j, -0.1682-0.4698j,  0.2941+0.0642j,
                   0.1781-0.0226j,  0.4171-0.1087j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_Spn is None: rand_Spn = rand_Spn_SVD
    assert rand_Spn in [rand_Spn_SVD]
    assert 1<=k<=n
    x = rand_Spn(N=N,n=n,k=k,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    return x

def rand_Gr_R(N, n, k=1, seed=None, rand_On=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random real Grassmannian matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the space.
        k (int): Dimension of the subspace.
        seed (int): Random seed for reproducibility.
        rand_On (callable): Function to generate random orthogonal matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random real Grassmannian matrices of size `(N, n, n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_Gr_R(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 4, 4])
        >>> x
        tensor([[[ 0.2631, -0.2864, -0.7190,  0.5760],
                 [-0.2864,  0.8887, -0.2795,  0.2239],
                 [-0.7190, -0.2795,  0.2984,  0.5620],
                 [ 0.5760,  0.2239,  0.5620,  0.5498]],
        <BLANKLINE>
                [[ 0.5176,  0.1079, -0.3608, -0.7683],
                 [ 0.1079,  0.9759,  0.0807,  0.1718],
                 [-0.3608,  0.0807,  0.7302, -0.5746],
                 [-0.7683,  0.1718, -0.5746, -0.2237]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_On is None: rand_On = rand_On_QR
    assert rand_On in [rand_On_QR,rand_On_eig]
    assert 1<=k<=n
    q = rand_On(N=N,n=n,k=k,seed=seed,qp_unif_gen=qp_unif_gen)
    x = 2*torch.einsum("...ij,...kj->...ik",q,q)-torch.eye(n)
    return x

def rand_Gr_C(N, n, k=1, seed=None, rand_Un=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random complex Grassmannian matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the space.
        k (int): Dimension of the subspace.
        seed (int): Random seed for reproducibility.
        rand_Un (callable): Function to generate random unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random complex Grassmannian matrices of size `(N, n, n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_Gr_C(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 4, 4])
        >>> torch.complex(x.real.round(decimals=4)+0.,x.imag.round(decimals=4)+0.)
        tensor([[[ 0.1911+0.0000j,  0.3068-0.0473j,  0.2115+0.1051j, -0.2456+0.8666j],
                 [ 0.3068+0.0473j,  0.8809+0.0000j, -0.0741-0.0522j,  0.1438-0.3143j],
                 [ 0.2115-0.1051j, -0.0741+0.0522j,  0.9310+0.0000j, -0.0484-0.2585j],
                 [-0.2456-0.8666j,  0.1438+0.3143j, -0.0484+0.2585j, -0.0031+0.0000j]],
        <BLANKLINE>
                [[ 0.8180+0.0000j,  0.2741+0.3828j, -0.2263-0.2302j, -0.0199-0.0682j],
                 [ 0.2741-0.3828j, -0.2178+0.0000j,  0.8249-0.1292j,  0.1734+0.0610j],
                 [-0.2263+0.2302j,  0.8249+0.1292j,  0.4275+0.0000j, -0.1110-0.0597j],
                 [-0.0199+0.0682j,  0.1734-0.0610j, -0.1110+0.0597j,  0.9723+0.0000j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_Un is None: rand_Un = rand_Un_QR
    assert rand_Un in [rand_Un_QR,rand_Un_eig]
    assert 1<=k<=n
    q = rand_Un(N=N,n=n,k=k,seed=seed,qp_unif_gen=qp_unif_gen)
    x = 2*torch.einsum("...ij,...kj->...ik",q,q.conj())-torch.eye(n)
    return x

def rand_Gr_H(N, n, k=1, seed=None, rand_Spn=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random quaternionic Grassmannian matrices of size `(N, 2n, 2n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Half the dimension of the space.
        k (int): Half the dimension of the subspace.
        seed (int): Random seed for reproducibility.
        rand_Spn (callable): Function to generate random symplectic unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random quaternionic Grassmannian matrices of size `(N, 2n, 2n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_Gr_H(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 8, 8])
        >>> torch.complex(x.real.round(decimals=4)+0.,x.imag.round(decimals=4)+0.)
        tensor([[[ 0.2373+0.0000j,  0.5093-0.1209j, -0.3505-0.3980j, -0.0150+0.0857j,
                   0.0000+0.0000j,  0.2004+0.0635j, -0.0076+0.2635j, -0.4841-0.1813j],
                 [ 0.5093+0.1209j,  0.5828+0.0000j,  0.1510+0.3912j,  0.1659-0.0622j,
                  -0.2004-0.0635j,  0.0000+0.0000j, -0.0784-0.0994j,  0.2977+0.1740j],
                 [-0.3505+0.3980j,  0.1510-0.3912j,  0.5401+0.0000j,  0.0957+0.2163j,
                   0.0076-0.2635j,  0.0784+0.0994j,  0.0000+0.0000j, -0.2873+0.1649j],
                 [-0.0150-0.0857j,  0.1659+0.0622j,  0.0957-0.2163j,  0.6397+0.0000j,
                   0.4841+0.1813j, -0.2977-0.1740j,  0.2873-0.1649j,  0.0000+0.0000j],
                 [ 0.0000+0.0000j, -0.2004+0.0635j,  0.0076+0.2635j,  0.4841-0.1813j,
                   0.2373+0.0000j,  0.5093+0.1209j, -0.3505+0.3980j, -0.0150-0.0857j],
                 [ 0.2004-0.0635j,  0.0000+0.0000j,  0.0784-0.0994j, -0.2977+0.1740j,
                   0.5093-0.1209j,  0.5828+0.0000j,  0.1510-0.3912j,  0.1659+0.0622j],
                 [-0.0076-0.2635j, -0.0784+0.0994j,  0.0000+0.0000j,  0.2873+0.1649j,
                  -0.3505-0.3980j,  0.1510+0.3912j,  0.5401+0.0000j,  0.0957-0.2163j],
                 [-0.4841+0.1813j,  0.2977-0.1740j, -0.2873-0.1649j,  0.0000+0.0000j,
                  -0.0150+0.0857j,  0.1659-0.0622j,  0.0957+0.2163j,  0.6397+0.0000j]],
        <BLANKLINE>
                [[ 0.8864+0.0000j,  0.0124+0.3467j, -0.1082-0.0184j,  0.1425-0.0718j,
                   0.0000+0.0000j,  0.1712+0.0225j,  0.0439+0.0287j, -0.1433-0.0589j],
                 [ 0.0124-0.3467j, -0.3211+0.0000j, -0.0039-0.2936j,  0.4309+0.3820j,
                  -0.1712-0.0225j,  0.0000+0.0000j, -0.2588+0.1369j,  0.3956-0.2944j],
                 [-0.1082+0.0184j, -0.0039+0.2936j,  0.8699+0.0000j,  0.1942-0.0780j,
                  -0.0439-0.0287j,  0.2588-0.1369j,  0.0000+0.0000j, -0.1091+0.0307j],
                 [ 0.1425+0.0718j,  0.4309-0.3820j,  0.1942+0.0780j,  0.5649+0.0000j,
                   0.1433+0.0589j, -0.3956+0.2944j,  0.1091-0.0307j,  0.0000+0.0000j],
                 [ 0.0000+0.0000j, -0.1712+0.0225j, -0.0439+0.0287j,  0.1433-0.0589j,
                   0.8864+0.0000j,  0.0124-0.3467j, -0.1082+0.0184j,  0.1425+0.0718j],
                 [ 0.1712-0.0225j,  0.0000+0.0000j,  0.2588+0.1369j, -0.3956-0.2944j,
                   0.0124+0.3467j, -0.3211+0.0000j, -0.0039+0.2936j,  0.4309-0.3820j],
                 [ 0.0439-0.0287j, -0.2588-0.1369j,  0.0000+0.0000j,  0.1091+0.0307j,
                  -0.1082-0.0184j, -0.0039-0.2936j,  0.8699+0.0000j,  0.1942+0.0780j],
                 [-0.1433+0.0589j,  0.3956+0.2944j, -0.1091-0.0307j,  0.0000+0.0000j,
                   0.1425-0.0718j,  0.4309+0.3820j,  0.1942-0.0780j,  0.5649+0.0000j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_Spn is None: rand_Spn = rand_Spn_SVD
    assert rand_Spn in [rand_Spn_SVD]
    assert 1<=k<=n
    q = rand_Spn(N=N,n=n,k=k,seed=seed,qp_unif_gen=qp_unif_gen)
    x = 2*torch.einsum("...ij,...kj->...ik",q,q.conj())-torch.eye(2*n)
    return x
