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

def rand_flag_R(N, n, delta=None, seed=None, rand_coe=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random real flag matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        delta (torch.Tensor): Eigenvalues of the flag matrices, defaults to `torch.arange(1,n+1)/torch.linalg.norm(torch.arange(1,n+1))`.
        seed (int): Random seed for reproducibility.
        rand_coe (callable): Function to generate random orthogonal matrices.
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
    if rand_coe is None: rand_coe = rand_On_QR
    assert rand_coe in [rand_On_QR,rand_On_eig]
    assert not torch.is_complex(delta)
    q = rand_coe(N=N,n=delta.size(-1),seed=seed,device=delta.device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...j,...kj->...ik",q,delta.to(q.dtype),q)
    return x

def rand_flag_C(N, n, delta=None, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random complex flag matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        delta (torch.Tensor): Eigenvalues of the flag matrices, defaults to `torch.arange(1,n+1)/torch.linalg.norm(torch.arange(1,n+1))`.
        seed (int): Random seed for reproducibility.
        rand_cue (callable): Function to generate random unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random complex flag matrices of size `(N, n, n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_flag_C(2,3,seed=7)
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
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if delta is None: 
        delta = torch.arange(1,n+1,device=device,dtype=torch.get_default_dtype())
        delta = delta/torch.linalg.norm(delta)
    assert delta.shape==(n,)
    assert not delta.is_complex()
    if rand_cue is None: rand_cue = rand_Un_QR
    assert rand_cue in [rand_Un_QR,rand_Un_eig]
    assert not torch.is_complex(delta)
    q = rand_cue(N=N,n=delta.size(-1),seed=seed,device=delta.device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...j,...kj->...ik",q,delta.to(q.dtype),q.conj())
    return x

def rand_flag_H(N, n, delta=None, seed=None, rand_cqe=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random quaternionic flag matrices of size `(N, 2n, 2n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        delta (torch.Tensor): Eigenvalues of the flag matrices, defaults to `torch.arange(1,n+1)/torch.linalg.norm(torch.arange(1,n+1))`.
        seed (int): Random seed for reproducibility.
        rand_cqe (callable): Function to generate random symplectic unitary matrices.
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
        tensor([[[ 0.1039+0.0000j, -0.0657+0.2741j,  0.2670+0.1920j, -0.2458+0.0377j,
                  -0.1937+0.0053j, -0.0665-0.1468j],
                 [-0.0657-0.2741j, -0.1392+0.0000j, -0.1863+0.1634j, -0.1937+0.0053j,
                  -0.1535-0.2037j,  0.0983+0.2447j],
                 [ 0.2670-0.1920j, -0.1863-0.1634j, -0.0219+0.0000j, -0.0665-0.1468j,
                   0.0983+0.2447j, -0.0638-0.2726j],
                 [ 0.2458+0.0377j,  0.1937+0.0053j,  0.0665-0.1468j,  0.1039+0.0000j,
                  -0.0657-0.2741j,  0.2670-0.1920j],
                 [ 0.1937+0.0053j,  0.1535-0.2037j, -0.0983+0.2447j, -0.0657+0.2741j,
                  -0.1392+0.0000j, -0.1863-0.1634j],
                 [ 0.0665-0.1468j, -0.0983+0.2447j,  0.0638-0.2726j,  0.2670+0.1920j,
                  -0.1863+0.1634j, -0.0219+0.0000j]],
        <BLANKLINE>
                [[-0.1280+0.0000j, -0.0233+0.0291j, -0.1971+0.0732j, -0.0638+0.3195j,
                  -0.2302-0.0179j,  0.2307+0.0797j],
                 [-0.0233-0.0291j, -0.0390+0.0000j,  0.0028-0.2075j, -0.2302-0.0179j,
                   0.1097+0.2249j, -0.3193+0.1491j],
                 [-0.1971-0.0732j,  0.0028+0.2075j,  0.0545+0.0000j,  0.2307+0.0797j,
                  -0.3193+0.1491j, -0.0072+0.3986j],
                 [ 0.0638+0.3195j,  0.2302-0.0179j, -0.2307+0.0797j, -0.1280+0.0000j,
                  -0.0233-0.0291j, -0.1971-0.0732j],
                 [ 0.2302-0.0179j, -0.1097+0.2249j,  0.3193+0.1491j, -0.0233+0.0291j,
                  -0.0390+0.0000j,  0.0028+0.2075j],
                 [-0.2307+0.0797j,  0.3193+0.1491j,  0.0072+0.3986j, -0.1971+0.0732j,
                   0.0028-0.2075j,  0.0545+0.0000j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if delta is None: 
        delta = torch.arange(1,n+1,device=device,dtype=torch.get_default_dtype())
        delta = delta/torch.linalg.norm(delta)
    assert delta.shape==(n,)
    assert not delta.is_complex()
    if rand_cqe is None: rand_cqe = rand_Spn_SVD
    assert rand_cqe in [rand_Spn_SVD]
    assert not torch.is_complex(delta)
    q = rand_cqe(N=N,n=n,seed=seed,device=delta.device,qp_unif_gen=qp_unif_gen)
    I_nn = torch.diag(torch.cat([torch.ones(n,device=delta.device),-torch.ones(n,device=delta.device)])).to(q.dtype)
    q_H = q.conj().transpose(-2,-1)
    q_left_corner = torch.einsum("ij,...jk,kl->...il",I_nn,q_H,I_nn)
    lam_paired = torch.cat([delta,delta],dim=-1)
    x = torch.einsum("...ij,...j,...jk->...ik",q,lam_paired.to(q.dtype),q_left_corner)
    return x

def rand_LGr_R(N, n, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random real Lagrangian Grassmannian matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the Lagrangian subspace.
        seed (int): Random seed for reproducibility.
        rand_cue (callable): Function to generate random unitary matrices.
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
    if rand_cue is None: rand_cue = rand_Un_QR
    assert rand_cue in [rand_Un_QR,rand_Un_eig]
    q = rand_cue(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...kj->...ik",q,q)
    return x

def rand_LGr_C(N, n, seed=None, rand_cqe=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random complex Lagrangian Grassmannian matrices of size `(N, 2n, 2n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Half the dimension of the square matrix.
        seed (int): Random seed for reproducibility.
        rand_cqe (callable): Function to generate random symplectic unitary matrices.
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
    if rand_cqe is None: rand_cqe = rand_Spn_SVD
    assert rand_cqe in [rand_Spn_SVD]
    q = rand_cqe(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    I_nn = torch.diag(torch.cat([
        torch.ones(n, device=device), 
        -torch.ones(n, device=device)
    ])).to(q.dtype)
    q_left_corner = torch.einsum("ij,...kj,kl->...il",I_nn,q.conj(),I_nn)
    x = torch.einsum("...ij,...jk->...ik",q,q_left_corner)
    return x

def rand_LGr_H(N, n, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random quaternionic Lagrangian Grassmannian matrices of size `(N, 2n, 2n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Half the dimension of the square matrix.
        seed (int): Random seed for reproducibility.
        rand_cue (callable): Function to generate random unitary matrices.
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
    if rand_cue is None: rand_cue = rand_Un_QR
    assert rand_cue in [rand_Un_QR,rand_Un_eig]
    q = rand_cue(N=N,n=2*n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    I_N = torch.eye(n,device=device,dtype=q.dtype)
    O_N = torch.zeros(n,n,device=device,dtype=q.dtype)
    J = torch.cat([
        torch.cat([ O_N,I_N],dim=-1),
        torch.cat([-I_N,O_N],dim=-1)],dim=-2)
    q_S = -torch.einsum("ij,...kj,kl->...il",J,q,J)
    x = torch.einsum("...ij,...jk->...ik",q,q_S)
    return x

def rand_Stiefel_R(N, n, k=1, seed=None, rand_coe=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random real Stiefel matrices of size `(N, n, k)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Number of rows.
        k (int): Number of columns.
        seed (int): Random seed for reproducibility.
        rand_coe (callable): Function to generate random orthogonal matrices.
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
        tensor([[[-0.8103, -0.3240, -0.2849],
                 [ 0.2444, -0.5330,  0.6263],
                 [ 0.1163, -0.7775, -0.2398],
                 [-0.5198,  0.0805,  0.6849]],
        <BLANKLINE>
                [[-0.6600,  0.6605,  0.3158],
                 [ 0.3410,  0.5402, -0.6430],
                 [ 0.0840,  0.4057, -0.1982],
                 [ 0.6642,  0.3277,  0.6690]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_coe is None: rand_coe = rand_On_QR
    assert rand_coe in [rand_On_QR,rand_On_eig]
    assert 1<=k<=n
    x = rand_coe(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)[...,:k]
    return x

def rand_Stiefel_C(N, n, k=1, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random complex Stiefel matrices of size `(N, n, k)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Number of rows.
        k (int): Number of columns.
        seed (int): Random seed for reproducibility.
        rand_cue (callable): Function to generate random unitary matrices.
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
        tensor([[[-0.5791+0.0748j, -0.2694-0.3827j, -0.1688-0.5285j],
                 [ 0.0832-0.5342j, -0.0508-0.4185j, -0.4533+0.0390j],
                 [-0.5636+0.1908j,  0.5696+0.1528j,  0.1186+0.0222j],
                 [ 0.0717+0.0866j, -0.4132-0.2909j,  0.6626+0.1775j]],
        <BLANKLINE>
                [[-0.1572-0.5114j, -0.0644-0.0293j,  0.0032+0.0135j],
                 [-0.2608+0.2782j,  0.3961-0.6225j,  0.3570+0.3802j],
                 [-0.6609-0.2334j, -0.2368-0.2797j, -0.1690-0.3820j],
                 [-0.1562-0.2296j, -0.5166+0.2225j,  0.3754+0.6422j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_cue is None: rand_cue = rand_Un_QR
    assert rand_cue in [rand_Un_QR,rand_Un_eig]
    assert 1<=k<=n
    x = rand_cue(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)[..., :k]
    return x

def rand_Stiefel_H(N, n, k=1, seed=None, rand_cqe=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random quaternionic Stiefel matrices of size `(N, 2n, 2k)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Half the number of rows.
        k (int): Half the number of columns.
        seed (int): Random seed for reproducibility.
        rand_cqe (callable): Function to generate random symplectic unitary matrices.
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
    if rand_cqe is None: rand_cqe = rand_Spn_SVD
    assert rand_cqe in [rand_Spn_SVD]
    assert 1<=k<=n
    q = rand_cqe(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    x = torch.cat([q[..., :k],q[...,n:n+k]],dim=-1)
    return x

def rand_Gr_R(N, n, k=1, seed=None, rand_coe=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random real Grassmannian matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the space.
        k (int): Dimension of the subspace.
        seed (int): Random seed for reproducibility.
        rand_coe (callable): Function to generate random orthogonal matrices.
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
        tensor([[[ 0.6854, -0.4075,  0.4519,  0.4000],
                 [-0.4075,  0.4721,  0.5853,  0.5181],
                 [ 0.4519,  0.5853,  0.3510, -0.5745],
                 [ 0.4000,  0.5181, -0.5745,  0.4915]],
        <BLANKLINE>
                [[ 0.9430, -0.1426,  0.2999, -0.0213],
                 [-0.1426,  0.6432,  0.7504, -0.0533],
                 [ 0.2999,  0.7504, -0.5782,  0.1122],
                 [-0.0213, -0.0533,  0.1122,  0.9920]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_coe is None: rand_coe = rand_On_QR
    assert rand_coe in [rand_On_QR,rand_On_eig]
    assert 1<=k<=n
    delta = torch.cat([torch.ones(k,device=device),-torch.ones(n-k,device=device)])
    x = rand_flag_R(N=N,n=n,delta=delta,seed=seed,rand_coe=rand_coe,qp_unif_gen=qp_unif_gen)
    return x

def rand_Gr_C(N, n, k=1, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random complex Grassmannian matrices of size `(N, n, n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the space.
        k (int): Dimension of the subspace.
        seed (int): Random seed for reproducibility.
        rand_cue (callable): Function to generate random unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random complex Grassmannian matrices of size `(N, n, n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_Gr_C(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 4, 4])
        >>> x
        tensor([[[ 0.7358+2.8368e-18j,  0.2834-3.0054e-01j,  0.1940-3.3491e-01j,
                  -0.0361-3.6989e-01j],
                 [ 0.2834+3.0054e-01j,  0.3541-5.3636e-18j, -0.5891+1.3857e-01j,
                  -0.3821+4.3791e-01j],
                 [ 0.1940+3.3491e-01j, -0.5891-1.3857e-01j,  0.4329+5.9115e-18j,
                  -0.4424+3.1745e-01j],
                 [-0.0361+3.6989e-01j, -0.3821-4.3791e-01j, -0.4424-3.1745e-01j,
                   0.4771+1.3055e-17j]],
        <BLANKLINE>
                [[-0.4170-9.4448e-18j, -0.2046+2.5806e-01j,  0.4820+5.7829e-01j,
                   0.3572+1.5258e-01j],
                 [-0.2046-2.5806e-01j,  0.9235-4.2559e-18j, -0.0357+1.7128e-01j,
                   0.0238+8.7094e-02j],
                 [ 0.4820-5.7829e-01j, -0.0357-1.7128e-01j,  0.6001+9.9685e-19j,
                  -0.1838+9.3896e-02j],
                 [ 0.3572-1.5258e-01j,  0.0238-8.7094e-02j, -0.1838-9.3896e-02j,
                   0.8935+8.6315e-18j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_cue is None: rand_cue = rand_Un_QR
    assert rand_cue in [rand_Un_QR,rand_Un_eig]
    assert 1<=k<=n
    delta = torch.cat([torch.ones(k,device=device),-torch.ones(n-k,device=device)])
    x = rand_flag_C(N=N,n=n,delta=delta,seed=seed,rand_cue=rand_cue,qp_unif_gen=qp_unif_gen)
    return x

def rand_Gr_H(N, n, k=1, seed=None, rand_cqe=None, qp_unif_gen=None, device=None):
    r"""
    Generate a batch of `N` random quaternionic Grassmannian matrices of size `(N, 2n, 2n)`.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Half the dimension of the space.
        k (int): Half the dimension of the subspace.
        seed (int): Random seed for reproducibility.
        rand_cqe (callable): Function to generate random symplectic unitary matrices.
        qp_unif_gen (qmcpy.DiscreteDistribution): QMCPy distribution generator.
        device (str): Device to store the tensor on.

    Returns:
        x (torch.Tensor): A batch of `N` random quaternionic Grassmannian matrices of size `(N, 2n, 2n)`.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> x = rand_Gr_H(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 8, 8])
        >>> x
        tensor([[[ 0.1175+0.0000j,  0.2598-0.2068j, -0.2905-0.0135j,  0.2096-0.3365j,
                  -0.0102+0.1331j, -0.3316-0.3418j,  0.1288-0.4876j,  0.1856+0.3176j],
                 [ 0.2598+0.2068j,  0.1405+0.0000j,  0.0434-0.1098j, -0.2902-0.1208j,
                  -0.3316-0.3418j, -0.2505-0.4770j, -0.3335+0.3315j, -0.1346-0.0318j],
                 [-0.2905+0.0135j,  0.0434+0.1098j, -0.3432+0.0000j,  0.3599-0.0520j,
                   0.1288-0.4876j, -0.3335+0.3315j, -0.1223+0.1892j, -0.2565+0.2438j],
                 [ 0.2096+0.3365j, -0.2902+0.1208j,  0.3599+0.0520j,  0.3649+0.0000j,
                   0.1856+0.3176j, -0.1346-0.0318j, -0.2565+0.2438j,  0.1413+0.4231j],
                 [ 0.0102+0.1331j,  0.3316-0.3418j, -0.1288-0.4876j, -0.1856+0.3176j,
                   0.1175+0.0000j,  0.2598+0.2068j, -0.2905+0.0135j,  0.2096+0.3365j],
                 [ 0.3316-0.3418j,  0.2505-0.4770j,  0.3335+0.3315j,  0.1346-0.0318j,
                   0.2598-0.2068j,  0.1405+0.0000j,  0.0434+0.1098j, -0.2902+0.1208j],
                 [-0.1288-0.4876j,  0.3335+0.3315j,  0.1223+0.1892j,  0.2565+0.2438j,
                  -0.2905-0.0135j,  0.0434-0.1098j, -0.3432+0.0000j,  0.3599+0.0520j],
                 [-0.1856+0.3176j,  0.1346-0.0318j,  0.2565+0.2438j, -0.1413+0.4231j,
                   0.2096-0.3365j, -0.2902-0.1208j,  0.3599-0.0520j,  0.3649+0.0000j]],
        <BLANKLINE>
                [[-0.0149+0.0000j,  0.1562-0.0075j,  0.4329+0.1111j,  0.2940+0.0937j,
                  -0.2253+0.3376j, -0.2412+0.0593j,  0.2220-0.4920j, -0.3949-0.0822j],
                 [ 0.1562+0.0075j,  0.2730+0.0000j,  0.1457-0.1226j, -0.1133-0.2403j,
                  -0.2412+0.0593j, -0.7185-0.0566j, -0.2008+0.2001j,  0.3572-0.0710j],
                 [ 0.4329-0.1111j,  0.1457+0.1226j, -0.0375+0.0000j,  0.3229+0.4110j,
                   0.2220-0.4920j, -0.2008+0.2001j, -0.2601+0.0177j, -0.2135-0.0645j],
                 [ 0.2940-0.0937j, -0.1133+0.2403j,  0.3229-0.4110j, -0.3198+0.0000j,
                  -0.3949-0.0822j,  0.3572-0.0710j, -0.2135-0.0645j, -0.1073-0.3195j],
                 [ 0.2253+0.3376j,  0.2412+0.0593j, -0.2220-0.4920j,  0.3949-0.0822j,
                  -0.0149+0.0000j,  0.1562+0.0075j,  0.4329-0.1111j,  0.2940-0.0937j],
                 [ 0.2412+0.0593j,  0.7185-0.0566j,  0.2008+0.2001j, -0.3572-0.0710j,
                   0.1562-0.0075j,  0.2730+0.0000j,  0.1457+0.1226j, -0.1133+0.2403j],
                 [-0.2220-0.4920j,  0.2008+0.2001j,  0.2601+0.0177j,  0.2135-0.0645j,
                   0.4329+0.1111j,  0.1457-0.1226j, -0.0375+0.0000j,  0.3229-0.4110j],
                 [ 0.3949-0.0822j, -0.3572-0.0710j,  0.2135-0.0645j,  0.1073-0.3195j,
                   0.2940+0.0937j, -0.1133-0.2403j,  0.3229+0.4110j, -0.3198+0.0000j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_cqe is None: rand_cqe = rand_Spn_SVD
    assert rand_cqe in [rand_Spn_SVD]
    assert 1<=k<=n
    delta = torch.cat([torch.ones(k,device=device),-torch.ones(n-k,device=device)])
    x = rand_flag_H(N=N,n=n,delta=delta,seed=seed,rand_cqe=rand_cqe,qp_unif_gen=qp_unif_gen)
    return x
