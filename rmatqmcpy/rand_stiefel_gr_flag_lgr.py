import agsutil
import torch 
import numpy as np
import qmcpy as qp

from .rand_coe_cue_cqe import (
  rand_coe_qr,
  rand_cue_qr,
  rand_coe_eig,
  rand_cue_eig,
  rand_cqe_svd,
)

def rand_flag_real(N, n, delta=None, seed=None, rand_coe=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_flag_real(2,3,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.3178,  0.0630,  0.0835],
                 [ 0.0630,  0.6904, -0.1516],
                 [ 0.0835, -0.1516,  0.5953]],
        <BLANKLINE>
                [[ 0.7379, -0.0859,  0.1154],
                 [-0.0859,  0.3014,  0.0494],
                 [ 0.1154,  0.0494,  0.5643]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if delta is None: 
        delta = torch.arange(1,n+1,device=device,dtype=torch.get_default_dtype())
        delta = delta/torch.linalg.norm(delta)
    assert delta.shape==(n,)
    assert not delta.is_complex()
    if rand_coe is None: rand_coe = rand_coe_eig
    assert rand_coe in [rand_coe_qr,rand_coe_eig]
    assert not torch.is_complex(delta)
    q = rand_coe(N=N,n=delta.size(-1),seed=seed,device=delta.device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...j,...kj->...ik",q,delta.to(q.dtype),q)
    return x

def rand_flag_complex(N, n, delta=None, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_flag_complex(2,3,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.3538-3.3957e-18j,  0.0460-3.7808e-02j,  0.0604+1.2777e-01j],
                 [ 0.0460+3.7808e-02j,  0.6703-1.2031e-17j, -0.1432-2.8555e-02j],
                 [ 0.0604-1.2777e-01j, -0.1432+2.8555e-02j,  0.5795+1.0825e-17j]],
        <BLANKLINE>
                [[ 0.6892+2.2244e-17j,  0.0162-4.9683e-02j, -0.1092-8.3062e-02j],
                 [ 0.0162+4.9683e-02j,  0.3004-7.6753e-18j,  0.0758+3.9791e-02j],
                 [-0.1092+8.3062e-02j,  0.0758-3.9791e-02j,  0.6140+7.8776e-20j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if delta is None: 
        delta = torch.arange(1,n+1,device=device,dtype=torch.get_default_dtype())
        delta = delta/torch.linalg.norm(delta)
    assert delta.shape==(n,)
    assert not delta.is_complex()
    if rand_cue is None: rand_cue = rand_cue_eig
    assert rand_cue in [rand_cue_qr,rand_cue_eig]
    assert not torch.is_complex(delta)
    q = rand_cue(N=N,n=delta.size(-1),seed=seed,device=delta.device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...j,...kj->...ik",q,delta.to(q.dtype),q.conj())
    return x

def rand_flag_quaternionic(N, n, delta=None, seed=None, rand_cqe=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_flag_quaternionic(2,3,seed=7)
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
    if rand_cqe is None: rand_cqe = rand_cqe_svd
    assert rand_cqe in [rand_cqe_svd]
    assert not torch.is_complex(delta)
    q = rand_cqe(N=N,n=n,seed=seed,device=delta.device,qp_unif_gen=qp_unif_gen)
    I_nn = torch.diag(torch.cat([torch.ones(n,device=delta.device),-torch.ones(n,device=delta.device)])).to(q.dtype)
    q_H = q.conj().transpose(-2,-1)
    q_left_corner = torch.einsum("ij,...jk,kl->...il",I_nn,q_H,I_nn)
    lam_paired = torch.cat([delta,delta],dim=-1)
    x = torch.einsum("...ij,...j,...jk->...ik",q,lam_paired.to(q.dtype),q_left_corner)
    return x

def rand_lgr_real(N, n, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_lgr_real(2,3,seed=7)
        >>> x.shape 
        torch.Size([2, 3, 3])
        >>> x
        tensor([[[ 0.3178+0.6180j, -0.2441+0.0138j, -0.4326+0.5198j],
                 [-0.2441+0.0138j,  0.8313+0.2826j, -0.1686+0.3754j],
                 [-0.4326+0.5198j, -0.1686+0.3754j,  0.6077+0.0631j]],
        <BLANKLINE>
                [[ 0.1087+0.2529j, -0.0766-0.3726j, -0.7681-0.4353j],
                 [-0.0766-0.3726j, -0.6980+0.4304j, -0.0309-0.4265j],
                 [-0.7681-0.4353j, -0.0309-0.4265j, -0.1256+0.1476j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_cue is None: rand_cue = rand_cue_eig
    assert rand_cue in [rand_cue_qr,rand_cue_eig]
    q = rand_cue(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    x = torch.einsum("...ij,...kj->...ik",q,q)
    return x

def rand_lgr_complex(N, n, seed=None, rand_cqe=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_lgr_complex(2,3,seed=7)
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
    if rand_cqe is None: rand_cqe = rand_cqe_svd
    assert rand_cqe in [rand_cqe_svd]
    q = rand_cqe(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    I_nn = torch.diag(torch.cat([
        torch.ones(n, device=device), 
        -torch.ones(n, device=device)
    ])).to(q.dtype)
    q_left_corner = torch.einsum("ij,...kj,kl->...il",I_nn,q.conj(),I_nn)
    x = torch.einsum("...ij,...jk->...ik",q,q_left_corner)
    return x

def rand_lgr_quaternionic(N, n, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_lgr_quaternionic(2,3,seed=7)
        >>> x.shape 
        torch.Size([2, 6, 6])
        >>> torch.complex(x.real.round(decimals=4)+0.,x.imag.round(decimals=4)+0.)
        tensor([[[-0.0078-0.2145j, -0.1052+0.1555j, -0.3302+0.4815j,  0.0000+0.0000j,
                  -0.4079+0.1238j, -0.6273+0.0499j],
                 [ 0.3170-0.3322j,  0.3424+0.4642j, -0.0552+0.2809j,  0.4079-0.1238j,
                   0.0000+0.0000j,  0.3971-0.1873j],
                 [ 0.2119+0.3043j,  0.0318-0.5068j,  0.1257+0.0113j,  0.6273-0.0499j,
                  -0.3971+0.1873j,  0.0000+0.0000j],
                 [ 0.0000+0.0000j,  0.0300-0.4565j, -0.4383+0.4519j, -0.0078-0.2145j,
                   0.3170-0.3322j,  0.2119+0.3043j],
                 [-0.0300+0.4565j,  0.0000+0.0000j, -0.1126+0.3902j, -0.1052+0.1555j,
                   0.3424+0.4642j,  0.0318-0.5068j],
                 [ 0.4383-0.4519j,  0.1126-0.3902j,  0.0000+0.0000j, -0.3302+0.4815j,
                  -0.0552+0.2809j,  0.1257+0.0113j]],
        <BLANKLINE>
                [[-0.5342+0.4689j, -0.0446+0.2017j, -0.4843+0.1871j,  0.0000+0.0000j,
                   0.3491-0.1905j,  0.0442+0.1498j],
                 [ 0.1458+0.2085j,  0.3481-0.4403j, -0.4243-0.2789j, -0.3491+0.1905j,
                   0.0000+0.0000j, -0.2218-0.3938j],
                 [-0.3098+0.2706j,  0.0235-0.5785j,  0.4580+0.2391j, -0.0442-0.1498j,
                   0.2218+0.3938j,  0.0000+0.0000j],
                 [ 0.0000+0.0000j,  0.1981+0.3767j,  0.2599+0.1106j, -0.5342+0.4689j,
                   0.1458+0.2085j, -0.3098+0.2706j],
                 [-0.1981-0.3767j,  0.0000+0.0000j,  0.2113+0.2851j, -0.0446+0.2017j,
                   0.3481-0.4403j,  0.0235-0.5785j],
                 [-0.2599-0.1106j, -0.2113-0.2851j,  0.0000+0.0000j, -0.4843+0.1871j,
                  -0.4243-0.2789j,  0.4580+0.2391j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_cue is None: rand_cue = rand_cue_eig
    assert rand_cue in [rand_cue_qr,rand_cue_eig]
    q = rand_cue(N=N,n=2*n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    I_N = torch.eye(n,device=device,dtype=q.dtype)
    O_N = torch.zeros(n,n,device=device,dtype=q.dtype)
    J = torch.cat([
        torch.cat([ O_N,I_N],dim=-1),
        torch.cat([-I_N,O_N],dim=-1)],dim=-2)
    q_S = -torch.einsum("ij,...kj,kl->...il",J,q,J)
    x = torch.einsum("...ij,...jk->...ik",q,q_S)
    return x

def rand_stiefel_real(N, n, k=1, seed=None, rand_coe=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_stiefel_real(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 4, 3])
        >>> x
        tensor([[[-0.1176,  0.6180,  0.0477],
                 [-0.9874,  0.0093,  0.0088],
                 [-0.0780, -0.6628,  0.5684],
                 [-0.0713, -0.4227, -0.8213]],
        <BLANKLINE>
                [[ 0.4306,  0.6373,  0.6389],
                 [ 0.1513,  0.4616, -0.5819],
                 [-0.0197,  0.4892, -0.4525],
                 [-0.8896,  0.3762,  0.2202]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_coe is None: rand_coe = rand_coe_eig
    assert rand_coe in [rand_coe_qr,rand_coe_eig]
    assert 1<=k<=n
    x = rand_coe(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)[...,:k]
    return x

def rand_stiefel_complex(N, n, k=1, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_stiefel_complex(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 4, 3])
        >>> x
        tensor([[[ 0.1736-0.4397j,  0.1390-0.1819j,  0.0704+0.2981j],
                 [-0.5270+0.0571j,  0.0988-0.5270j, -0.4812+0.4144j],
                 [-0.1272+0.1617j, -0.5136-0.1608j, -0.3481-0.5962j],
                 [-0.2991+0.6031j,  0.5796+0.1858j,  0.1019-0.1258j]],
        <BLANKLINE>
                [[ 0.0870-0.3583j,  0.5889+0.1773j, -0.0978-0.6111j],
                 [-0.0064+0.2081j,  0.3948+0.2854j, -0.0519+0.6386j],
                 [ 0.0865-0.5659j,  0.3603+0.0144j,  0.0202+0.4266j],
                 [ 0.6559-0.2505j, -0.2665-0.4282j, -0.1214+0.0970j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_cue is None: rand_cue = rand_cue_eig
    assert rand_cue in [rand_cue_qr,rand_cue_eig]
    assert 1<=k<=n
    x = rand_cue(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)[..., :k]
    return x

def rand_stiefel_quaternionic(N, n, k=1, seed=None, rand_cqe=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_stiefel_quaternionic(2,4,3,seed=7)
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
    if rand_cqe is None: rand_cqe = rand_cqe_svd
    assert rand_cqe in [rand_cqe_svd]
    assert 1<=k<=n
    q = rand_cqe(N=N,n=n,seed=seed,device=device,qp_unif_gen=qp_unif_gen)
    x = torch.cat([q[..., :k],q[...,n:n+k]],dim=-1)
    return x

def rand_gr_real(N, n, k=1, seed=None, rand_coe=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_gr_real(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 4, 4])
        >>> x
        tensor([[[-0.2041,  0.2445, -0.7467, -0.5840],
                 [ 0.2445,  0.9504,  0.1516,  0.1186],
                 [-0.7467,  0.1516,  0.5370, -0.3622],
                 [-0.5840,  0.1186, -0.3622,  0.7168]],
        <BLANKLINE>
                [[ 0.9993, -0.0249,  0.0284, -0.0052],
                 [-0.0249,  0.1492,  0.9723, -0.1783],
                 [ 0.0284,  0.9723, -0.1111,  0.2037],
                 [-0.0052, -0.1783,  0.2037,  0.9627]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_coe is None: rand_coe = rand_coe_eig
    assert rand_coe in [rand_coe_qr,rand_coe_eig]
    assert 1<=k<=n
    delta = torch.cat([torch.ones(k,device=device),-torch.ones(n-k,device=device)])
    x = rand_flag_real(N=N,n=n,delta=delta,seed=seed,rand_coe=rand_coe,qp_unif_gen=qp_unif_gen)
    return x

def rand_gr_complex(N, n, k=1, seed=None, rand_cue=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_gr_complex(2,4,3,seed=7)
        >>> x.shape 
        torch.Size([2, 4, 4])
        >>> x
        tensor([[[-0.2607-1.8716e-18j,  0.1653+2.0901e-01j, -0.6750+1.6372e-01j,
                  -0.6013-1.3045e-01j],
                 [ 0.1653-2.0901e-01j,  0.9437-1.2204e-17j,  0.0614-1.3338e-01j,
                   0.1005-8.2584e-02j],
                 [-0.6750-1.6372e-01j,  0.0614+1.3338e-01j,  0.6173-1.3142e-17j,
                  -0.3050-1.4794e-01j],
                 [-0.6013+1.3045e-01j,  0.1005+8.2584e-02j, -0.3050+1.4794e-01j,
                   0.6997-5.7473e-18j]],
        <BLANKLINE>
                [[ 0.7944+9.7022e-19j, -0.3542-3.9320e-02j,  0.3247+2.0615e-01j,
                  -0.2669+1.5072e-01j],
                 [-0.3542+3.9320e-02j,  0.3822-7.5994e-18j,  0.5989+2.9306e-01j,
                  -0.4311+3.1070e-01j],
                 [ 0.3247-2.0615e-01j,  0.5989-2.9306e-01j,  0.2804-2.9655e-18j,
                   0.2705-5.0568e-01j],
                 [-0.2669-1.5072e-01j, -0.4311-3.1070e-01j,  0.2705+5.0568e-01j,
                   0.5430-1.0418e-17j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    if rand_cue is None: rand_cue = rand_cue_eig
    assert rand_cue in [rand_cue_qr,rand_cue_eig]
    assert 1<=k<=n
    delta = torch.cat([torch.ones(k,device=device),-torch.ones(n-k,device=device)])
    x = rand_flag_complex(N=N,n=n,delta=delta,seed=seed,rand_cue=rand_cue,qp_unif_gen=qp_unif_gen)
    return x

def rand_gr_quaternionic(N, n, k=1, seed=None, rand_cqe=None, qp_unif_gen=None, device=None):
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

        >>> x = rand_gr_quaternionic(2,4,3,seed=7)
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
    if rand_cqe is None: rand_cqe = rand_cqe_svd
    assert rand_cqe in [rand_cqe_svd]
    assert 1<=k<=n
    delta = torch.cat([torch.ones(k,device=device),-torch.ones(n-k,device=device)])
    x = rand_flag_quaternionic(N=N,n=n,delta=delta,seed=seed,rand_cqe=rand_cqe,qp_unif_gen=qp_unif_gen)
    return x
