import agsutil
import torch 
import numpy as np
import qmcpy as qp 

from .tf_On_Un_Spn import (
    tf_On_QR,
    tf_On_eig,
    tf_Un_QR,
    tf_Un_eig,
    tf_Spn_SVD,
)

def rand_On_QR(N, n, seed=None, device=None, qp_unif_gen=None):
    r"""
    Generate a batch of `N` random orthogonal matrices of size `(N, n, n)` using the QR decomposition. 
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        seed (int, optional): Random seed for reproducibility.
        device (str, optional): Device to store the tensor on (e.g., "cpu", "cuda").
        qp_unif_gen (qmcpy.DiscreteDistribution, optional): QMCPy distribution generator.

    Returns:
        x (torch.Tensor): A batch of `N` random orthogonal matrices of size `(N, n, n)`.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        
        >>> q = rand_On_QR(2,3,seed=7)
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
        
        >>> q = rand_On_QR(2,3,seed=7,qp_unif_gen=qp.Net)
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
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    rng = agsutil.get_torch_rng(seed,device=device)
    u = torch.from_numpy(qp_unif_gen(dimension=n**2,seed=seed)(N)).to(device)
    q = tf_On_QR(u)
    return q

def rand_Un_QR(N, n, seed=None, device=None, qp_unif_gen=None):
    r"""
    Generate a batch of `N` random unitary matrices of size `(N, n, n)` using the QR decomposition. 
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        seed (int, optional): Random seed for reproducibility.
        device (str, optional): Device to store the tensor on (e.g., "cpu", "cuda").
        qp_unif_gen (qmcpy.DiscreteDistribution, optional): QMCPy distribution generator.

    Returns:
        x (torch.Tensor): A batch of `N` random unitary matrices of size `(N, n, n)`.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = rand_Un_QR(2,3,seed=7)
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
        
        >>> q = rand_Un_QR(2,3,seed=7,qp_unif_gen=qp.Net)
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
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    rng = agsutil.get_torch_rng(seed,device=device)
    u = torch.from_numpy(qp_unif_gen(dimension=2*n**2,seed=seed)(N)).to(device)
    q = tf_Un_QR(u)
    return q

def rand_On_eig(N, n, seed=None, device=None, qp_unif_gen=None):
    r"""
    Generate a batch of `N` random orthogonal matrices of size `(N, n, n)` using the eigenvalue decomposition.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        seed (int, optional): Random seed for reproducibility.
        device (str, optional): Device to store the tensor on (e.g., "cpu", "cuda").
        qp_unif_gen (qmcpy.DiscreteDistribution, optional): QMCPy distribution generator.

    Returns:
        x (torch.Tensor): A batch of `N` random orthogonal matrices of size `(N, n, n)`.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = rand_On_eig(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[ 0.3748,  0.4469, -0.8123],
                 [-0.1178, -0.8461, -0.5199],
                 [ 0.9196, -0.2905,  0.2645]],
        <BLANKLINE>
                [[ 0.9026, -0.2322, -0.3624],
                 [-0.1140, -0.9409,  0.3190],
                 [ 0.4150,  0.2467,  0.8757]]])
        
        >>> q = rand_On_eig(2,3,seed=7,qp_unif_gen=qp.Net)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.7446, -0.3508,  0.5679],
                 [-0.5217,  0.8366, -0.1671],
                 [-0.4165, -0.4207, -0.8060]],
        <BLANKLINE>
                [[-0.0434, -0.8815,  0.4702],
                 [-0.2524,  0.4650,  0.8486],
                 [ 0.9666,  0.0818,  0.2427]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    u = torch.from_numpy(qp_unif_gen(dimension=n*(n-1)//2+2*n,seed=seed)(N)).to(device)
    q = tf_On_eig(u)
    return q

def rand_Un_eig(N, n, seed=None, device=None, qp_unif_gen=None):
    r"""
    Generate a batch of `N` random unitary matrices of size `(N, n, n)` using the eigenvalue decomposition.
    
    Args:
        N (int): Number of samples to generate.
        n (int): Dimension of the square matrix.
        seed (int, optional): Random seed for reproducibility.
        device (str, optional): Device to store the tensor on (e.g., "cpu", "cuda").
        qp_unif_gen (qmcpy.DiscreteDistribution, optional): QMCPy distribution generator.

    Returns:
        x (torch.Tensor): A batch of `N` random unitary matrices of size `(N, n, n)`.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = rand_Un_eig(2,3,seed=7)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[-0.4009-0.1134j,  0.5152+0.3147j, -0.4627+0.4978j],
                 [-0.0383-0.0098j, -0.5419+0.5188j,  0.3717+0.5453j],
                 [-0.5107-0.7510j, -0.0557-0.2638j,  0.3130-0.0670j]],
        <BLANKLINE>
                [[ 0.1113+0.1290j,  0.0108-0.7796j, -0.3388+0.4983j],
                 [ 0.0135-0.9445j,  0.1002+0.0782j, -0.2065+0.2213j],
                 [ 0.1714+0.2221j,  0.3955+0.4685j,  0.1415+0.7248j]]])
        
        >>> q = rand_Un_eig(2,3,seed=7,qp_unif_gen=qp.Net)
        >>> q.shape 
        torch.Size([2, 3, 3])
        >>> q
        tensor([[[ 0.5988-0.2838j, -0.3694+0.2998j, -0.5360+0.2174j],
                 [ 0.0116-0.3634j,  0.6324-0.2327j, -0.5162-0.3838j],
                 [-0.6395+0.1405j, -0.2209+0.5204j, -0.4418-0.2376j]],
        <BLANKLINE>
                [[-0.6903+0.0581j, -0.1665+0.1366j, -0.4519-0.5192j],
                 [-0.6830+0.0339j,  0.0105-0.3853j,  0.3131+0.5346j],
                 [-0.2250-0.0432j,  0.2931+0.8480j,  0.3613+0.1091j]]])
    """
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    u = torch.from_numpy(qp_unif_gen(dimension=n**2+n,seed=seed)(N)).to(device)
    q = tf_Un_eig(u)
    return q

def rand_Spn_SVD(N, n, seed=None, device=None, qp_unif_gen=None):
    r"""
    Generate a batch of `N` random symplectic unitary matrices of size `(N, 2n, 2n)` using the SVD decomposition. 
    
    Args:
        N (int): Number of samples to generate.
        n (int): Half the dimension of the square matrix.
        seed (int, optional): Random seed for reproducibility.
        device (str, optional): Device to store the tensor on (e.g., "cpu", "cuda").
        qp_unif_gen (qmcpy.DiscreteDistribution, optional): QMCPy distribution generator.

    Returns:
        x (torch.Tensor): A batch of `N` random symplectic unitary matrices of size `(N, 2n, 2n)`.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> q = rand_Spn_SVD(2,3,seed=7)
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
        
        >>> q = rand_Spn_SVD(2,3,seed=7,qp_unif_gen=qp.Net)
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
    if qp_unif_gen is None: qp_unif_gen = qp.IIDStdUniform
    u = torch.from_numpy(qp_unif_gen(dimension=4*n**2,seed=seed)(N)).to(device)
    q = tf_Spn_SVD(u)
    return q
