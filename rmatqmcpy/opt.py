import torch 

from .kernels import KernelMatern
from .rand_Stiefel_Gr_flag_LGr import rand_flag_R

def opt_weights_sum_1(x, kernel, dims=2, eps=1e-12):
    r"""
    Compute the optimal weights for a kernel quadrature rule.

    Args:
        x (torch.Tensor): A batch of samples of shape `(..., N, *m)` where `len(m)==dims`.
        kernel (callable): Kernel function.
        dims (int): See description of `x`. 
        eps (float, optional): Regularization parameter for numerical stability.

    Returns:
        w (torch.Tensor): Optimal weights of shape `(..., N)`.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> kernel = KernelMatern(nu=1/2)
        >>> x = rand_flag_R(3,7,seed=7)
        >>> x.shape
        torch.Size([3, 7, 7])
        >>> w = opt_weights_sum_1(x,kernel)
        >>> w.shape 
        torch.Size([3])
        >>> w.sum() 
        tensor(1.)
        >>> w
        tensor([0.3506, 0.3218, 0.3276])

        >>> x2 = rand_flag_R(3,7,seed=11)
        >>> xfull = torch.stack([x,x2],dim=0)
        >>> xfull.shape 
        torch.Size([2, 3, 7, 7])
        >>> wfull = opt_weights_sum_1(xfull,kernel)
        >>> wfull.shape 
        torch.Size([2, 3])
        >>> wfull.sum(-1) 
        tensor([1., 1.])
        >>> wfull
        tensor([[0.3506, 0.3218, 0.3276],
                [0.3015, 0.3055, 0.3931]])
    """
    assert x.ndim>=(dims+1)
    kmat = kernel(x.unsqueeze(-dims-1),x.unsqueeze(-dims-2),dim=tuple(j for j in range(-dims,0,1))) # (...,N,N)
    b = torch.ones(kmat.shape[:-1],device=x.device) # (...,N)
    L = torch.linalg.cholesky(kmat+eps*torch.eye(kmat.size(-1),device=x.device),upper=False) # (...,N,N)
    w = torch.cholesky_solve(b[...,None],L,upper=False)[...,0] # (...,N)
    return w/w.sum(-1,keepdim=True)

def wce_squared_plus_kernel_integral(x, kernel, w, dims=2):
    r"""
    Square of the worst case error (WCE) plus the integral value, 
    
    $$\mathrm{WCE}^2+\gamma = \boldsymbol{w}^\intercal \mathsf{K} \boldsymbol{w}$$
    
    where $\gamma = \int_\mathcal{M} K(\boldsymbol{x},\boldsymbol{y}) \mu(\mathrm{d} \boldsymbol{x})$ is assumed to be constant as a function of $\boldsymbol{x}$. 

    Args:
        x (torch.Tensor): A batch of samples of shape `(..., N, *m)` where `len(m)==dims`.
        w (torch.Tensor): A torch.tensor of shape `(..., N)`
        kernel (callable): Kernel function.
        dims (int): See description of `x`. 

    Returns:
        val (torch.Tensor): Squree WCE plus kernel integral value.
    
    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> kernel = KernelMatern(nu=1/2)
        >>> x = rand_flag_R(3,7,seed=7)
        >>> x.shape
        torch.Size([3, 7, 7])
        >>> w = opt_weights_sum_1(x,kernel)
        >>> w.shape 
        torch.Size([3])
        >>> w.sum() 
        tensor(1.)
        >>> w
        tensor([0.3506, 0.3218, 0.3276])
        >>> 
        >>> wce_squared_plus_kernel_integral(x,kernel,w)
        tensor(0.7027)

        >>> x2 = rand_flag_R(3,7,seed=11)
        >>> xfull = torch.stack([x,x2],dim=0)
        >>> xfull.shape 
        torch.Size([2, 3, 7, 7])
        >>> wfull = opt_weights_sum_1(xfull,kernel)
        >>> wfull.shape 
        torch.Size([2, 3])
        >>> wfull.sum(-1) 
        tensor([1., 1.])
        >>> wfull
        tensor([[0.3506, 0.3218, 0.3276],
                [0.3015, 0.3055, 0.3931]])
        >>> wce_squared_plus_kernel_integral(xfull,kernel,wfull)
        tensor([0.7027, 0.7009])
    """
    assert x.ndim>=(dims+1)
    assert w.shape==x.shape[:-dims]
    kmat = kernel(x.unsqueeze(-dims-1),x.unsqueeze(-dims-2),dim=tuple(j for j in range(-dims,0,1))) # (...,N,N)
    return torch.einsum("...i,...ij,...j->...",w,kmat,w) # (...,)
