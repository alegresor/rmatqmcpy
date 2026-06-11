import agsutil
import torch 
import numpy as np

from .rand_stiefel_gr_flag_lgr import (
    rand_flag_real,
    rand_flag_complex,
    rand_flag_quaternionic,
    rand_stiefel_real,
    rand_stiefel_complex,
    rand_stiefel_quaternionic,
    rand_gr_real,
    rand_gr_complex,
    rand_gr_quaternionic,
    rand_lgr_real,
    rand_lgr_complex,
    rand_lgr_quaternionic,
)

class KernelMatern(object):
    r"""
    Matern chordal kernel.

    Args:
        nu (float): Smoothness parameter. Must be in `[1/2, 3/2, 5/2, 7/2, 9/2, 11/2, 13/2, np.inf]`.
        sigma2 (float, optional): Variance parameter.
        rho (float, optional): Length-scale parameter.

    Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> kernel = KernelMatern(nu=1/2)

        >>> x = rand_flag_real(3,7,seed=7)
        >>> x.shape 
        torch.Size([3, 7, 7])
        >>> k = kernel(x[:,None,:,:],x[None,:,:,:])
        >>> k.shape 
        torch.Size([3, 3])
        >>> k
        tensor([[1.0000, 0.5458, 0.5388],
                [0.5458, 1.0000, 0.5787],
                [0.5388, 0.5787, 1.0000]])
        
        >>> for rand_manifold in [
        ...         rand_flag_real,
        ...         rand_flag_complex,
        ...         rand_flag_quaternionic,
        ...         rand_stiefel_real,
        ...         rand_stiefel_complex,
        ...         rand_stiefel_quaternionic,
        ...         rand_gr_real,
        ...         rand_gr_complex,
        ...         rand_gr_quaternionic,
        ...         rand_lgr_real,
        ...         rand_lgr_complex,
        ...         rand_lgr_quaternionic,]:
        ...     x = rand_manifold(3,7,seed=7)
        ...     for nu in [1/2,3/2,5/2,7/2,9/2,11/2,13/2,np.inf]:
        ...         kernel = KernelMatern(nu=nu)
        ...         k = kernel(x[:,None,:,:],x[None,:,:,:])
        ...         assert k.shape==(3,3)
        ...         assert not k.isnan().any()
        ...         assert k.isfinite().all()
        ...         assert not k.is_complex()
    """
    SUPPORTED_NU = [1/2,3/2,5/2,7/2,9/2,11/2,13/2,np.inf]
    def __init__(self, nu=5/2, sigma2=1, rho=1):
        self.nu = nu
        self.sigma2 = sigma2 
        self.rho = rho 
        assert self.nu in self.SUPPORTED_NU, "nu should be in %s"%str(self.SUPPORTED_NU)
        if self.nu==1/2: 
            self.c = torch.tensor([1])
            self.nu_str = r"$\nu = 1/2$"
        elif self.nu==3/2:
            self.c = torch.tensor([1,np.sqrt(3)])
            self.nu_str = r"$\nu = 3/2$"
        elif self.nu==5/2:
            self.c = torch.tensor([1,np.sqrt(5),5/3])
            self.nu_str = r"$\nu = 5/2$"
        elif self.nu==7/2:
            self.c = torch.tensor([1,np.sqrt(7),14/5,7*np.sqrt(7)/15])
            self.nu_str = r"$\nu = 7/2$"
        elif self.nu==9/2:
            self.c = torch.tensor([1,3,27/7,18/7,27/35])
            self.nu_str = r"$\nu = 9/2$"
        elif self.nu==11/2:
            self.c = torch.tensor([1,np.sqrt(11),44/9,11*np.sqrt(11)/9,121/63,121*np.sqrt(11)/945])
            self.nu_str = r"$\nu = 11/2$"
        elif self.nu==13/2:
            self.c = torch.tensor([1,np.sqrt(13),65/11,52*np.sqrt(13)/33,338/99,169*np.sqrt(13)/495,2197/10395])
            self.nu_str = r"$\nu = 13/2$"
        elif self.nu==np.inf:
            self.nu_str = r"$\nu = \infty$"
    def __call__(self, x, y, dim=(-2,-1)):
        r"""
        Compute the Matern kernel between two batches of tensors.

        Args:
            x (torch.Tensor): First batch of tensors.
            y (torch.Tensor): Second batch of tensors.
            dim (tuple, optional): Dimensions over which to take the Frobenius distance..

        Returns:
            k (torch.Tensor): The computed Matern kernel values.
        """
        frob_dist = torch.linalg.norm(x-y,dim=dim).real
        rscaled = frob_dist/self.rho
        if self.nu==np.inf:
            k = torch.exp(-(rscaled)**2/2)
        else:
            exp_term = torch.exp(-np.sqrt(2*self.nu)*rscaled) 
            poly_term = (self.c.to(x.device)*rscaled[...,None]**torch.arange(len(self.c),dtype=x.real.dtype)).sum(-1)
            k = exp_term*poly_term
        return self.sigma2*k

class KernelPoly(object):
    r"""
    Polynomial kernel.

    Args:
        r (int): Degree of the polynomial.
        c (float): Constant offset.

   Examples:
        >>> torch.set_default_dtype(torch.float64) 

        >>> kernel = KernelPoly()

        >>> x = rand_flag_real(3,7,seed=7)
        >>> x.shape 
        torch.Size([3, 7, 7])
        >>> k = kernel(x[:,None,:,:],x[None,:,:,:])
        >>> k.shape 
        torch.Size([3, 3])
        >>> k
        tensor([[4.0000, 3.3004, 3.2718],
                [3.3004, 4.0000, 3.4239],
                [3.2718, 3.4239, 4.0000]])
        
        >>> for rand_manifold in [
        ...         rand_flag_real,
        ...         rand_flag_complex,
        ...         rand_flag_quaternionic,
        ...         rand_stiefel_real,
        ...         rand_stiefel_complex,
        ...         rand_stiefel_quaternionic,
        ...         rand_gr_real,
        ...         rand_gr_complex,
        ...         rand_gr_quaternionic,
        ...         rand_lgr_real,
        ...         rand_lgr_complex,
        ...         rand_lgr_quaternionic,]:
        ...     x = rand_manifold(3,7,seed=7)
        ...     k = kernel(x[:,None,:,:],x[None,:,:,:])
        ...     assert k.shape==(3,3)
        ...     assert not k.isnan().any()
        ...     assert k.isfinite().all()
        ...     assert not k.is_complex()
    """
    def __init__(self, t=2, c=1):
        assert t>=1 
        assert t%1==0
        assert c>0
        self.t = t
        self.c = c
    def __call__(self, x, y, dim=(-2,-1)):
        r"""
        Compute the polynomial kernel between two batches of tensors.

        Args:
            x (torch.Tensor): First batch of tensors.
            y (torch.Tensor): Second batch of tensors.
            dim (tuple, optional): Dimensions over which to take the Frobenius inner product.

        Returns:
            k (torch.Tensor): The computed polynomial kernel values.
        """
        frob_inner_prod = (x*y.conj()).sum(dim=dim).real
        v = (self.c+frob_inner_prod)**self.t
        return v