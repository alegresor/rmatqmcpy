import torch 
import agsutil
from jackpy.jack import ZonalPol
import sympy
from sympy.combinatorics.partitions import IntegerPartition

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
    device = lam.device
    tau = torch.ones(r+1,device=device)
    lamlist = lam.cpu().tolist()
    onesN = [1.]*N
    for t in range(1,r+1):
        exprt = 0
        for m in agsutil.enumerate_partitions(t):
            mpart = sympy.combinatorics.partitions.IntegerPartition(m)
            numerator = ZonalPol(N,mpart)(*lamlist)
            denominator = ZonalPol(N,mpart)(*onesN)
            exprtm = (numerator**2)/denominator
            exprt = exprt+exprtm
        exprt = sympy.simplify(exprt)
        tau[t] = float(exprt)
        # tau[t] = float(exprt.subs({Nsymbol:N}))
    return tau
