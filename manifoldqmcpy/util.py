import torch 
import agsutil
import haarpy 
import sympy

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
    lamlampows = (lam[:,None,None]*lam[None,:,None])**torch.arange(r+1,device=device)
    Nrange = torch.arange(N,device=device)
    Nsymbol = sympy.Symbol("N")
    for t in range(1,r+1):
        exprt = 0
        for ms in agsutil.enumerate_sums(N**2, t):
            mvec = torch.tensor(ms, dtype=int, device=device)
            mmat = mvec.reshape((N, N))
            c = agsutil.multinomialcoeff(torch.tensor(t, device=device), *mvec)
            lam_prod = lamlampows[Nrange[:,None],Nrange[None,:],mmat].prod()
            row_indices = []
            col_indices = []
            for i in range(N):
                for k in range(N):
                    exponent = mmat[i,k].item()
                    row_indices.extend([i]*(2*exponent))
                    col_indices.extend([k]*(2*exponent))
            exprtm = float(c*lam_prod)*haarpy.haar_integral_orthogonal((tuple(row_indices),tuple(col_indices)),Nsymbol)
            exprt = exprt+exprtm
        exprt = sympy.simplify(exprt)
        tau[t] = float(exprt.subs({Nsymbol:N}))
    return tau
