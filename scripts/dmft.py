import os
import pickle
import numpy as np
from scipy.linalg import toeplitz
from scipy.interpolate import interp1d,RegularGridInterpolator
from scipy.integrate import quad,simpson
from mpmath import fp
import time

dmu = 1e-4
sr2pi = np.sqrt(2*np.pi)

jtheta = np.vectorize(fp.jtheta, "D")

def wrapnormdens(x,s,L=180):
    '''
    Compute the wrapped normal PDF, centered at zero.
    
    Parameters
    ----------
    x : array_like
        The points at which to evaluate the PDF.
    s : float
        The standard deviation of the distribution.
    L : float, optional
        The period of the distribution. Default is 180.
        
    Returns
    -------
    array_like
        The PDF evaluated at the points in x.
    '''
    return np.real(jtheta(3,x*np.pi/L,np.exp(-(s*(2*np.pi)/L)**2/2)))/(2*np.pi)

def basesubwrapnorm(x,s,L=180):
    '''
    Compute the normalized baseline-subtracted wrapped normal PDF, centered at zero.
    
    Parameters
    ----------
    x : array_like
        The points at which to evaluate the PDF.
    s : float
        The standard deviation of the distribution.
    L : float, optional
        The period of the distribution. Default is 180.
        
    Returns
    -------
    array_like
        The normalized baseline-subtracted PDF evaluated at the points in x.
    '''
    return (wrapnormdens(x,s,L)-wrapnormdens(L/2,s,L))/(wrapnormdens(0,s,L)-wrapnormdens(L/2,s,L))

def mutox(mu):
    return np.sign(mu/100-0.2)*np.abs(mu/100-0.2)**0.5

def xtomu(x):
    return 100*(np.sign(x)*np.abs(x)**2.0+0.2)
    
def base_itp_moments(res_dir):
    '''
    Loads precomputed rate moments and returns interpolation functions.
    
    Parameters
    ----------
    res_dir : str
        The directory containing the precomputed rate moments.
    '''
    with open(os.path.join(res_dir,"itp_ranges.pkl"), "rb") as handle:
        ranges_dict = pickle.load(handle)
    
    Fxrange = ranges_dict["Ph"]["xrange"]
    Fxs = np.linspace(Fxrange[0],Fxrange[1],round(Fxrange[2])).astype(np.float32)
    FEs = np.load(os.path.join(res_dir,"PhE_itp.npy")).astype(np.float32)
    FIs = np.load(os.path.join(res_dir,"PhI_itp.npy")).astype(np.float32)
    
    Mxrange = ranges_dict["M"]["xrange"]
    Mxs = np.linspace(Mxrange[0],Mxrange[1],round(Mxrange[2])).astype(np.float32)
    Msrange = ranges_dict["M"]["σrange"]
    Mss = np.linspace(Msrange[0],Msrange[1],round(Msrange[2])).astype(np.float32)
    MEs = np.load(os.path.join(res_dir,"ME_itp.npy")).astype(np.float32)
    MIs = np.load(os.path.join(res_dir,"MI_itp.npy")).astype(np.float32)
    
    Cxrange = ranges_dict["C"]["xrange"]
    Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
    Csrange = ranges_dict["C"]["σrange"]
    Css = np.linspace(Csrange[0],Csrange[1],round(Csrange[2])).astype(np.float32)
    Ccrange = ranges_dict["C"]["crange"]
    Ccs = np.linspace(Ccrange[0],Ccrange[1],round(Ccrange[2])).astype(np.float32)
    CEs = np.load(os.path.join(res_dir,"CE_itp.npy")).astype(np.float32)
    CIs = np.load(os.path.join(res_dir,"CI_itp.npy")).astype(np.float32)
    
    FE_itp = RegularGridInterpolator((Fxs,),FEs,bounds_error=False,fill_value=None)
    FI_itp = RegularGridInterpolator((Fxs,),FIs,bounds_error=False,fill_value=None)
    
    ME_itp = RegularGridInterpolator((Mss,Mxs),MEs,bounds_error=False,fill_value=None)
    MI_itp = RegularGridInterpolator((Mss,Mxs),MIs,bounds_error=False,fill_value=None)
    
    CE_itp = RegularGridInterpolator((Ccs,Css,Cxs),CEs,bounds_error=False,fill_value=None)
    CI_itp = RegularGridInterpolator((Ccs,Css,Cxs),CIs,bounds_error=False,fill_value=None)
    
    def FE(mu):
        try:
            return FE_itp(mutox(1e3*mu)[:,None])
        except:
            return FE_itp([mutox(1e3*mu)])
    def FI(mu):
        try:
            return FI_itp(mutox(1e3*mu)[:,None])
        except:
            return FI_itp([mutox(1e3*mu)])
    
    def ME(mu,Sig):
        return ME_itp(np.row_stack(list(np.broadcast(1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    def MI(mu,Sig):
        return MI_itp(np.row_stack(list(np.broadcast(1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    
    def CE(mu,Sig,k):
        c = np.sign(k)*np.fmin(np.abs(k)/Sig,1)
        return CE_itp(np.row_stack(list(np.broadcast(c,1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    def CI(mu,Sig,k):
        c = np.sign(k)*np.fmin(np.abs(k)/Sig,1)
        return CI_itp(np.row_stack(list(np.broadcast(c,1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    
    return FE,FI,ME,MI,CE,CI
    
def opto_itp_moments(res_dir,L,CVL):
    '''
    Loads precomputed rate moments in the presence of optogenetic stimulation and returns interpolation functions.
    
    Parameters
    ----------
    res_dir : str
        The directory containing the precomputed rate moments.
    L : float
        Mean optogenetic input strength.
    CVL : float
        Coefficient of variation of optogenetic input strength.
    '''
    with open(os.path.join(res_dir,"itp_ranges.pkl"), "rb") as handle:
        ranges_dict = pickle.load(handle)
    
    Fxrange = ranges_dict["PhL"]["xrange"]
    Fxs = np.linspace(Fxrange[0],Fxrange[1],round(Fxrange[2])).astype(np.float32)
    FLs = np.load(os.path.join(res_dir,"PhL_itp_L={:.2f}_CVL={:.2f}.npy".format(L,CVL))).astype(np.float32)
    
    Mxrange = ranges_dict["ML"]["xrange"]
    Mxs = np.linspace(Mxrange[0],Mxrange[1],round(Mxrange[2])).astype(np.float32)
    Msrange = ranges_dict["ML"]["σrange"]
    Mss = np.linspace(Msrange[0],Msrange[1],round(Msrange[2])).astype(np.float32)
    MLs = np.load(os.path.join(res_dir,"ML_itp_L={:.2f}_CVL={:.2f}.npy".format(L,CVL))).astype(np.float32)
    
    Cxrange = ranges_dict["CL"]["xrange"]
    Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
    Csrange = ranges_dict["CL"]["σrange"]
    Css = np.linspace(Csrange[0],Csrange[1],round(Csrange[2])).astype(np.float32)
    Ccrange = ranges_dict["CL"]["crange"]
    Ccs = np.linspace(Ccrange[0],Ccrange[1],round(Ccrange[2])).astype(np.float32)
    CLs = np.load(os.path.join(res_dir,"CL_itp_L={:.2f}_CVL={:.2f}.npy".format(L,CVL))).astype(np.float32)
    
    FL_itp = RegularGridInterpolator((Fxs,),FLs,bounds_error=False,fill_value=None)
    
    ML_itp = RegularGridInterpolator((Mss,Mxs),MLs,bounds_error=False,fill_value=None)
    
    CL_itp = RegularGridInterpolator((Ccs,Css,Cxs),CLs,bounds_error=False,fill_value=None)
    
    def FL(mu):
        try:
            return FL_itp(mutox(1e3*mu)[:,None])
        except:
            return FL_itp([mutox(1e3*mu)])
    
    def ML(mu,Sig):
        return ML_itp(np.row_stack(list(np.broadcast(1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    
    def CL(mu,Sig,k):
        c = np.sign(k)*np.fmin(np.abs(k)/Sig,1)
        return CL_itp(np.row_stack(list(np.broadcast(c,1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    
    return FL,ML,CL

def R(M1,M2,mu1,mu2,Sig1,Sig2,k):
    '''
    Compute the cross-correlation of two functions, whose expectation values are computed from M1 and M2
    
    Parameters
    ----------
    M1 : function
        Expectation value function of the first function. It should take as input the mean and variance of the input distribution.
    M2 : function
        Expectation value function of the second function. See M1.
    mu1 : float
        Mean of the input to the first function.
    mu2 : float
        Mean of the input to the second function.
    Sig1 : float
        Variance of the input to the first function.
    Sig2 : float
        Variance of the input to the second function.
    k : float
        Covariance of the input to the two functions.
        
    Returns
    -------
    float
        The cross-correlation of the two functions.
    '''
    c = np.sign(k)*np.fmin(np.abs(k)/np.sqrt(Sig1*Sig2),1)
    sig1 = np.sign(c)*np.sqrt(Sig1*np.abs(c))
    sig2 = np.sqrt(Sig2*np.abs(c))
    Del1 = Sig1*(1-np.abs(c))
    Del2 = Sig2*(1-np.abs(c))
    return quad(lambda x: np.exp(-0.5*x**2)/sr2pi*\
                M1(mu1+sig1*x,Del1)*\
                M2(mu2+sig2*x,Del2),-8,8)[0]

def R_int(M1,M2,mu1,mu2,Sig1,Sig2,k,x):
    '''
    Compute the integrand of the Gaussian integral needed to compute the cross-correlation of two functions, whose expectation values are computed from M1 and M2
    
    Parameters
    ----------
    M1 : function
        Expectation value function of the first function. It should take as input the mean and variance of the input distribution.
    M2 : function
        Expectation value function of the second function. See M1.
    mu1 : float
        Mean of the input to the first function.
    mu2 : float
        Mean of the input to the second function.
    Sig1 : float
        Variance of the input to the first function.
    Sig2 : float
        Variance of the input to the second function.
    k : float
        Covariance of the input to the two functions.
        
    Returns
    -------
    float
        The integrand of the Gaussian integral needed to compute the cross-correlation of the two functions.
    '''
    c = np.sign(k)*np.fmin(np.abs(k)/np.sqrt(Sig1*Sig2),1)
    sig1 = np.sign(c)*np.sqrt(Sig1*np.abs(c))
    sig2 = np.sqrt(Sig2*np.abs(c))
    Del1 = Sig1*(1-np.abs(c))
    Del2 = Sig2*(1-np.abs(c))
    return np.exp(-0.5*x**2)/sr2pi*\
                M1(mu1+sig1*x,Del1)*\
                M2(mu2+sig2*x,Del2)

def R_simp(M1,M2,mu1,mu2,Sig1,Sig2,k):
    '''
    Compute the cross-correlation of two functions using Simpson's rule, whose expectation values are computed from M1 and M2
    
    Parameters
    ----------
    M1 : function
        Expectation value function of the first function. It should take as input the mean and variance of the input distribution.
    M2 : function
        Expectation value function of the second function. See M1.
    mu1 : float
        Mean of the input to the first function.
    mu2 : float
        Mean of the input to the second function.
    Sig1 : float
        Variance of the input to the first function.
    Sig2 : float
        Variance of the input to the second function.
    k : float
        Covariance of the input to the two functions.
        
    Returns
    -------
    float
        The cross-correlation of the two functions.
    '''
    xs = np.linspace(-8,8,1001)
    return simpson(R_int(M1,M2,mu1,mu2,Sig1,Sig2,k,xs),x=xs)

def doub_vec(A):
    '''
    Double the input vector or double each row of the input matrix (ie [1,2,3] -> [1,2,3,1,2,3])
    
    Parameters
    ----------
    A : array_like
        The input vector or matrix to be doubled.
    '''
    if A.ndim==1:
        return np.concatenate([A,A])
    else:
        return np.kron(np.ones(2)[...,None,:],A)

def doub_mat(A):
    '''
    Double the input matrix, creating a block-diagonal matrix (ie [A] -> [[A,0],[0,A]]). If A is 4D, the second and fourth dimensions will be doubled.
    
    Parameters
    ----------
    A : array_like
        The input matrix or 4D array to be doubled.
    '''
    if A.ndim==2:
        return np.block([[A,np.zeros_like(A)],[np.zeros_like(A),A]])
    elif A.ndim==4:
        return np.kron(np.eye(2)[...,None,:,:],A.transpose(0,2,1,3)).transpose(0,2,1,3)
    else:
        return np.kron(np.eye(2)[...,None,:,:],A)

def each_diag(A,k=0):
    '''
    Extract the k-th diagonal of the last two dimensions of A.
    
    Parameters
    ----------
    A : array_like
        (...,N,N) array
    '''
    if k == 0:
        return np.einsum("...jj->...j",A)
    else:
        new_shape = np.array(A.shape[:-1])
        new_shape[-1] -= k
        out = np.zeros(new_shape)
        mult_shape = A.shape[:-2]
        for i in range(np.prod(mult_shape)):
            mult_idx = np.unravel_index(i,mult_shape)
            out[mult_idx] = np.diag(A[mult_idx],k)
        return out

def each_matmul(A,B):
    if A.ndim==2:
        return np.einsum("ijk,jk->ik",A[:,:,None],B)
    else:
        return np.einsum("ijk,jk->ik",A,B)

def grid_stat(stat,A,Tstat,dt):
    '''
    Compute the variance of a time-averaged quantity from the autocorrelation function.
    
    Parameters
    ----------
    A : array_like
        Array whose last dimension contains the autocorrelation function.
    Tstat : float
        Time over which to compute the time average.
    dt : float
        Time step.
        
    Returns
    -------
    array_like
        The variance of the time-averaged quantity, broadcasted over all but the last dimensions of A.
    '''
    Nsav = A.shape[-1]
    Nstat = round(Tstat/dt)+1
    new_shape = np.array(A.shape)
    new_shape[-1] = Nstat
    A_ext = np.zeros(new_shape)
    if Nsav < Nstat:
        A_ext[...,:Nsav] = A
        A_ext[...,Nsav:] = A[...,-1:]
    else:
        A_ext = A[...,:Nstat]
    mult_shape = A.shape[:-1]
    new_shape = np.concatenate([new_shape,[Nstat]])
    A_mat = np.zeros(new_shape)
    for i in range(np.prod(mult_shape)):
        mult_idx = np.unravel_index(i,mult_shape)
        A_mat[mult_idx] = toeplitz(A_ext[mult_idx])
    return stat(A_mat,axis=(-1,-2))

def d2_stencil(Tsav,dt):
    '''
    Create a finite difference matrix for the second derivative operator with zero-derivative left-boundary conditions and asymptotically constant right-boundary conditions.
    
    Parameters
    ----------
    Tsav : float
        Amount of time over which autocorrelations were saved.
    dt : float
        Time step.
        
    Returns
    -------
    array_like
        2D finite difference matrix for the second derivative operator.
    '''
    Nsav = round(Tsav/dt)+1
    d2_mat = np.zeros((Nsav,Nsav))
    d2_mat[(np.arange(Nsav), np.arange(Nsav))] = -2/dt**2
    d2_mat[(np.arange(Nsav-1), np.arange(1,Nsav))] = 1/dt**2
    d2_mat[(np.arange(1,Nsav), np.arange(Nsav-1))] = 1/dt**2
    d2_mat[0,1] = 2/dt**2
    d2_mat[-1,-1] = -1/dt**2
    return d2_mat

def get_time_freq_func(f):
    '''
    Given a symmetric autocorrelation function evaluated at non-negative times (interval [0,t]), compute the periodic extension (interval (-t,t]) and its Fourier transform.
    
    Parameters
    ----------
    f : array_like
        The autocorrelation function evaluated at non-negative times.
        
    Returns
    -------
    array_like
        The periodic extension of the autocorrelation function.
    array_like
        The Fourier transform of the periodic extension.
    '''
    N = f.shape[-1]
    new_shape = np.array(f.shape)
    new_shape[-1] += N-2
    ft = np.zeros(new_shape)
    ft[...,:N] = f
    ft[...,N:] = f[...,-1:1:-1]
    fo = np.real(np.fft.fft(ft))
    return ft,fo

def smooth_func(f,dt,fcut=17,beta=1):
    '''
    Smooth an autocorrelation function by applying a low-pass filter.
    
    Parameters
    ----------
    f : array_like
        The autocorrelation function to be smoothed.
    dt : float
        Time step.
    fcut : float, optional
        Cutoff frequency for the low-pass filter. Default is 17.
    beta : float, optional
        Smoothing parameter for the low-pass filter. Default is 1.
        
    Returns
    -------
    array_like
        The smoothed autocorrelation function.
    '''
    N = f.shape[-1]
    _,fo = get_time_freq_func(f)
    fo *= 1/(1 + np.exp((np.abs(np.fft.fftfreq(2*(N-1),dt)) - fcut)*beta))
    return np.real(np.fft.ifft(fo))[...,:N]

def gauss_dmft(tau,muW,SigW,muH,SigH,M_fn,C_fn,Twrm,Tsav,dt,r0=None,Cr0=None):
    '''
    Solve for stationary rate moments by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    muW : array_like
        2D Array of mean total weights per connection type.
    SigW : array_like
        2D Array of variance of total weights per connection type.
    muH : array_like
        1D Array of mean external input per cell type.
    SigH : array_like
        1D Array of variance of external input per cell type.
    M_fn : function
        Function to compute the mean rates. It should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored.
    C_fn : function
        Function to compute the autocorrelation of the rates. It should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    r0 : array_like, optional
        Initial condition for the mean rates. If None, defaults to 1e-8 for all cell types.
    Cr0 : array_like, optional
        Initial condition for the autocorrelation function. If None, defaults to 1e2 for all cell types for all time lags.
        
    Returns
    -------
    array_like
        2D Array of the mean rates over time (shape: (Ntyp,Nint)).
    array_like
        3D Array of the autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding population converged.
    '''
    Ntyp = len(muH)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    r = np.zeros((Ntyp,Nint),dtype=np.float32)
    Cr = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if r0 is None:
        r0 = 1e-8*np.ones((Ntyp),dtype=np.float32)
    if Cr0 is None:
        Cr0 = 1e2*np.ones((Ntyp,1),dtype=np.float32)
        
    if SigH.ndim==1:
        SigH = SigH[:,None] * np.ones(Nclc)[None,:]
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    r[:,0] = r0
    
    NCr0 = Cr0.shape[1]
    if Nclc > NCr0:
        Cr[:,0,:NCr0] = Cr0
        Cr[:,0,NCr0:Nclc] = Cr0[:,-1:]
        Cr[:,:NCr0,0] = Cr0
        Cr[:,NCr0:Nclc,0] = Cr0[:,-1:]
    else:
        Cr[:,0,:Nclc] = Cr0[:,:Nclc]
        Cr[:,:Nclc,0] = Cr0[:,:Nclc]
        
    Mphi = np.empty((Ntyp,),dtype=np.float32)
    Cphi = np.empty((Ntyp,),dtype=np.float32)
    
    def drdt(ri,Sigii):
        mui = muW@ri + muH
        M_fn(mui,Sigii,Mphi)
        return tauinv*(-ri + Mphi)
    
    for i in range(Nint-1):
        Crii = Cr[:,i,i]
        Sigii = SigW@Crii + SigH[:,0]
        
        k1 = drdt(r[:,i]          ,Sigii)
        k2 = drdt(r[:,i]+0.5*dt*k1,Sigii)
        k3 = drdt(r[:,i]+0.5*dt*k2,Sigii)
        k4 = drdt(r[:,i]+    dt*k3,Sigii)
        
        r[:,i+1] = r[:,i] + dt/6*(k1+2*k2+2*k3+k4)
        ri = r[:,i]
        mui = muW@ri + muH
        
        if np.any(np.abs(r[:,i+1]) > 1e10) or np.any(np.isnan(r[:,i+1])):
            print("system diverged when integrating r")
            return r,Cr,False

        if i > Nclc-1:
            Cr[:,i+1,i-Nclc] = Cr[:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            ij_idx = np.fmin(i-j,Nsav-1)
            
            Crij = Cr[:,i,j]
            Sigij = SigW@Crij + SigH[:,ij_idx]
            C_fn(mui,Sigii,Sigij,Cphi)
            Cr[:,i+1,j+1] = Cr[:,i,j+1]+Cr[:,i+1,j]-Cr[:,i,j] +\
                dttauinv*(-Cr[:,i+1,j]-Cr[:,i,j+1]+2*Cr[:,i,j]) + dttauinv2*(-Cr[:,i,j]+Cphi)
                
            Cr[:,i+1,j+1] = np.maximum(Cr[:,i+1,j+1],ri**2)
            
            if np.any(np.abs(Cr[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cr[:,i+1,j+1])):
                print("system diverged when integrating Cr")
                return r,Cr,False
                
            Cr[:,j+1,i+1] = Cr[:,i+1,j+1]
            
        Ndiv = 5
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Cr_diag = each_diag(Cr)
    
    return r,Cr,\
        (np.max(Cr_diag[:,-Nsav:],axis=1)-np.min(Cr_diag[:,-Nsav:],axis=1))/\
            np.mean(Cr_diag[:,-Nsav:],axis=1) < 1e-3

def doub_gauss_dmft(tau,muW,SigW,muH,SigH,M_fns,C_fns,Twrm,Tsav,dt,r0=None,Cr0=None):
    '''
    Solve for stationary rate moments simultaneously without and with a perturbation by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    muW : array_like
        2D Array of mean total weights per connection type.
    SigW : array_like
        2D Array of variance of total weights per connection type.
    muH : array_like
        1D Array of mean external input per cell type.
    SigH : array_like
        1D Array of variance of external input per cell type.
    M_fns : list of functions
        List of two functions to compute the mean rates. Each function should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored. The first function is for the unperturbed system, and the second function is for the perturbed system.
    C_fns : list of functions
        List of two functions to compute the autocorrelation of the rates. Each function should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored. The first function is for the unperturbed system, and the second function is for the perturbed system.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    r0 : array_like, optional
        Initial condition for the mean rates. If None, defaults to 1e-8 for all cell types.
    Cr0 : array_like, optional
        Initial condition for the autocorrelation function. If None, defaults to 1e2 for all cell types for all time lags.
        
    Returns
    -------
    array_like
        2D Array of the mean rates over time (shape: (2*Ntyp,Nint)).
    array_like
        3D Array of the autocorrelation function over time (shape: (2*Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding population converged.
    '''
    Ntyp = len(muH)
    
    doub_tau = doub_vec(tau)
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muH = doub_vec(muH)
    doub_SigH = np.concatenate([SigH,SigH],axis=0)
    
    def doub_M(mui,Sigii,out):
        M_fns[0](mui[:Ntyp],Sigii[:Ntyp],out[:Ntyp])
        M_fns[1](mui[Ntyp:],Sigii[Ntyp:],out[Ntyp:])
        
    def doub_C(mui,Sigii,Sigij,out):
        C_fns[0](mui[:Ntyp],Sigii[:Ntyp],Sigij[:Ntyp],out[:Ntyp])
        C_fns[1](mui[Ntyp:],Sigii[Ntyp:],Sigij[Ntyp:],out[Ntyp:])
        
    if r0 is not None and len(r0) == Ntyp:
        r0 = doub_vec(r0)
    if Cr0 is not None and len(Cr0) == Ntyp:
        Cr0 = np.concatenate([Cr0,Cr0],axis=0)
        
    return gauss_dmft(doub_tau,doub_muW,doub_SigW,doub_muH,doub_SigH,doub_M,doub_C,Twrm,Tsav,dt,r0,Cr0)

def gauss_struct_dmft(tau,muWs,SigWs,muHs,SigHs,M_fn,C_fn,mu_fn,Sig_fn,Twrm,Tsav,dt,rs0,Crs0):
    '''
    Solve for stationary rate moments in a structured network by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    muWs : array_like
        Array of mean total weights. Shape depends on implementation of mu_fn and Sig_fn.
    SigWs : array_like
        Array of variance of total weights. Shape depends on implementation of mu_fn and Sig_fn.
    muHs : array_like
        Array of mean external input. Shape depends on implementation of mu_fn and Sig_fn.
    SigHs : array_like
        Array of variance of external input. Shape depends on implementation of mu_fn and Sig_fn.
    M_fn : function
        Function to compute the mean rates. It should take as input the 2D arrays of the mean and variance of the net input, and a 2D array to where the computed mean rates will be stored (inputs should have shape (Nsit,Ntyp)).
    C_fn : function
        Function to compute the autocorrelation of the rates. It should take as input the 2D arrays of the mean, variance, and covariance of the net inputs, and a 2D array to where the computed autocorrelation of the rates will be stored (inputs should have shape (Nsit,Ntyp)).
    mu_fn : function
        Function to compute the mean inputs. It's inputs should be (rs,muWs,muHs,mus), where rs is the 2D array of the mean rates at a given time (shape (Nsit,Ntyp)) and mus is the 2D array where the computed mean inputs at that time will be stored (shape (Nsit,Ntyp)).
    Sig_fn : function
        Function to compute the autocovariance of the inputs. It's inputs should be (Crs,SigWs,SigHs,Sigs), where Crs is the 2D array of the rate autocorrelations at a given time pair (shape (Nsit,Ntyp)) and Sigs is the 2D array where the computed autocovariance of the inputs at that time pair will be stored (shape (Nsit,Ntyp)).
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rs0 : array_like
        Initial condition for the mean rates. Not optional for this function, as no clear default is available.
    Crs0 : array_like
        Initial condition for the autocorrelation function. Not optional for this function, as no clear default is available.
        
    Returns
    -------
    array_like
        3D Array of the mean rates over time (shape: (Nsit,Ntyp,Nint)).
    array_like
        4D Array of the autocorrelation function over time (shape: (Nsit,Ntyp,Nint,Nint)).
    array_like
        3D Boolean array indicating whether the corresponding population converged.
    '''
    Nsit = muHs.shape[0]
    Ntyp = muHs.shape[1]
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    rs = np.zeros((Nsit,Ntyp,Nint),dtype=np.float32)
    Crs = np.zeros((Nsit,Ntyp,Nint,Nint),dtype=np.float32)
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    rs[:,:,0] = rs0
    
    NCr0 = Crs0.shape[2]
    if Nclc > NCr0:
        Crs[:,:,0,:NCr0] = Crs0
        Crs[:,:,0,NCr0:Nclc] = Crs0[:,:,-1:]
        Crs[:,:,:NCr0,0] = Crs0
        Crs[:,:,NCr0:Nclc,0] = Crs0[:,:,-1:]
    else:
        Crs[:,:,0,:Nclc] = Crs0[:,:,:Nclc]
        Crs[:,:,:Nclc,0] = Crs0[:,:,:Nclc]
        
    Mphis = np.empty((Nsit,Ntyp),dtype=np.float32)
    Cphis = np.empty((Nsit,Ntyp),dtype=np.float32)
    musi = np.empty((Nsit,Ntyp),dtype=np.float32)
    Sigsii = np.empty((Nsit,Ntyp),dtype=np.float32)
    Sigsij = np.empty((Nsit,Ntyp),dtype=np.float32)
    
    def drdt(rsi,Sigsii):
        mu_fn(rsi,muWs,muHs,musi)
        for sit_idx in range(Nsit):
            M_fn(musi[sit_idx],Sigsii[sit_idx],Mphis[sit_idx])
        return tauinv*(-rsi + Mphis)
    
    for i in range(Nint-1):
        Crsii = Crs[:,:,i,i]
        Sig_fn(Crsii,SigWs,SigHs,Sigsii)
            
        kb1 = drdt(rs[:,:,i]           ,Sigsii)
        kb2 = drdt(rs[:,:,i]+0.5*dt*kb1,Sigsii)
        kb3 = drdt(rs[:,:,i]+0.5*dt*kb2,Sigsii)
        kb4 = drdt(rs[:,:,i]+    dt*kb2,Sigsii)
        
        rs[:,:,i+1] = rs[:,:,i] + dt/6*(kb1+2*kb2+2*kb3+kb4)
        rsi = rs[:,:,i]
        mu_fn(rsi,muWs,muHs,musi)
        
        if np.any(np.abs(rs[:,:,i+1]) > 1e10) or np.any(np.isnan(rs[:,:,i+1])):
            print(musi)
            print(Sigsii)
            print("system diverged when integrating rb")
            return rs,Crs,False

        if i > Nclc-1:
            Crs[:,:,i+1,i-Nclc] = Crs[:,:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            Crsij = Crs[:,:,i,j]
            Sig_fn(Crsij,SigWs,SigHs,Sigsij)
            for sit_idx in range(Nsit):
                C_fn(musi[sit_idx],Sigsii[sit_idx],Sigsij[sit_idx],Cphis[sit_idx])
            Crs[:,:,i+1,j+1] = Crs[:,:,i,j+1]+Crs[:,:,i+1,j]-Crs[:,:,i,j] +\
                dttauinv*(-Crs[:,:,i+1,j]-Crs[:,:,i,j+1]+2*Crs[:,:,i,j]) + dttauinv2*(-Crs[:,:,i,j]+Cphis)
                
            Crs[:,:,i+1,j+1] = np.maximum(Crs[:,:,i+1,j+1],rsi**2)
            
            if np.any(np.abs(Crs[:,:,i+1,j+1]) > 1e10) or np.any(np.isnan(Crs[:,:,i+1,j+1])):
                print(musi)
                print(Sigsii)
                print(Sigsij)
                print("system diverged when integrating Crb")
                return rs,Crs,False
                
            Crs[:,:,j+1,i+1] = Crs[:,:,i+1,j+1]
            
        Ndiv = 5
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Crs_diag = each_diag(Crs)
    
    return rs,Crs,\
        (np.max(Crs_diag[:,:,-Nsav:],axis=-1)-np.min(Crs_diag[:,:,-Nsav:],axis=-1))/\
            np.mean(Crs_diag[:,:,-Nsav:],axis=-1) < 1e-3

def diff_gauss_dmft(tau,muW,SigW,muH,SigH,R_fn,Twrm,Tsav,dt,r,Cr,Cdr0=None,SigdW=None):
    '''
    Solve for stationary rate change autocorrelation given the rate moments without and with a perturbation by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    muW : array_like
        2D Array of mean total weights per connection type.
    SigW : array_like
        2D Array of variance of total weights per connection type.
    muH : array_like
        1D Array of mean external input per cell type.
    SigH : array_like
        1D Array of variance of external input per cell type.
    R_fn : function
        Function to compute the cross-correlation of the rates without and with the perturbation. It should take as input the 1D arrays of the unperturbed and perturbed means, variances, and the unperturbed-perturbed covariance of the net inputs, and a 1D array to where the computed cross-correlation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    r : array_like
        1D Doubled array of the stationary mean rates without and with the perturbation (shape: (2*Ntyp,)).
    Cr : array_like
        2D Doubled array of the stationary rate autocorrelations without and with the perturbation (shape: (2*Ntyp,Nsav)).
    Cdr0 : array_like, optional
        Initial condition for the rate change autocorrelation function. If None, defaults to 1e3 plus the squared difference of the mean rates.
    SigdW : array_like, optional
        2D Array of the variance of the total weight to be used in the computation of the rate change autocorrelation function. If None, defaults to SigW.
        
    Returns
    -------
    array_like
        3D Array of the rate change autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding population converged.
    '''
    Ntyp = len(muH)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    dr = r[Ntyp:] - r[:Ntyp]
    
    Cdr = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if Cdr0 is None:
        Cdr0 = dr.astype(np.float32)[:,None]**2 + 1e3
        
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    
    mu = doub_muW@r + doub_vec(muH)
    if SigH.ndim==1:
        Sig = doub_SigW@Cr + doub_vec(SigH)[:,None]
    else:
        Sig = doub_SigW@Cr + np.concatenate([SigH,SigH],axis=0)
    
    NCdr0 = Cdr0.shape[1]
    if Nclc > NCdr0:
        Cdr[:,0,:NCdr0] = Cdr0
        Cdr[:,0,NCdr0:Nclc] = Cdr0[:,-1:]
        Cdr[:,:NCdr0,0] = Cdr0
        Cdr[:,NCdr0:Nclc,0] = Cdr0[:,-1:]
    else:
        Cdr[:,0,:Nclc] = Cdr0[:,:Nclc]
        Cdr[:,:Nclc,0] = Cdr0[:,:Nclc]
        
    Rphi = np.empty((Ntyp,),dtype=np.float32)
    Cphi = Cr - (doub_vec(tau)**2 - dt*doub_vec(tau))[:,None] * np.einsum("ij,kj->ki",d2_stencil(Tsav,dt),Cr)
    
    if SigdW is None:
        SigdW = SigW
    
    for i in range(Nint-1):
        if i > Nclc-1:
            Cdr[:,i+1,i-Nclc] = Cdr[:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            ij_idx = np.fmin(i-j,Nsav-1)
            
            Cdrij = Cdr[:,i,j]
            Sigdij = SigdW@Cdrij
            
            kij = 0.5*(Sig[:Ntyp,ij_idx]+Sig[Ntyp:,ij_idx]-Sigdij)
            
            R_fn(mu[:Ntyp],mu[Ntyp:],Sig[:Ntyp,0],Sig[Ntyp:,0],kij,Rphi)
            
            Cdr[:,i+1,j+1] = Cdr[:,i,j+1]+Cdr[:,i+1,j]-Cdr[:,i,j] +\
                dttauinv*(-Cdr[:,i+1,j]-Cdr[:,i,j+1]+2*Cdr[:,i,j]) +\
                dttauinv2*(-Cdr[:,i,j]+Cphi[:Ntyp,ij_idx]+Cphi[Ntyp:,ij_idx]-2*Rphi)
                
            Cdr[:,i+1,j+1] = np.maximum(Cdr[:,i+1,j+1],dr**2)
            
            if np.any(np.abs(Cdr[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdr[:,i+1,j+1])):
                print("system diverged when integrating Cdr")
                return Cdr,False
                
            Cdr[:,j+1,i+1] = Cdr[:,i+1,j+1]
            
        Ndiv = 20
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Cdr_diag = each_diag(Cdr)
    
    return Cdr,\
        (np.max(Cdr_diag[:,-Nsav:],axis=1)-np.min(Cdr_diag[:,-Nsav:],axis=1))/\
            np.mean(Cdr_diag[:,-Nsav:],axis=1) < 1e-3

def diff_gauss_struct_dmft(tau,muWs,SigWs,muHs,SigHs,R_fn,mu_fn,Sig_fn,Sigd_fn,Twrm,Tsav,dt,rs,Crs,Cdrs0):
    '''
    Solve for stationary rate change autocorrelation given the rate moments without and with a perturbation in a structured network by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    muWs : array_like
        Array of mean total weights. Shape depends on implementation of R_fn.
    SigWs : array_like
        Array of variance of total weights. Shape depends on implementation of R_fn.
    muHs : array_like
        Array of mean external input. Shape depends on implementation of R_fn.
    SigHs : array_like
        Array of variance of external input. Shape depends on implementation of R_fn.
    R_fn : function
        Function to compute thecross-correlation of the rates without and with the perturbation. It should take as input the 2D arrays of the unperturbed and perturbed means, variances, and the unperturbed-perturbed covariance of the net inputs, and a 2D array to where the computed cross-correlation of the rates will be stored (inputs should have shape (Nsit,Ntyp)).
    mu_fn : function
        Function to compute the mean inputs. It's inputs should be (rs,muWs,muHs,mus), where rs is the 2D array of the mean rates at a given time (shape (Nsit,Ntyp)) and mus is the 2D array where the computed mean inputs at that time will be stored (shape (Nsit,Ntyp)).
    Sig_fn : function
        Function to compute the autocovariance of the inputs. It's inputs should be (Crs,SigWs,SigHs,Sigs), where Crs is the 2D array of the rate autocorrelations at a given time pair (shape (Nsit,Ntyp)) and Sigs is the 2D array where the computed autocovariance of the inputs at that time pair will be stored (shape (Nsit,Ntyp)).
    Sigd_fn : function
        Function to compute the autocovariance of the change in inputs. It's inputs should be (Cdrs,SigWs,Sigds), where Cdrs is the 2D array of the rate change autocorrelations at a given time pair (shape (Nsit,Ntyp)) and Sigds is the 2D array where the computed autocovariance of the input changes at that time pair will be stored (shape (Nsit,Ntyp)).
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rs : array_like
        2D Doubled array of the stationary mean rates without and with the perturbation (shape: (Nsit,2*Ntyp)).
    Crs : array_like
        3D Doubled array of the stationary rate autocorrelations without and with the perturbation (shape: (Nsit,2*Ntyp,Nsav)).
    Cdrs0 : array_like
        Initial condition for the rate change autocorrelation function. Not optional for this function, as no clear default is available.
        
    Returns
    -------
    array_like
        3D Array of the rate change autocorrelation function over time (shape: (Nsit,Ntyp,Nint,Nint)).
    array_like
        2D Boolean array indicating whether the corresponding population converged.
    '''
    Nsit = muHs.shape[0]
    Ntyp = muHs.shape[1]
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    drs = rs[:,Ntyp:] - rs[:,:Ntyp]
    
    Cdrs = np.zeros((Nsit,Ntyp,Nint,Nint),dtype=np.float32)
    
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    doub_muWs = doub_mat(muWs)
    doub_SigWs = doub_mat(SigWs)
    doub_muHs = doub_vec(muHs)
    doub_SigHs = doub_vec(SigHs)
    
    mus = np.zeros_like(rs)
    Sigs = np.zeros_like(Crs)
    mu_fn(rs,doub_muWs,doub_muHs,mus)
    Sig_fn(Crs,doub_SigWs,doub_SigHs,Sigs)
    
    NCdr0 = Cdrs0.shape[1]
    if Nclc > NCdr0:
        Cdrs[:,:,0,:NCdr0] = Cdrs0
        Cdrs[:,:,0,NCdr0:Nclc] = Cdrs0[:,:,-1:]
        Cdrs[:,:,:NCdr0,0] = Cdrs0
        Cdrs[:,:,NCdr0:Nclc,0] = Cdrs0[:,:,-1:]
    else:
        Cdrs[:,:,0,:Nclc] = Cdrs0[:,:,:Nclc]
        Cdrs[:,:,:Nclc,0] = Cdrs0[:,:,:Nclc]
        
    Rphis = np.empty((Nsit,Ntyp),dtype=np.float32)
    Cphis = Crs - (doub_vec(tau)**2 - dt*doub_vec(tau))[None,:,None] * np.einsum("ij,klj->kli",d2_stencil(Tsav,dt),Crs)
    Sigdsij = np.empty((Nsit,Ntyp),dtype=np.float32)
    
    for i in range(Nint-1):
        if i > Nclc-1:
            Cdrs[:,:,i+1,i-Nclc] = Cdrs[:,:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            ij_idx = np.fmin(i-j,Nsav-1)
            
            Cdrsij = Cdrs[:,:,i,j]
            Sigd_fn(Cdrsij,SigWs,Sigdsij)
            
            ksij = 0.5*(Sigs[:,:Ntyp,ij_idx]+Sigs[:,Ntyp:,ij_idx]-Sigdsij)
            
            for sit_idx in range(Nsit):
                R_fn(mus[sit_idx,:Ntyp],mus[sit_idx,Ntyp:],Sigs[sit_idx,:Ntyp,0],Sigs[sit_idx,Ntyp:,0],
                     ksij[sit_idx],Rphis[sit_idx])
            
            Cdrs[:,:,i+1,j+1] = Cdrs[:,:,i,j+1]+Cdrs[:,:,i+1,j]-Cdrs[:,:,i,j] +\
                dttauinv*(-Cdrs[:,:,i+1,j]-Cdrs[:,:,i,j+1]+2*Cdrs[:,:,i,j]) +\
                dttauinv2*(-Cdrs[:,:,i,j]+Cphis[:,:Ntyp,ij_idx]+Cphis[:,Ntyp:,ij_idx]-2*Rphis)
                
            Cdrs[:,:,i+1,j+1] = np.maximum(Cdrs[:,:,i+1,j+1],drs**2)
            
            if np.any(np.abs(Cdrs[:,:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdrs[:,:,i+1,j+1])):
                print(Sigdsij)
                print("system diverged when integrating Cdrb")
                return Cdrs,False
                
            Cdrs[:,:,j+1,i+1] = Cdrs[:,:,i+1,j+1]
            
        Ndiv = 20
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Cdrs_diag = each_diag(Cdrs)
    
    return Cdrs,\
        (np.max(Cdrs_diag[:,:,-Nsav:],axis=-1)-np.min(Cdrs_diag[:,:,-Nsav:],axis=-1))/\
            np.mean(Cdrs_diag[:,:,-Nsav:],axis=-1) < 1e-3

def sparse_dmft(tau,W,K,H,CVH,M_fn,C_fn,Twrm,Tsav,dt,r0=None,Cr0=None):
    '''
    Solve for stationary rate moments in a sparse network by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    H : array_like
        1D Array of mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    M_fn : function
        Function to compute the mean rates. It should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored.
    C_fn : function
        Function to compute the autocorrelation of the rates. It should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    r0 : array_like, optional
        Initial condition for the mean rates. If None, defaults to 1e-8 for all cell types.
    Cr0 : array_like, optional
        Initial condition for the autocorrelation function. If None, defaults to 1e2 for all cell types for all time lags.
        
    Returns
    -------
    array_like
        2D Array of the mean rates over time (shape: (Ntyp,Nint)).
    array_like
        3D Array of the autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding population converged.
    '''
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muH = tau*H
    SigH = (muH*CVH)**2
    
    return gauss_dmft(tau,muW,SigW,muH,SigH,M_fn,C_fn,Twrm,Tsav,dt,r0=r0,Cr0=Cr0)

def doub_sparse_dmft(tau,W,K,H,CVH,M_fns,C_fns,Twrm,Tsav,dt,r0=None,Cr0=None):
    '''
    Solve for stationary rate moments simultaneously without and with a perturbation in a sparse network by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    H : array_like
        1D Array of mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    M_fns : list of functions
        List of two functions to compute the mean rates. Each function should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored. The first function is for the unperturbed system, and the second function is for the perturbed system.
    C_fns : list of functions
        List of two functions to compute the autocorrelation of the rates. Each function should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored. The first function is for the unperturbed system, and the second function is for the perturbed system.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    r0 : array_like, optional
        Initial condition for the mean rates. If None, defaults to 1e-8 for all cell types.
    Cr0 : array_like, optional
        Initial condition for the autocorrelation function. If None, defaults to 1e2 for all cell types for all time lags.
        
    Returns
    -------
    array_like
        2D Array of the mean rates over time (shape: (2*Ntyp,Nint)).
    array_like
        3D Array of the autocorrelation function over time (shape: (2*Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding population converged.
    '''
    Ntyp = len(H)
    
    doub_tau = doub_vec(tau)
    doub_W = doub_mat(W)
    doub_K = doub_vec(K)
    doub_H = doub_vec(H)
    
    def doub_M(mui,Sigii,out):
        M_fns[0](mui[:Ntyp],Sigii[:Ntyp],out[:Ntyp])
        M_fns[1](mui[Ntyp:],Sigii[Ntyp:],out[Ntyp:])
        
    def doub_C(mui,Sigii,Sigij,out):
        C_fns[0](mui[:Ntyp],Sigii[:Ntyp],Sigij[:Ntyp],out[:Ntyp])
        C_fns[1](mui[Ntyp:],Sigii[Ntyp:],Sigij[Ntyp:],out[Ntyp:])
        
    return sparse_dmft(doub_tau,doub_W,doub_K,doub_H,CVH,doub_M,doub_C,Twrm,Tsav,dt,r0,Cr0)

def diff_sparse_dmft(tau,W,K,H,CVH,R_fn,Twrm,Tsav,dt,r,Cr,Cdr0=None):
    '''
    Solve for stationary rate change autocorrelation given the rate moments without and with a perturbation in a sparse network by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    H : array_like
        1D Array of mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    R_fn : function
        Function to compute the cross-correlation of the rates without and with the perturbation. It should take as input the 1D arrays of the unperturbed and perturbed means, variances, and the unperturbed-perturbed covariance of the net inputs, and a 1D array to where the computed cross-correlation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    r : array_like
        1D Doubled array of the stationary mean rates without and with the perturbation (shape: (2*Ntyp,)).
    Cr : array_like
        2D Doubled array of the stationary rate autocorrelations without and with the perturbation (shape: (2*Ntyp,Nsav)).
    Cdr0 : array_like, optional
        Initial condition for the rate change autocorrelation function. If None, defaults to 1e3 plus the squared difference of the mean rates.
        
    Returns
    -------
    array_like
        3D Array of the rate change autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding population converged.
    '''
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muH = tau*H
    SigH = (muH*CVH)**2
    
    return diff_gauss_dmft(tau,muW,SigW,muH,SigH,R_fn,Twrm,Tsav,dt,r,Cr,Cdr0=Cdr0)
    
def get_solve_width(sa,L=180):
    '''
    Get a function to solve for the width of the wrapped Gaussian kernel given the normalized baseline-subtracted value at a predefined orientation value relative to the stimulus.
    
    Parameters
    ----------
    sa : float
        The predefined auxiliary orientation location relative to the stimulus.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    function
        Function to solve for the width of the wrapped Gaussian kernel given the normalized baseline-subtracted value at a predefined orientation value relative to the stimulus.
    '''
    widths = np.linspace(1,L*3/4,135)
    fbars = basesubwrapnorm(sa,widths,L)
    max_fbar = np.max(fbars)
    min_fbar = np.min(fbars)
    widths_vs_fbars_itp = interp1d(fbars,widths)
    def solve_widths(fbar):
        return widths_vs_fbars_itp(np.fmax(min_fbar,np.fmin(max_fbar,fbar)))
    return solve_widths
    
def get_2feat_solve_width(sa,dori=45,L=180):
    '''
    Get a function to solve for the widths of two identical but offset wrapped Gaussian kernel given the normalized baseline-subtracted value at a predefined orientation value relative to the right-most stimulus.
    
    Parameters
    ----------
    sa : float
        The predefined auxiliary orientation location relative to the right-most stimulus.
    dori : int, optional
        The offset between the two identical wrapped Gaussian kernel centers. Default is 45.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    function
        Function to solve for the widths of two identical but offset wrapped Gaussian kernel given the normalized baseline-subtracted value at a predefined orientation value relative to the right-most stimulus.
    '''
    widths = np.linspace(1,L*3/4,135)
    fbars = (basesubwrapnorm(sa,widths,L) + basesubwrapnorm(sa+dori,widths,L)) /\
        (1 + basesubwrapnorm(dori,widths,L))
    max_fbar = np.max(fbars)
    min_fbar = np.min(fbars)
    widths_vs_fbars_itp = interp1d(fbars,widths)
    def solve_widths(fbar):
        return widths_vs_fbars_itp(np.fmax(min_fbar,np.fmin(max_fbar,fbar)))
    return solve_widths

def unstruct_fact(s,L=180):
    '''
    Compute the convolution of a uniform distribution with a normalized baseline-subtracted wrapped Gaussian.
    
    Parameters
    ----------
    s : float
        The width of the normalized baseline-subtracted wrapped Gaussian.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    float
        The convolution of a uniform distribution with the normalized baseline-subtracted wrapped Gaussian of width s.
    '''
    return (1/L-wrapnormdens(L/2,s,L))/(wrapnormdens(0,s,L)-wrapnormdens(L/2,s,L))

def struct_fact(x,sconv,sorig,L=180):
    '''
    Compute the convolution of a wrapped Gaussian distribution with a normalized baseline-subtracted wrapped Gaussian.
    
    Parameters
    ----------
    x : float
        The orientation value at which to compute the convolution relative to the normalized baseline-subtracted wrapped Gaussian center.
    sconv : float
        The width of the convolution of the wrapped Gaussian distribution with the Gaussian component of the normalized baseline-subtracted wrapped Gaussian.
    sorig : float
        The width of the Gaussian component of the normalized baseline-subtracted wrapped Gaussian.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    float
        The convolution of a wrapped Gaussian distribution with the normalized baseline-subtracted wrapped Gaussian of width sorig, where the width of the Gaussian part of the convolution has width sconv, evaluated at x.
    '''
    return (wrapnormdens(x,sconv,L)-wrapnormdens(L/2,sorig,L))/\
        (wrapnormdens(0,sorig,L)-wrapnormdens(L/2,sorig,L))
    
def inv_overlap(xs,ss,L=180):
    '''
    Compute the inverse of the overlap matrix for a mixture of normalized baseline-subtraced wrapped Gaussians.
    
    Parameters
    ----------
    xs : array_like
        1D Array of the normalized baseline-subtracted wrapped Gaussian centers.
    ss : array_like
        1D Array of the widths of the normalized baseline-subtracted wrapped Gaussians.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    array_like
        2D Array of the inverse of the overlap matrix for the mixture of normalized baseline-subtracted wrapped Gaussians.
    '''
    overlap_mat = basesubwrapnorm(xs[None,:,None]-xs[None,None,:],ss[:,None,:],L)
    return np.linalg.inv(overlap_mat)

def sparse_ring_dmft(tau,W,K,Hb,Hm,CVH,sW,sH,sa,M_fn,C_fn,Twrm,Tsav,dt,
                rb0=None,ra0=None,rm0=None,Crb0=None,Cra0=None,Crm0=None,Kb=None,L=180):
    '''
    Solve for stationary three-site rate moments in a sparse ring network assuming baseline-plus-Gaussian rate moments by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    Hb : array_like
        1D Array of baseline mean external input per cell type.
    Hm : array_like
        1D Array of matched mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    sW : float or array_like
        The connection probability width. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the connection probability width per presynaptic cell type. If a 2D array then the elements should list the connection probability width per connection type.
    sH : float or array_like
        The external input width. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the external input width per cell type.
    M_fn : function
        Function to compute the mean rates. It should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored.
    C_fn : function
        Function to compute the autocorrelation of the rates. It should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rb0 : array_like, optional
        Initial condition for the baseline mean rates. If None, defaults to 1 for all cell types.
    ra0 : array_like, optional
        Initial condition for the auxiliary mean rates. If None, defaults to 2 for all cell types.
    rm0 : array_like, optional
        Initial condition for the matched mean rates. If None, defaults to 5 for all cell types.
    Crb0 : array_like, optional
        Initial condition for the baseline autocorrelation function. If None, defaults to 1e2 for all cell types for all time lags.
    Cra0 : array_like, optional
        Initial condition for the auxliary autocorrelation function. If None, defaults to 4e2 for all cell types for all time lags.
    Crm0 : array_like, optional
        Initial condition for the matched autocorrelation function. If None, defaults to 25e2 for all cell types for all time lags.
    Kb : float or array_like, optional
        The mean unstructured out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type. If None, defaults to K.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    array_like
        2D Array of the baseline mean rates over time (shape: (Ntyp,Nint)).
    array_like
        2D Array of the auxiliary mean rates over time (shape: (Ntyp,Nint)).
    array_like
        2D Array of the matched mean rates over time (shape: (Ntyp,Nint)).
    array_like
        3D Array of the baseline autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        3D Array of the auxiliary autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        3D Array of the matched autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding baseline population converged.
    array_like
        1D Boolean array indicating whether the corresponding auxiliary population converged.
    array_like
        1D Boolean array indicating whether the corresponding matched population converged.
    '''
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    
    if rb0 is None:
        rb0 = 1*np.ones((Ntyp),dtype=np.float32)
    if ra0 is None:
        ra0 = 2*np.ones((Ntyp),dtype=np.float32)
    if rm0 is None:
        rm0 = 5*np.ones((Ntyp),dtype=np.float32)
    if Crb0 is None:
        Crb0 = 1e2*np.ones((Ntyp,1),dtype=np.float32)
    if Cra0 is None:
        Cra0 = 4e2*np.ones((Ntyp,1),dtype=np.float32)
    if Crm0 is None:
        Crm0 = 25e2*np.ones((Ntyp,1),dtype=np.float32)
    
    solve_width = get_solve_width(sa,L)
    
    sW2 = sW**2
    
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muHb = tau*Hb
    SigHb = (muHb*CVH)**2
    muHa = tau*(Hb+(Hm-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*CVH)**2
    muHm = tau*Hm
    SigHm = (muHm*CVH)**2
    
    muWs = np.concatenate([muWb[None,:,:],muW[None,:,:]],0)
    SigWs = np.concatenate([SigWb[None,:,:],SigW[None,:,:]],0)
    muHs = np.concatenate([muHb[None,:],muHa[None,:],muHm[None,:]],0)
    SigHs = np.concatenate([SigHb[None,:],SigHa[None,:],SigHm[None,:]],0)
    rs0 = np.concatenate([rb0[None,:],ra0[None,:],rm0[None,:]],0)
    Crs0 = np.concatenate([Crb0[None,:,:],Cra0[None,:,:],Crm0[None,:,:]],0)
    
    def mu_fn(rsi,muWs,muHs,musi):
        '''
        rbi = rsi[0]
        rai = rsi[1]
        rmi = rsi[2]
        
        muWb = muWs[0]
        muW = muWs[1]
        
        muHb = muHs[0]
        muHa = muHs[1]
        muHm = muHs[2]
        
        mubi = musi[0]
        muai = musi[1]
        mumi = musi[2]
        '''
        sri = solve_width((rsi[1]-rsi[0])/(rsi[2]-rsi[0]))
        sWri = np.sqrt(sW2+sri**2)
        rmsubbi = rsi[2] - rsi[0]
        musi[0] = (muWs[1]+muWs[0])@rsi[0] + (unstruct_fact(sri,L)*muWs[0])@rmsubbi + muHs[0]
        musi[1] = musi[0] + (struct_fact(sa,sWri,sri,L)*muWs[1])@rmsubbi + muHs[1]-muHs[0]
        musi[2] = musi[0] + (struct_fact(0,sWri,sri,L)*muWs[1])@rmsubbi + muHs[2]-muHs[0]
        musi[0] = musi[0] + (struct_fact(L/2,sWri,sri,L)*muWs[1])@rmsubbi
        
    def Sig_fn(Crsi,SigWs,SigHs,Sigsi):
        '''
        Crbi = Crsi[0]
        Crai = Crsi[1]
        Crmi = Crsi[2]
        
        SigWb = SigWs[0]
        SigW = SigWs[1]
        
        SigHb = SigHs[0]
        SigHa = SigHs[1]
        SigHm = SigHs[2]
        
        Sigbi = Sigsi[0]
        Sigai = Sigsi[1]
        Sigmi = Sigsi[2]
        '''
        sCri = solve_width((Crsi[1]-Crsi[0])/(Crsi[2]-Crsi[0]))
        sWCri = np.sqrt(sW2+sCri**2)
        Crmsubbi = Crsi[2] - Crsi[0]
        Sigsi[0] = (SigWs[1]+SigWs[0])@Crsi[0] + (unstruct_fact(sCri,L)*SigWs[0])@Crmsubbi + SigHs[0]
        Sigsi[1] = Sigsi[0] + (struct_fact(sa,sWCri,sCri,L)*SigWs[1])@Crmsubbi + SigHs[1]-SigHs[0]
        Sigsi[2] = Sigsi[0] + (struct_fact(0,sWCri,sCri,L)*SigWs[1])@Crmsubbi + SigHs[2]-SigHs[0]
        Sigsi[0] = Sigsi[0] + (struct_fact(L/2,sWCri,sCri,L)*SigWs[1])@Crmsubbi
                
    rs,Crs,convs = gauss_struct_dmft(tau,muWs,SigWs,muHs,SigHs,M_fn,C_fn,mu_fn,Sig_fn,Twrm,Tsav,dt,rs0,Crs0)
    
    return rs[0],rs[1],rs[2],Crs[0],Crs[1],Crs[2],convs[0],convs[1],convs[2]

def sparse_2feat_ring_dmft(tau,W,K,Hb,Hm,CVH,sW,sH,sa,M_fn,C_fn,Twrm,Tsav,dt,
                rb0=None,ra0=None,rm0=None,Crb0=None,Cra0=None,Crm0=None,Kb=None,dori=45,L=180):
    '''
    Solve for stationary three-site rate moments in a sparse ring network driven by two equal contrast visual stimuli assuming baseline-plus-identical-Gaussian-mixture rate moments by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    Hb : array_like
        1D Array of baseline mean external input per cell type.
    Hm : array_like
        1D Array of matched mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    sW : float or array_like
        The connection probability width. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the connection probability width per presynaptic cell type. If a 2D array then the elements should list the connection probability width per connection type.
    sH : float or array_like
        The external input width. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the external input width per cell type.
    M_fn : function
        Function to compute the mean rates. It should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored.
    C_fn : function
        Function to compute the autocorrelation of the rates. It should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rb0 : array_like, optional
        Initial condition for the baseline mean rates. If None, defaults to 1 for all cell types.
    ra0 : array_like, optional
        Initial condition for the auxiliary mean rates. If None, defaults to 2 for all cell types.
    rm0 : array_like, optional
        Initial condition for the matched mean rates. If None, defaults to 5 for all cell types.
    Crb0 : array_like, optional
        Initial condition for the baseline autocorrelation function. If None, defaults to 1e2 for all cell types for all time lags.
    Cra0 : array_like, optional
        Initial condition for the auxliary autocorrelation function. If None, defaults to 4e2 for all cell types for all time lags.
    Crm0 : array_like, optional
        Initial condition for the matched autocorrelation function. If None, defaults to 25e2 for all cell types for all time lags.
    Kb : float or array_like, optional
        The mean unstructured out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type. If None, defaults to K.
    dori : int, optional
        The offset between the two identical wrapped Gaussian kernel centers. Default is 45.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    array_like
        2D Array of the baseline mean rates over time (shape: (Ntyp,Nint)).
    array_like
        2D Array of the auxiliary mean rates over time (shape: (Ntyp,Nint)).
    array_like
        2D Array of the matched mean rates over time (shape: (Ntyp,Nint)).
    array_like
        3D Array of the baseline autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        3D Array of the auxiliary autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        3D Array of the matched autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding baseline population converged.
    array_like
        1D Boolean array indicating whether the corresponding auxiliary population converged.
    array_like
        1D Boolean array indicating whether the corresponding matched population converged.
    '''
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    
    if rb0 is None:
        rb0 = 1*np.ones((Ntyp),dtype=np.float32)
    if ra0 is None:
        ra0 = 2*np.ones((Ntyp),dtype=np.float32)
    if rm0 is None:
        rm0 = 5*np.ones((Ntyp),dtype=np.float32)
    if Crb0 is None:
        Crb0 = 1e2*np.ones((Ntyp,1),dtype=np.float32)
    if Cra0 is None:
        Cra0 = 4e2*np.ones((Ntyp,1),dtype=np.float32)
    if Crm0 is None:
        Crm0 = 25e2*np.ones((Ntyp,1),dtype=np.float32)
    
    xmatched = np.array([0,-dori])
    solve_width = get_2feat_solve_width(sa,dori,L)
    
    sW2 = sW**2
        
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muHb = tau*Hb
    SigHb = (muHb*CVH)**2
    muHa = tau*(Hb+(Hm-Hb)*(basesubwrapnorm(sa,sH,L)+basesubwrapnorm(dori+sa,sH,L)))
    SigHa = (muHa*CVH)**2
    muHm = tau*(Hm+(Hm-Hb)*basesubwrapnorm(dori,sH,L))
    SigHm = (muHm*CVH)**2
    
    muWs = np.concatenate([muWb[None,:,:],muW[None,:,:]],0)
    SigWs = np.concatenate([SigWb[None,:,:],SigW[None,:,:]],0)
    muHs = np.concatenate([muHb[None,:],muHa[None,:],muHm[None,:]],0)
    SigHs = np.concatenate([SigHb[None,:],SigHa[None,:],SigHm[None,:]],0)
    rs0 = np.concatenate([rb0[None,:],ra0[None,:],rm0[None,:]],0)
    Crs0 = np.concatenate([Crb0[None,:,:],Cra0[None,:,:],Crm0[None,:,:]],0)
    
    def mu_fn(rsi,muWs,muHs,musi):
        '''
        rbi = rsi[0]
        rai = rsi[1]
        rmi = rsi[2]
        
        muWb = muWs[0]
        muW = muWs[1]
        
        muHb = muHs[0]
        muHa = muHs[1]
        muHm = muHs[2]
        
        mubi = musi[0]
        muai = musi[1]
        mumi = musi[2]
        '''
        sri = solve_width((rsi[1]-rsi[0])/(rsi[2]-rsi[0]))
        rOinv = np.sum(inv_overlap(xmatched,sri[:,None])[:,:,0],-1)
        sWri = np.sqrt(sW2+sri**2)
        rmsubbi = (rsi[2] - rsi[0])*rOinv
        musi[0] = (muWs[1]+muWs[0])@rsi[0] + (unstruct_fact(sri,L)*muWs[0])@rmsubbi + muHs[0]
        musi[1] = musi[0] + ((struct_fact(sa,sWri,sri,L)+struct_fact(sa+dori,sWri,sri,L))*muWs[1])@rmsubbi +\
            muHs[1]-muHs[0]
        musi[2] = musi[0] + ((struct_fact(0,sWri,sri,L)+struct_fact(dori,sWri,sri,L))*muWs[1])@rmsubbi + muHs[2]-muHs[0]
        musi[0] = musi[0] + (2*struct_fact(L/2,sWri,sri,L)*muWs[1])@rmsubbi
        
    def Sig_fn(Crsi,SigWs,SigHs,Sigsi):
        '''
        Crbi = Crsi[0]
        Crai = Crsi[1]
        Crmi = Crsi[2]
        
        SigWb = SigWs[0]
        SigW = SigWs[1]
        
        SigHb = SigHs[0]
        SigHa = SigHs[1]
        SigHm = SigHs[2]
        
        Sigbi = Sigsi[0]
        Sigai = Sigsi[1]
        Sigmi = Sigsi[2]
        '''
        sCri = solve_width((Crsi[1]-Crsi[0])/(Crsi[2]-Crsi[0]))
        CrOinv = np.sum(inv_overlap(xmatched,sCri[:,None])[:,:,0],-1)
        sWCri = np.sqrt(sW2+sCri**2)
        Crmsubbi = (Crsi[2] - Crsi[0])*CrOinv
        Sigsi[0] = (SigWs[1]+SigWs[0])@Crsi[0] + (unstruct_fact(sCri,L)* SigWs[0])@Crmsubbi + SigHs[0]
        Sigsi[1] = Sigsi[0] + ((struct_fact(sa,sWCri,sCri,L)+\
            struct_fact(sa+dori,sWCri,sCri,L))*SigWs[1])@Crmsubbi + SigHs[1]-SigHs[0]
        Sigsi[2] = Sigsi[0] + ((struct_fact(0,sWCri,sCri,L)+\
            struct_fact(dori,sWCri,sCri,L))*SigWs[1])@Crmsubbi + SigHs[2]-SigHs[0]
        Sigsi[0] = Sigsi[0] + (2*struct_fact(L/2,sWCri,sCri,L)*SigWs[1])@Crmsubbi
                
    rs,Crs,convs = gauss_struct_dmft(tau,muWs,SigWs,muHs,SigHs,M_fn,C_fn,mu_fn,Sig_fn,Twrm,Tsav,dt,rs0,Crs0)
    
    return rs[0],rs[1],rs[2],Crs[0],Crs[1],Crs[2],convs[0],convs[1],convs[2]

def doub_sparse_ring_dmft(tau,W,K,Hb,Hm,CVH,sW,sH,sa,M_fns,C_fns,Twrm,Tsav,dt,
                     rb0=None,ra0=None,rm0=None,Crb0=None,Cra0=None,Crm0=None,Kb=None,L=180):
    '''
    Solve for stationary three-site rate moments simultaneously without and with a perturbation in a sparse ring network assuming baseline-plus-Gaussian rate moments by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    Hb : array_like
        1D Array of baseline mean external input per cell type.
    Hm : array_like
        1D Array of matched mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    sW : float or array_like
        The connection probability width. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the connection probability width per presynaptic cell type. If a 2D array then the elements should list the connection probability width per connection type.
    sH : float or array_like
        The external input width. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the external input width per cell type.
    M_fn : function
        Function to compute the mean rates. It should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored.
    C_fn : function
        Function to compute the autocorrelation of the rates. It should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rb0 : array_like, optional
        Initial condition for the baseline mean rates. If None, defaults to 1 for all cell types.
    ra0 : array_like, optional
        Initial condition for the auxiliary mean rates. If None, defaults to 2 for all cell types.
    rm0 : array_like, optional
        Initial condition for the matched mean rates. If None, defaults to 5 for all cell types.
    Crb0 : array_like, optional
        Initial condition for the baseline autocorrelation function. If None, defaults to 1e2 for all cell types for all time lags.
    Cra0 : array_like, optional
        Initial condition for the auxliary autocorrelation function. If None, defaults to 4e2 for all cell types for all time lags.
    Crm0 : array_like, optional
        Initial condition for the matched autocorrelation function. If None, defaults to 25e2 for all cell types for all time lags.
    Kb : float or array_like, optional
        The mean unstructured out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type. If None, defaults to K.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    array_like
        2D Array of the baseline mean rates over time (shape: (2*Ntyp,Nint)).
    array_like
        2D Array of the auxiliary mean rates over time (shape: (2*Ntyp,Nint)).
    array_like
        2D Array of the matched mean rates over time (shape: (2*Ntyp,Nint)).
    array_like
        3D Array of the baseline autocorrelation function over time (shape: (2*Ntyp,Nint,Nint)).
    array_like
        3D Array of the auxiliary autocorrelation function over time (shape: (2*Ntyp,Nint,Nint)).
    array_like
        3D Array of the matched autocorrelation function over time (shape: (2*Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding baseline population converged.
    array_like
        1D Boolean array indicating whether the corresponding auxiliary population converged.
    array_like
        1D Boolean array indicating whether the corresponding matched population converged.
    '''
    if Kb is None:
        doub_Kb = None
    else:
        doub_Kb = doub_vec(Kb)
        
    Ntyp = len(Hb)
    
    doub_tau = doub_vec(tau)
    doub_W = doub_mat(W)
    doub_K = doub_vec(K)
    doub_Hb = doub_vec(Hb)
    doub_Hm = doub_vec(Hm)
    doub_sW = doub_mat(sW)
    doub_sH = doub_vec(sH)
    
    def doub_M(mui,Sigii,out):
        M_fns[0](mui[:Ntyp],Sigii[:Ntyp],out[:Ntyp])
        M_fns[1](mui[Ntyp:],Sigii[Ntyp:],out[Ntyp:])
        
    def doub_C(mui,Sigii,Sigij,out):
        C_fns[0](mui[:Ntyp],Sigii[:Ntyp],Sigij[:Ntyp],out[:Ntyp])
        C_fns[1](mui[Ntyp:],Sigii[Ntyp:],Sigij[Ntyp:],out[Ntyp:])
        
    return sparse_ring_dmft(doub_tau,doub_W,doub_K,doub_Hb,doub_Hm,CVH,doub_sW,doub_sH,sa,doub_M,doub_C,Twrm,Tsav,dt,
                      rb0,ra0,rm0,Crb0,Cra0,Crm0,Kb=doub_Kb,L=L)

def doub_sparse_2feat_ring_dmft(tau,W,K,Hb,Hm,CVH,sW,sH,sa,M_fns,C_fns,Twrm,Tsav,dt,
                     rb0=None,ra0=None,rm0=None,Crb0=None,Cra0=None,Crm0=None,Kb=None,dori=45,L=180):
    '''
    Solve for stationary three-site rate moments simultaneously without and with a perturbation in a sparse ring network driven by two equal contrast visual stimuli assuming baseline-plus-identical-Gaussian-mixture rate moments by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    Hb : array_like
        1D Array of baseline mean external input per cell type.
    Hm : array_like
        1D Array of matched mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    sW : float or array_like
        The connection probability width. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the connection probability width per presynaptic cell type. If a 2D array then the elements should list the connection probability width per connection type.
    sH : float or array_like
        The external input width. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the external input width per cell type.
    M_fn : function
        Function to compute the mean rates. It should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored.
    C_fn : function
        Function to compute the autocorrelation of the rates. It should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rb0 : array_like, optional
        Initial condition for the baseline mean rates. If None, defaults to 1 for all cell types.
    ra0 : array_like, optional
        Initial condition for the auxiliary mean rates. If None, defaults to 2 for all cell types.
    rm0 : array_like, optional
        Initial condition for the matched mean rates. If None, defaults to 5 for all cell types.
    Crb0 : array_like, optional
        Initial condition for the baseline autocorrelation function. If None, defaults to 1e2 for all cell types for all time lags.
    Cra0 : array_like, optional
        Initial condition for the auxliary autocorrelation function. If None, defaults to 4e2 for all cell types for all time lags.
    Crm0 : array_like, optional
        Initial condition for the matched autocorrelation function. If None, defaults to 25e2 for all cell types for all time lags.
    Kb : float or array_like, optional
        The mean unstructured out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type. If None, defaults to K.
    dori : int, optional
        The offset between the two identical wrapped Gaussian kernel centers. Default is 45.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    array_like
        2D Array of the baseline mean rates over time (shape: (2*Ntyp,Nint)).
    array_like
        2D Array of the auxiliary mean rates over time (shape: (2*Ntyp,Nint)).
    array_like
        2D Array of the matched mean rates over time (shape: (2*Ntyp,Nint)).
    array_like
        3D Array of the baseline autocorrelation function over time (shape: (2*Ntyp,Nint,Nint)).
    array_like
        3D Array of the auxiliary autocorrelation function over time (shape: (2*Ntyp,Nint,Nint)).
    array_like
        3D Array of the matched autocorrelation function over time (shape: (2*Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding baseline population converged.
    array_like
        1D Boolean array indicating whether the corresponding auxiliary population converged.
    array_like
        1D Boolean array indicating whether the corresponding matched population converged.
    '''
    if Kb is None:
        doub_Kb = None
    else:
        doub_Kb = doub_vec(Kb)
        
    Ntyp = len(Hb)
    
    doub_tau = doub_vec(tau)
    doub_W = doub_mat(W)
    doub_K = doub_vec(K)
    doub_Hb = doub_vec(Hb)
    doub_Hm = doub_vec(Hm)
    doub_sW = doub_mat(sW)
    doub_sH = doub_vec(sH)
    
    def doub_M(mui,Sigii,out):
        M_fns[0](mui[:Ntyp],Sigii[:Ntyp],out[:Ntyp])
        M_fns[1](mui[Ntyp:],Sigii[Ntyp:],out[Ntyp:])
        
    def doub_C(mui,Sigii,Sigij,out):
        C_fns[0](mui[:Ntyp],Sigii[:Ntyp],Sigij[:Ntyp],out[:Ntyp])
        C_fns[1](mui[Ntyp:],Sigii[Ntyp:],Sigij[Ntyp:],out[Ntyp:])
        
    return sparse_2feat_ring_dmft(doub_tau,doub_W,doub_K,doub_Hb,doub_Hm,CVH,doub_sW,doub_sH,sa,doub_M,doub_C,
                                  Twrm,Tsav,dt,rb0,ra0,rm0,Crb0,Cra0,Crm0,Kb=doub_Kb,dori=dori,L=L)

def diff_sparse_ring_dmft(tau,W,K,Hb,Hm,CVH,sW,sH,sa,R_fn,Twrm,Tsav,dt,rb,ra,rm,Crb,Cra,Crm,
                     Cdrb0=None,Cdra0=None,Cdrm0=None,Kb=None,L=180):
    '''
    Solve for stationary rate change autocorrelation given the rate moments without and with a perturbation in a sparse ring network assuming baseline-plus-Gaussian rate moments by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    Hb : array_like
        1D Array of baseline mean external input per cell type.
    Hm : array_like
        1D Array of matched mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    sW : float or array_like
        The connection probability width. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the connection probability width per presynaptic cell type. If a 2D array then the elements should list the connection probability width per connection type.
    sH : float or array_like
        The external input width. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the external input width per cell type.
    R_fn : function
        Function to compute the cross-correlation of the rates without and with the perturbation. It should take as input the 1D arrays of the unperturbed and perturbed means, variances, and the unperturbed-perturbed covariance of the net inputs, and a 1D array to where the computed cross-correlation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rb : array_like
        1D Doubled array of the stationary baseline mean rates without and with the perturbation (shape: (2*Ntyp,)).
    ra : array_like
        1D Doubled array of the stationary auxiliary mean rates without and with the perturbation (shape: (2*Ntyp,)).
    rm : array_like
        1D Doubled array of the stationary matched mean rates without and with the perturbation (shape: (2*Ntyp,)).
    Crb : array_like
        2D Doubled array of the stationary baseline rate autocorrelations without and with the perturbation (shape: (2*Ntyp,Nsav)).
    Cra : array_like
        2D Doubled array of the stationary auxiliary rate autocorrelations without and with the perturbation (shape: (2*Ntyp,Nsav)).
    Crm : array_like
        2D Doubled array of the stationary matched rate autocorrelations without and with the perturbation (shape: (2*Ntyp,Nsav)).
    Cdrb0 : array_like, optional
        Initial condition for the baseline rate change autocorrelation function. If None, defaults to 1e3 plus the squared difference of the baseline mean rates.
    Cdra0 : array_like, optional
        Initial condition for the auxiliary rate change autocorrelation function. If None, defaults to 2e3 plus the squared difference of the auxiliary mean rates.
    Cdrm0 : array_like, optional
        Initial condition for the matched rate change autocorrelation function. If None, defaults to 5e3 plus the squared difference of the matched mean rates.
    Kb : float or array_like, optional
        The mean unstructured out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type. If None, defaults to K.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    array_like
        3D Array of the baseline rate change autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        3D Array of the auxiliary rate change autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        3D Array of the matched rate change autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding baseline population converged.
    array_like
        1D Boolean array indicating whether the corresponding auxiliary population converged.
    array_like
        1D Boolean array indicating whether the corresponding matched population converged.
    '''
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    
    if Cdrb0 is None:
        Cdrb0 = (rb[Ntyp:] - rb[:Ntyp]).astype(np.float32)[:,None]**2 + 1e3
    if Cdra0 is None:
        Cdra0 = (ra[Ntyp:] - ra[:Ntyp]).astype(np.float32)[:,None]**2 + 2e3
    if Cdrm0 is None:
        Cdrm0 = (rm[Ntyp:] - rm[:Ntyp]).astype(np.float32)[:,None]**2 + 5e3
    
    solve_width = get_solve_width(sa,L)
    
    sW2 = sW**2
    
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    
    muHb = tau*Hb
    SigHb = (muHb*CVH)**2
    muHa = tau*(Hb+(Hm-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*CVH)**2
    muHm = tau*Hm
    SigHm = (muHm*CVH)**2
    
    muWs = np.concatenate([muWb[None,:,:],muW[None,:,:]],0)
    SigWs = np.concatenate([SigWb[None,:,:],SigW[None,:,:]],0)
    muHs = np.concatenate([muHb[None,:],muHa[None,:],muHm[None,:]],0)
    SigHs = np.concatenate([SigHb[None,:],SigHa[None,:],SigHm[None,:]],0)
    rs = np.concatenate([rb[None,:],ra[None,:],rm[None,:]],0)
    Crs = np.concatenate([Crb[None,:,:],Cra[None,:,:],Crm[None,:,:]],0)
    Cdrs0 = np.concatenate([Cdrb0[None,:,:],Cdra0[None,:,:],Cdrm0[None,:,:]],0)
    
    def mu_fn(rs,muWs,muHs,mus):
        '''
        rbi = rs[0]
        rai = rs[1]
        rmi = rs[2]
        
        muWb = muWs[0]
        muW = muWs[1]
        
        muHb = muHs[0]
        muHa = muHs[1]
        muHm = muHs[2]
        
        mubi = mus[0]
        muai = mus[1]
        mumi = mus[2]
        '''
        sr = solve_width((rs[1]-rs[0])/(rs[2]-rs[0]))
        sWr = np.sqrt(doub_mat(sW2)+sr**2)
        rmsubb = rs[2] - rs[0]
        mus[0] = (muWs[1]+muWs[0])@rs[0] + (unstruct_fact(sr,L)*muWs[0])@rmsubb + muHs[0]
        mus[1] = mus[0] + (struct_fact(sa,sWr,sr,L)*muWs[1])@rmsubb + muHs[1]-muHs[0]
        mus[2] = mus[0] + (struct_fact(0,sWr,sr,L)*muWs[1])@rmsubb + muHs[2]-muHs[0]
        mus[0] = mus[0] + (struct_fact(L/2,sWr,sr,L)*muWs[1])@rmsubb
        
    def Sig_fn(Crs,SigWs,SigHs,Sigs):
        '''
        Crbi = Crs[0]
        Crai = Crs[1]
        Crmi = Crs[2]
        
        SigWb = SigWs[0]
        SigW = SigWs[1]
        
        SigHb = SigHs[0]
        SigHa = SigHs[1]
        SigHm = SigHs[2]
        
        Sigbi = Sigs[0]
        Sigai = Sigs[1]
        Sigmi = Sigs[2]
        '''
        sCr = solve_width((Crs[1]-Crs[0])/(Crs[2]-Crs[0]))
        sWCr = np.sqrt(doub_mat(sW2)[:,:,None]+sCr[None,:,:]**2)
        Crmsubb = Crs[2] - Crs[0]
        Sigs[0] = each_matmul((SigWs[1]+SigWs[0])[:,:,None],Crs[0]) +\
            each_matmul(unstruct_fact(sCr,L)*SigWs[0][:,:,None],Crmsubb) + SigHs[0][:,None]
        Sigs[1] = Sigs[0] + each_matmul(struct_fact(sa,sWCr,sCr,L)*SigWs[1][:,:,None],Crmsubb) +\
            (SigHs[1]-SigHs[0])[:,None]
        Sigs[2] = Sigs[0] + each_matmul(struct_fact(0,sWCr,sCr,L)*SigWs[1][:,:,None],Crmsubb) +\
            (SigHs[2]-SigHs[0])[:,None]
        Sigs[0] = Sigs[0] + each_matmul(struct_fact(L/2,sWCr,sCr,L)*SigWs[1][:,:,None],Crmsubb)
        
    def Sigd_fn(Cdrsi,SigWs,Sigdsi):
        '''
        Cdrbi = Cdrsi[0]
        Cdrai = Cdrsi[1]
        Cdrmi = Cdrsi[2]
        
        SigWb = SigWs[0]
        SigW = SigWs[1]
        
        Sigdbi = Sigdsi[0]
        Sigdai = Sigdsi[1]
        Sigdmi = Sigdsi[2]
        '''
        sCdri = solve_width((Cdrsi[1]-Cdrsi[0])/(Cdrsi[2]-Cdrsi[0]))
        sWCdri = np.sqrt(sW2+sCdri**2)
        Cdrmsubbi = Cdrsi[2] - Cdrsi[0]
        Sigdsi[0] = (SigWs[1]+SigWs[0])@Cdrsi[0] + (unstruct_fact(sCdri,L)*SigWs[0])@Cdrmsubbi
        Sigdsi[1] = Sigdsi[0] + (struct_fact(sa,sWCdri,sCdri,L)*SigWs[1])@Cdrmsubbi
        Sigdsi[2] = Sigdsi[0] + (struct_fact(0,sWCdri,sCdri,L)*SigWs[1])@Cdrmsubbi
        Sigdsi[0] = Sigdsi[0] + (struct_fact(L/2,sWCdri,sCdri,L)*SigWs[1])@Cdrmsubbi
        
    Cdrs,convs = diff_gauss_struct_dmft(tau,muWs,SigWs,muHs,SigHs,R_fn,mu_fn,Sig_fn,Sigd_fn,Twrm,Tsav,dt,rs,Crs,Cdrs0)
    
    return Cdrs[0],Cdrs[1],Cdrs[2],convs[0],convs[1],convs[2]

def diff_sparse_2feat_ring_dmft(tau,W,K,Hb,Hm,CVH,sW,sH,sa,R_fn,Twrm,Tsav,dt,rb,ra,rm,Crb,Cra,Crm,
                                Cdrb0=None,Cdra0=None,Cdrm0=None,Kb=None,dori=45,L=180):
    '''
    Solve for stationary rate change autocorrelation given the rate moments without and with a perturbation in a sparse ring network driven by two equal contrast visual stimuli assuming baseline-plus-identical-Gaussian-mixture rate moments by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    Hb : array_like
        1D Array of baseline mean external input per cell type.
    Hm : array_like
        1D Array of matched mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    sW : float or array_like
        The connection probability width. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the connection probability width per presynaptic cell type. If a 2D array then the elements should list the connection probability width per connection type.
    sH : float or array_like
        The external input width. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the external input width per cell type.
    R_fn : function
        Function to compute the cross-correlation of the rates without and with the perturbation. It should take as input the 1D arrays of the unperturbed and perturbed means, variances, and the unperturbed-perturbed covariance of the net inputs, and a 1D array to where the computed cross-correlation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rb : array_like
        1D Doubled array of the stationary baseline mean rates without and with the perturbation (shape: (2*Ntyp,)).
    ra : array_like
        1D Doubled array of the stationary auxiliary mean rates without and with the perturbation (shape: (2*Ntyp,)).
    rm : array_like
        1D Doubled array of the stationary matched mean rates without and with the perturbation (shape: (2*Ntyp,)).
    Crb : array_like
        2D Doubled array of the stationary baseline rate autocorrelations without and with the perturbation (shape: (2*Ntyp,Nsav)).
    Cra : array_like
        2D Doubled array of the stationary auxiliary rate autocorrelations without and with the perturbation (shape: (2*Ntyp,Nsav)).
    Crm : array_like
        2D Doubled array of the stationary matched rate autocorrelations without and with the perturbation (shape: (2*Ntyp,Nsav)).
    Cdrb0 : array_like, optional
        Initial condition for the baseline rate change autocorrelation function. If None, defaults to 1e3 plus the squared difference of the baseline mean rates.
    Cdra0 : array_like, optional
        Initial condition for the auxiliary rate change autocorrelation function. If None, defaults to 2e3 plus the squared difference of the auxiliary mean rates.
    Cdrm0 : array_like, optional
        Initial condition for the matched rate change autocorrelation function. If None, defaults to 5e3 plus the squared difference of the matched mean rates.
    Kb : float or array_like, optional
        The mean unstructured out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type. If None, defaults to K.
    dori : int, optional
        The offset between the two identical wrapped Gaussian kernel centers. Default is 45.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    array_like
        3D Array of the baseline rate change autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        3D Array of the auxiliary rate change autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        3D Array of the matched rate change autocorrelation function over time (shape: (Ntyp,Nint,Nint)).
    array_like
        1D Boolean array indicating whether the corresponding baseline population converged.
    array_like
        1D Boolean array indicating whether the corresponding auxiliary population converged.
    array_like
        1D Boolean array indicating whether the corresponding matched population converged.
    '''
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    Nsav = round(Tsav/dt)+1
    
    if Cdrb0 is None:
        Cdrb0 = (rb[Ntyp:] - rb[:Ntyp]).astype(np.float32)[:,None]**2 + 1e3
    if Cdra0 is None:
        Cdra0 = (ra[Ntyp:] - ra[:Ntyp]).astype(np.float32)[:,None]**2 + 2e3
    if Cdrm0 is None:
        Cdrm0 = (rm[Ntyp:] - rm[:Ntyp]).astype(np.float32)[:,None]**2 + 5e3
    
    xmatched = np.array([0,-dori])
    solve_width = get_2feat_solve_width(sa,dori,L)
    
    sW2 = sW**2
    
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    
    muHb = tau*Hb
    SigHb = (muHb*CVH)**2
    muHa = tau*(Hb+(Hm-Hb)*(basesubwrapnorm(sa,sH,L)+basesubwrapnorm(dori+sa,sH,L)))
    SigHa = (muHa*CVH)**2
    muHm = tau*(Hm+(Hm-Hb)*basesubwrapnorm(dori,sH,L))
    SigHm = (muHm*CVH)**2
    
    muWs = np.concatenate([muWb[None,:,:],muW[None,:,:]],0)
    SigWs = np.concatenate([SigWb[None,:,:],SigW[None,:,:]],0)
    muHs = np.concatenate([muHb[None,:],muHa[None,:],muHm[None,:]],0)
    SigHs = np.concatenate([SigHb[None,:],SigHa[None,:],SigHm[None,:]],0)
    rs = np.concatenate([rb[None,:],ra[None,:],rm[None,:]],0)
    Crs = np.concatenate([Crb[None,:,:],Cra[None,:,:],Crm[None,:,:]],0)
    Cdrs0 = np.concatenate([Cdrb0[None,:,:],Cdra0[None,:,:],Cdrm0[None,:,:]],0)
    
    def mu_fn(rs,muWs,muHs,mus):
        '''
        rb = rs[0]
        ra = rs[1]
        rm = rs[2]
        
        muWb = muWs[0]
        muW = muWs[1]
        
        muHb = muHs[0]
        muHa = muHs[1]
        muHm = muHs[2]
        
        mub = mus[0]
        mua = mus[1]
        mum = mus[2]
        '''
        sr = solve_width((rs[1]-rs[0])/(rs[2]-rs[0]))
        rOinv = np.sum(inv_overlap(xmatched,sr[:,None])[:,:,0],-1)
        sWr = np.sqrt(doub_mat(sW2)+sr**2)
        rmsubb = (rs[2] - rs[0])*rOinv
        mus[0] = (muWs[1]+muWs[0])@rs[0] + (unstruct_fact(sr,L)*muWs[0])@rmsubb + muHs[0]
        mus[1] = mus[0] + ((struct_fact(sa,sWr,sr,L)+struct_fact(sa+dori,sWr,sr,L))*muWs[1])@rmsubb +\
            muHs[1]-muHs[0]
        mus[2] = mus[0] + ((struct_fact(0,sWr,sr,L)+struct_fact(dori,sWr,sr,L))*muWs[1])@rmsubb + muHs[2]-muHs[0]
        mus[0] = mus[0] + (2*struct_fact(L/2,sWr,sr,L)*muWs[1])@rmsubb
        
    def Sig_fn(Crs,SigWs,SigHs,Sigs):
        '''
        Crbi = Crs[0]
        Crai = Crs[1]
        Crmi = Crs[2]
        
        SigWb = SigWs[0]
        SigW = SigWs[1]
        
        SigHb = SigHs[0]
        SigHa = SigHs[1]
        SigHm = SigHs[2]
        
        Sigbi = Sigs[0]
        Sigai = Sigs[1]
        Sigmi = Sigs[2]
        '''
        sCr = solve_width((Crs[1]-Crs[0])/(Crs[2]-Crs[0]))
        CrOinv = np.sum(inv_overlap(xmatched,sCr.flatten()[:,None])[:,:,0],-1).reshape(-1,Nsav)
        sWCr = np.sqrt(doub_mat(sW2)[:,:,None]+sCr[None,:,:]**2)
        Crmsubb = (Crs[2] - Crs[0])*CrOinv
        Sigs[0] = each_matmul((SigWs[1]+SigWs[0])[:,:,None],Crs[0]) +\
            each_matmul(unstruct_fact(sCr,L)*SigWs[0][:,:,None],Crmsubb) + SigHs[0][:,None]
        Sigs[1] = Sigs[0] + each_matmul((struct_fact(sa,sWCr,sCr,L)+\
            struct_fact(sa+dori,sWCr,sCr,L))*SigWs[1][:,:,None],Crmsubb) + (SigHs[1]-SigHs[0])[:,None]
        Sigs[2] = Sigs[0] + each_matmul((struct_fact(0,sWCr,sCr,L)+\
            struct_fact(dori,sWCr,sCr,L))*SigWs[1][:,:,None],Crmsubb) + (SigHs[2]-SigHs[0])[:,None]
        Sigs[0] = Sigs[0] + each_matmul(2*struct_fact(L/2,sWCr,sCr,L)*SigWs[1][:,:,None],Crmsubb)
        
    def Sigd_fn(Cdrsi,SigWs,Sigdsi):
        '''
        Cdrbi = Cdrsi[0]
        Cdrai = Cdrsi[1]
        Cdrmi = Cdrsi[2]
        
        SigWb = SigWs[0]
        SigW = SigWs[1]
        
        Sigdbi = Sigdsi[0]
        Sigdai = Sigdsi[1]
        Sigdmi = Sigdsi[2]
        '''
        sCdri = solve_width((Cdrsi[1]-Cdrsi[0])/(Cdrsi[2]-Cdrsi[0]))
        CdrOinv = np.sum(inv_overlap(xmatched,sCdri[:,None])[:,:,0],-1)
        sWCdri = np.sqrt(sW2+sCdri**2)
        Cdrmsubbi = (Cdrsi[2] - Cdrsi[0])*CdrOinv
        Sigdsi[0] = (SigWs[1]+SigWs[0])@Cdrsi[0] + (unstruct_fact(sCdri,L)* SigWs[0])@Cdrmsubbi
        Sigdsi[1] = Sigdsi[0] + ((struct_fact(sa,sWCdri,sCdri,L)+\
            struct_fact(sa+dori,sWCdri,sCdri,L))*SigWs[1])@Cdrmsubbi
        Sigdsi[2] = Sigdsi[0] + ((struct_fact(0,sWCdri,sCdri,L)+\
            struct_fact(dori,sWCdri,sCdri,L))*SigWs[1])@Cdrmsubbi
        Sigdsi[0] = Sigdsi[0] + (2*struct_fact(L/2,sWCdri,sCdri,L)*SigWs[1])@Cdrmsubbi
        
    Cdrs,convs = diff_gauss_struct_dmft(tau,muWs,SigWs,muHs,SigHs,R_fn,mu_fn,Sig_fn,Sigd_fn,Twrm,Tsav,dt,rs,Crs,Cdrs0)
    
    return Cdrs[0],Cdrs[1],Cdrs[2],convs[0],convs[1],convs[2]

def sparse_full_ring_dmft(tau,W,K,Hb,Hm,CVH,sW,sH,M_fn,C_fn,Twrm,Tsav,dt,
                          rs0=None,Crs0=None,Kb=None,L=180,Nori=20):
    '''
    Solve for stationary three-site rate moments in a sparse discrete ring network by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    Hb : array_like
        1D Array of baseline mean external input per cell type.
    Hm : array_like
        1D Array of matched mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    sW : float or array_like
        The connection probability width. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the connection probability width per presynaptic cell type. If a 2D array then the elements should list the connection probability width per connection type.
    sH : float or array_like
        The external input width. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the external input width per cell type.
    M_fn : function
        Function to compute the mean rates. It should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored.
    C_fn : function
        Function to compute the autocorrelation of the rates. It should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rs0 : array_like, optional
        Initial condition for the mean rates. Should be a 2D array of shape (Nori,Ntyp). If None, defaults to 1+4*basesubwrapnorm(oris,15) for all cell types.
    Crs0 : array_like, optional
        Initial condition for the autocorrelation function. Should be a 3D array of shape (Nori,Ntyp,...). If None, defaults to 1e2+24e2*basesubwrapnorm(oris,15) for all cell types for all time lags.
    Kb : float or array_like, optional
        The mean unstructured out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type. If None, defaults to K.
    L : int, optional
        Size of periodic dimension. Default is 180.
    Nori : int, optional
        Number of discrete orientation sites. Default is 20.
        
    Returns
    -------
    array_like
        3D Array of the mean rates over time (shape: (Nori,Ntyp,Nint)).
    array_like
        4D Array of the autocorrelation function over time (shape: (Nori,Ntyp,Nint,Nint)).
    array_like
        2D Boolean array indicating whether the corresponding population converged.
    '''
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    
    oris = np.arange(Nori)/Nori * L
    
    if rs0 is None:
        rs0 = (1+4*basesubwrapnorm(oris,15))[:,None]*np.ones((Ntyp),dtype=np.float32)[None,:]
    if Crs0 is None:
        Crs0 = (1e2+24e2*basesubwrapnorm(oris,15))[:,None,None]*np.ones((Ntyp,1),dtype=np.float32)[None,:,:]
    
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,None,:,None],sW[None,:,None,:],L)
    
    muWs = (muWb[None,:,None,:] + muW[None,:,None,:]*2*np.pi*kerns) / Nori
    SigWs = (SigWb[None,:,None,:] + SigW[None,:,None,:]*2*np.pi*kerns) / Nori
    
    muHs = tau*(Hb[None,:]+(Hm-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:],L))
    SigHs = (muHs*CVH)**2
    
    def mu_fn(rsi,muWs,muHs,musi):
        musi[:] = np.einsum("ijkl,kl->ij",muWs,rsi) + muHs
    
    def Sig_fn(Crsi,SigWs,SigHs,Sigsi):
        Sigsi[:] = np.einsum("ijkl,kl->ij",SigWs,Crsi) + SigHs
                
    return gauss_struct_dmft(tau,muWs,SigWs,muHs,SigHs,M_fn,C_fn,mu_fn,Sig_fn,Twrm,Tsav,dt,rs0,Crs0)

def doub_sparse_full_ring_dmft(tau,W,K,Hb,Hm,CVH,sW,sH,M_fns,C_fns,Twrm,Tsav,dt,
                               rs0=None,Crs0=None,Kb=None,L=180,Nori=20):
    '''
    Solve for stationary three-site rate moments simultaneously without and with a perturbation in a sparse discrete ring network by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    Hb : array_like
        1D Array of baseline mean external input per cell type.
    Hm : array_like
        1D Array of matched mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    sW : float or array_like
        The connection probability width. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the connection probability width per presynaptic cell type. If a 2D array then the elements should list the connection probability width per connection type.
    sH : float or array_like
        The external input width. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the external input width per cell type.
    M_fn : function
        Function to compute the mean rates. It should take as input the 1D arrays of the mean and variance of the net input, and a 1D array to where the computed mean rates will be stored.
    C_fn : function
        Function to compute the autocorrelation of the rates. It should take as input the 1D arrays of the mean, variance, and covariance of the net inputs, and a 1D array to where the computed autocorrelation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rs0 : array_like, optional
        Initial condition for the mean rates. Should be a 2D array of shape (Nori,Ntyp). If None, defaults to 1+4*basesubwrapnorm(oris,15) for all cell types.
    Crs0 : array_like, optional
        Initial condition for the autocorrelation function. Should be a 3D array of shape (Nori,Ntyp,...). If None, defaults to 1e2+24e2*basesubwrapnorm(oris,15) for all cell types for all time lags.
    Kb : float or array_like, optional
        The mean unstructured out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type. If None, defaults to K.
    L : int, optional
        Size of periodic dimension. Default is 180.
    Nori : int, optional
        Number of discrete orientation sites. Default is 20.
        
    Returns
    -------
    array_like
        3D Array of the mean rates over time (shape: (Nori,2*Ntyp,Nint)).
    array_like
        4D Array of the autocorrelation function over time (shape: (Nori,2*Ntyp,Nint,Nint)).
    array_like
        2D Boolean array indicating whether the corresponding population converged.
    '''
    if Kb is None:
        doub_Kb = None
    else:
        doub_Kb = doub_vec(Kb)
        
    Ntyp = len(Hb)
    
    doub_tau = doub_vec(tau)
    doub_W = doub_mat(W)
    doub_K = doub_vec(K)
    doub_Hb = doub_vec(Hb)
    doub_Hm = doub_vec(Hm)
    doub_sW = np.block([[sW,sW],[sW,sW]])
    doub_sH = doub_vec(sH)
    
    def doub_M(mui,Sigii,out):
        M_fns[0](mui[:Ntyp],Sigii[:Ntyp],out[:Ntyp])
        M_fns[1](mui[Ntyp:],Sigii[Ntyp:],out[Ntyp:])
        
    def doub_C(mui,Sigii,Sigij,out):
        C_fns[0](mui[:Ntyp],Sigii[:Ntyp],Sigij[:Ntyp],out[:Ntyp])
        C_fns[1](mui[Ntyp:],Sigii[Ntyp:],Sigij[Ntyp:],out[Ntyp:])
        
    return sparse_full_ring_dmft(doub_tau,doub_W,doub_K,doub_Hb,doub_Hm,CVH,doub_sW,doub_sH,doub_M,doub_C,Twrm,Tsav,dt,
                                 rs0,Crs0,Kb=doub_Kb,L=L,Nori=Nori)

def diff_sparse_full_ring_dmft(tau,W,K,Hb,Hm,CVH,sW,sH,R_fn,Twrm,Tsav,dt,rs,Crs,
                               Cdrs0=None,Kb=None,L=180,Nori=20):
    '''
    Solve for stationary rate change autocorrelation given the rate moments without and with a perturbation in a sparse discrete ring network by evolving the system in time using the DMFT equations.
    
    Parameters
    ----------
    tau : array_like
        1D Array of time constants per cell type.
    W : array_like
        2D Array of synaptic efficacies per connection type.
    K : float or array_like
        The structured mean out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type.
    Hb : array_like
        1D Array of baseline mean external input per cell type.
    Hm : array_like
        1D Array of matched mean external input per cell type.
    CVH : float or array_like
        The coefficient of variation of the external inputs. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the CV of the input per cell type.
    sW : float or array_like
        The connection probability width. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the connection probability width per presynaptic cell type. If a 2D array then the elements should list the connection probability width per connection type.
    sH : float or array_like
        The external input width. If a scalar, then it is assumed to be the value for all cell types. If a 1D array then the elements should list the external input width per cell type.
    R_fn : function
        Function to compute the cross-correlation of the rates without and with the perturbation. It should take as input the 1D arrays of the unperturbed and perturbed means, variances, and the unperturbed-perturbed covariance of the net inputs, and a 1D array to where the computed cross-correlation of the rates will be stored.
    Twrm : float
        Time over which to warm up the system.
    Tsav : float
        Time over which to save the autocorrelation function.
    dt : float
        Time step.
    rs : array_like
        2D Doubled array of the stationary mean rates without and with the perturbation (shape: (Nori,2*Ntyp)).
    Crs : array_like
        3D Doubled array of the stationary rate autocorrelations without and with the perturbation (shape: (Nori,2*Ntyp,Nsav)).
    Cdrs0 : array_like, optional
        Initial condition for the rate change autocorrelation function. Should be a 3D array of shape (Nori,Ntyp,...). If None, defaults to 1e3+4e3*basesubwrapnorm(oris,15) plus the squared difference of the mean rates.
    Kb : float or array_like, optional
        The mean unstructured out-degree. If a scalar, then it is assumed to be the value for all connection types. If a 1D array then the elements should list the mean out-degree per presynaptic cell type. If a 2D array then the elements should list the mean out-degree per connection type. If None, defaults to K.
    L : int, optional
        Size of periodic dimension. Default is 180.
        
    Returns
    -------
    array_like
        4D Array of the rate change autocorrelation function over time (shape: (Nori,Ntyp,Nint,Nint)).
    array_like
        2D Boolean array indicating whether the corresponding population converged.
    '''
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    
    oris = np.arange(Nori)/Nori * L
    
    if Cdrs0 is None:
        Cdrs0 = (rs[:,Ntyp:] - rs[:,:Ntyp]).astype(np.float32)[:,:,None]**2 +\
            (1e3+4e3*basesubwrapnorm(oris,15))[:,None,None]
    
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,None,:,None],sW[None,:,None,:],L)
    
    muWs = (muWb[None,:,None,:] + muW[None,:,None,:]*2*np.pi*kerns) / Nori
    SigWs = (SigWb[None,:,None,:] + SigW[None,:,None,:]*2*np.pi*kerns) / Nori
    
    muHs = tau*(Hb[None,:]+(Hm-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:],L))
    SigHs = (muHs*CVH)**2
    
    def mu_fn(rsi,muWs,muHs,musi):
        musi[:] = np.einsum("ijkl,kl->ij",muWs,rsi) + muHs
    
    def Sig_fn(Crsi,SigWs,SigHs,Sigsi):
        Sigsi[:] = np.einsum("ijkl,klm->ijm",SigWs,Crsi) + SigHs[:,:,None]
        
    def Sigd_fn(Cdrsi,SigWs,Sigdsi):
        Sigdsi[:] = np.einsum("ijkl,kl->ij",SigWs,Cdrsi)
    
    return diff_gauss_struct_dmft(tau,muWs,SigWs,muHs,SigHs,R_fn,mu_fn,Sig_fn,Sigd_fn,Twrm,Tsav,dt,rs,Crs,Cdrs0)

def run_two_stage_dmft(prms,rX,CVh,res_dir,rc,Twrm,Tsav,dt,return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms["K"]
    J = prms["J"]
    beta = prms["beta"]
    gE = prms["gE"]
    gI = prms["gI"]
    hE = prms["hE"]
    hI = prms["hI"]
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = np.array([K,K/4],dtype=np.float32)
    H = rX*K*J*np.array([hE,hI/beta],dtype=np.float32)
    CVH = CVh
    
    muH = tau*H
    SigH = (muH*CVH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms["L"],prms["CVL"])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])
    
    start = time.process_time()
    
    full_r,full_Cr,conv = doub_sparse_dmft(tau,W,Ks,H,CVH,[base_M,opto_M],[base_C,opto_C],Twrm,Tsav,dt)

    print("integrating first stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    r = full_r[:,-1]
    Cr = full_Cr[:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    
    mu = doub_muW@r + doub_vec(muH)
    Sig = doub_SigW@Cr + doub_vec(SigH)[:,None]

    start = time.process_time()

    full_Cdr,convd = diff_sparse_dmft(tau,W,Ks,H,CVH,diff_R,Twrm,Tsav,dt,r,Cr)

    print("integrating second stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    dr = r[:2] - r[2:]
    Cdr = full_Cdr[:,-1,-1:-Nsav-1:-1]

    dmu = mu[:2] - mu[2:]

    Sigd = SigW@Cdr
    
    res_dict = {}
    
    res_dict["r"] = r
    res_dict["Cr"] = Cr
    res_dict["dr"] = dr
    res_dict["Cdr"] = Cdr
    
    res_dict["mu"] = mu
    res_dict["Sig"] = Sig
    res_dict["dmu"] = dmu
    res_dict["Sigd"] = Sigd

    res_dict["conv"] = conv
    res_dict["convd"] = convd
    
    if return_full:
        res_dict["full_r"] = full_r
        res_dict["full_Cr"] = full_Cr
        res_dict["full_Cdr"] = full_Cdr
    
    return res_dict

def run_decoupled_two_site_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,L=180,
                                struct_dict=None,which="both",return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms["K"]
    SoriE = prms["SoriE"]
    SoriI = prms["SoriI"]
    SoriF = prms["SoriF"]
    J = prms["J"]
    beta = prms["beta"]
    gE = prms["gE"]
    gI = prms["gI"]
    hE = prms["hE"]
    hI = prms["hI"]
    basefrac = prms.get("basefrac",0)
    baseinp = prms.get("baseinp",0)
    baseprob = prms.get("baseprob",0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =    (1-basefrac)*(1-baseprob) *np.array([K,K/4],dtype=np.float32)
    Kbs =(1-(1-basefrac)*(1-baseprob))*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+(1-(1-basefrac)*(1-baseinp))*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hm = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    CVH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    
    sW2 = sW**2
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    sr = struct_dict["sr"]
    sCr = struct_dict["sCr"][:,-1]
    
    sWr = np.sqrt(sW2+sr**2)
    sWCr = np.sqrt(sW2+sCr**2)
    
    muWbb = (1 - struct_fact(180/2,sWr,sr,180)) * muW + (1 - unstruct_fact(sr,L)) * muWb
    muWbm = struct_fact(180/2,sWr,sr,180) * muW + unstruct_fact(sr,L) * muWb
    muWmb = (1 - struct_fact(0,sWr,sr,180)) * muW + (1 - unstruct_fact(sr,L)) * muWb
    muWmm = struct_fact(0,sWr,sr,180) * muW + unstruct_fact(sr,L) * muWb

    SigWbb = (1 - struct_fact(180/2,sWCr,sCr,180)) * SigW + (1 - unstruct_fact(sCr,L)) * SigWb
    SigWbm = struct_fact(180/2,sWCr,sCr,180) * SigW + unstruct_fact(sCr,L) * SigWb
    SigWmb = (1 - struct_fact(0,sWCr,sCr,180)) * SigW + (1 - unstruct_fact(sCr,L)) * SigWb
    SigWmm = struct_fact(0,sWCr,sCr,180) * SigW + unstruct_fact(sCr,L) * SigWb
    
    muHb = tau*Hb + muWbm@struct_dict.get("rm",0)
    SigHb = ((tau*Hb*CVH)**2)[:,None] + SigWbm@struct_dict.get("Crm",0)
    muHm = tau*Hm + muWmb@struct_dict.get("rb",0)
    SigHm = ((tau*Hm*CVH)**2)[:,None] + SigWmb@struct_dict.get("Crb",0)
    
    Norig = SigHb.shape[1]
    if Norig!=Nsav:
        temp = SigHb.copy()
        SigHb = np.zeros((2,Nsav))
        SigHb[:,:Norig] = temp
        SigHb[:,Norig:] = temp[:,-1:]
        
        temp = SigHm.copy()
        SigHm = np.zeros((2,Nsav))
        SigHm[:,:Norig] = temp
        SigHm[:,Norig:] = temp[:,-1:]
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms["L"],prms["CVL"])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    start = time.process_time()
    
    if which=="base":
        full_rb,full_Crb,convb = gauss_dmft(tau,muWbb,SigWbb,muHb,SigHb,base_M,base_C,Twrm,Tsav,dt)
        full_rm,full_Crm,convm = gauss_dmft(tau,muWmm,SigWmm,muHm,SigHm,base_M,base_C,Twrm,Tsav,dt)
    elif which=="opto":
        full_rb,full_Crb,convb = gauss_dmft(tau,muWbb,SigWbb,muHb,SigHb,opto_M,opto_C,Twrm,Tsav,dt)
        full_rm,full_Crm,convm = gauss_dmft(tau,muWmm,SigWmm,muHm,SigHm,opto_M,opto_C,Twrm,Tsav,dt)
    elif which=="both":
        full_rb,full_Crb,convb = doub_gauss_dmft(tau,muWbb,SigWbb,muHb,SigHb,[base_M,opto_M],[base_C,opto_C],
                                                 Twrm,Tsav,dt)
        full_rm,full_Crm,convm = doub_gauss_dmft(tau,muWmm,SigWmm,muHm,SigHm,[base_M,opto_M],[base_C,opto_C],
                                                 Twrm,Tsav,dt)

        def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
            out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
            out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])
            
        rb = full_rb[:,-1]
        rm = full_rm[:,-1]
        Crb = full_Crb[:,-1,-1:-Nsav-1:-1]
        Crm = full_Crm[:,-1,-1:-Nsav-1:-1]
        
        doub_muWbb = doub_mat(muWbb)
        doub_muWmm = doub_mat(muWmm)
        doub_SigWbb = doub_mat(SigWbb)
        doub_SigWmm = doub_mat(SigWmm)
        
        mub = doub_muWbb@rb + doub_vec(muHb)
        mum = doub_muWmm@rm + doub_vec(muHm)
        Sigb = doub_SigWbb@Crb + np.concatenate([SigHb,SigHb],axis=0)
        Sigm = doub_SigWmm@Crm + np.concatenate([SigHm,SigHm],axis=0)

        full_Cdrb,convdb = diff_gauss_dmft(tau,muWbb,SigWbb,muHb,SigHb,diff_R,Twrm,Tsav,dt,rb,Crb)
        full_Cdrm,convdm = diff_gauss_dmft(tau,muWmm,SigWmm,muHm,SigHm,diff_R,Twrm,Tsav,dt,rm,Crm)
    else:
        raise NotImplementedError("Only implemented options for \"which\" keyword are: \"base\", \"opto\", and \"both\"")
        
    print("integrating first stage took",time.process_time() - start,"s")
    
    if which in ("base","opto"):
        # extract predicted moments after long time evolution
        rb = full_rb[:,-1]
        rm = full_rm[:,-1]
        Crb = full_Crb[:,-1,-1:-Nsav-1:-1]
        Crm = full_Crm[:,-1,-1:-Nsav-1:-1]
        
        mub = muWbb@rb + muHb
        Sigb = SigWbb@Crb + SigHb
        mum = muWmm@rm + muHm
        Sigm = SigWmm@Crm + SigHm
    elif which=="both":
        drb = rb[:2] - rb[2:]
        drm = rm[:2] - rm[2:]
        Cdrb = full_Cdrb[:,-1,-1:-Nsav-1:-1]
        Cdrm = full_Cdrm[:,-1,-1:-Nsav-1:-1]

        dmub = mub[:2] - mub[2:]
        dmum = mum[:2] - mum[2:]

        Sigdb = SigWbb@Cdrb
        Sigdm = SigWmm@Cdrm
    else:
        raise NotImplementedError("Only implemented options for \"which\" keyword are: \"base\", \"opto\", and \"both\"")
    
    res_dict = {}
    
    res_dict["rb"] = rb
    res_dict["rm"] = rm
    res_dict["sr"] = sr
    res_dict["Crb"] = Crb
    res_dict["Crm"] = Crm
    res_dict["sCr"] = sCr
    
    res_dict["mub"] = mub
    res_dict["mum"] = mum
    res_dict["Sigb"] = Sigb
    res_dict["Sigm"] = Sigm
    
    res_dict["convb"] = convb
    res_dict["convm"] = convm
    
    if which=="both":
        res_dict["drb"] = drb
        res_dict["drm"] = drm
        res_dict["Cdrb"] = Cdrb
        res_dict["Cdrm"] = Cdrm
        
        res_dict["dmub"] = dmub
        res_dict["dmum"] = dmum
        res_dict["Sigdb"] = Sigdb
        res_dict["Sigdm"] = Sigdm
        
        res_dict["convdb"] = convdb
        res_dict["convdm"] = convdm
    
    if return_full:
        res_dict["full_rb"] = full_rb
        res_dict["full_rm"] = full_rm
        res_dict["full_Crb"] = full_Crb
        res_dict["full_Crm"] = full_Crm
        if which=="both":
            res_dict["full_Cdrb"] = full_Cdrb
            res_dict["full_Cdrm"] = full_Cdrm
    
    return res_dict

def run_decoupled_three_site_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,dori=45,L=180,
                                  struct_dict=None,which="both",couple_matched=False,return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms["K"]
    SoriE = prms["SoriE"]
    SoriI = prms["SoriI"]
    SoriF = prms["SoriF"]
    J = prms["J"]
    beta = prms["beta"]
    gE = prms["gE"]
    gI = prms["gI"]
    hE = prms["hE"]
    hI = prms["hI"]
    basefrac = prms.get("basefrac",0)
    baseinp = prms.get("baseinp",0)
    baseprob = prms.get("baseprob",0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =    (1-basefrac)*(1-baseprob) *np.array([K,K/4],dtype=np.float32)
    Kbs =(1-(1-basefrac)*(1-baseprob))*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+(1-(1-basefrac)*(1-baseinp))*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hm = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    CVH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    xmatched = np.array([0,-dori])
    sW2 = sW**2
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    sr = struct_dict["sr"]
    sCr = struct_dict["sCr"][:,-1]
    
    rOinv = inv_overlap(xmatched,sr[:,None])[:,:,0]
    CrOinv = inv_overlap(xmatched,sCr[:,None])[:,:,0]
    
    sWr = np.sqrt(sW2+sr**2)
    sWCr = np.sqrt(sW2+sCr**2)
    
    muWbb = (1 - 2*struct_fact(180/2,sWr,sr,180)*np.sum(rOinv,-1)[None,:]) * muW +\
        (1 - 2*unstruct_fact(sr,L)*np.sum(rOinv,-1)[None,:]) * muWb
    muWbm = (struct_fact(180/2,sWr,sr,180) * muW + unstruct_fact(sr,L) * muWb)*np.sum(rOinv,-1)[None,:]
    muWmb = (1 - (struct_fact(0,sWr,sr,180) + struct_fact(dori,sWr,sr,180))*np.sum(rOinv,-1)[None,:]) * muW +\
        (1 - 2*unstruct_fact(sr,L)*np.sum(rOinv,-1)[None,:]) * muWb
    muWps = (struct_fact(0,sWr,sr,180)*rOinv[None,:,0] + struct_fact(dori,sWr,sr,180)*rOinv[None,:,1]) * muW +\
        unstruct_fact(sr,L)*np.sum(rOinv,-1)[None,:] * muWb
    muWpc = (struct_fact(dori,sWr,sr,180)*rOinv[None,:,0] + struct_fact(0,sWr,sr,180)*rOinv[None,:,1]) * muW +\
        unstruct_fact(sr,L)*np.sum(rOinv,-1)[None,:] * muWb

    SigWbb = (1 - 2*struct_fact(180/2,sWCr,sCr,180)*np.sum(CrOinv,-1)[None,:]) * SigW +\
        (1 - 2*unstruct_fact(sCr,L)*np.sum(CrOinv,-1)[None,:]) * SigWb
    SigWbm = (struct_fact(180/2,sWCr,sCr,180) * SigW + unstruct_fact(sCr,L) * SigWb)*np.sum(CrOinv,-1)[None,:]
    SigWmb = (1 - (struct_fact(0,sWCr,sCr,180) + struct_fact(dori,sWCr,sCr,180))*np.sum(CrOinv,-1)[None,:]) * SigW +\
        (1 - 2*unstruct_fact(sCr,L)*np.sum(CrOinv,-1)[None,:]) * SigWb
    SigWps = (struct_fact(0,sWCr,sCr,180)*CrOinv[None,:,0] + struct_fact(dori,sWCr,sCr,180)*CrOinv[None,:,1]) * SigW +\
        unstruct_fact(sCr,L)*np.sum(CrOinv,-1) * SigWb
    SigWpc = (struct_fact(dori,sWCr,sCr,180)*CrOinv[None,:,0] + struct_fact(0,sWCr,sCr,180)*CrOinv[None,:,1]) * SigW +\
        unstruct_fact(sCr,L)*np.sum(CrOinv,-1) * SigWb
    
    muWmm = muWps.copy()
    SigWmm = SigWps.copy()
    if couple_matched:
        muWmm += muWpc
        SigWmm += SigWpc
    
    muHb = tau*Hb + 2*muWbm@struct_dict.get("rm",0)
    SigHb = ((tau*Hb*CVH)**2)[:,None] + 2*SigWbm@struct_dict.get("Crm",0)
    muHm = tau*(Hm+(Hm-Hb)*basesubwrapnorm(dori,sH,L)) + muWmb@struct_dict.get("rb",0)
    SigHm = ((tau*(Hm+(Hm-Hb)*basesubwrapnorm(dori,sH,L))*CVH)**2)[:,None] +\
        SigWmb@struct_dict.get("Crb",0)
    if not couple_matched:
        muHm += muWpc@struct_dict.get("rm",0)
        SigHm += SigWpc@struct_dict.get("Crm",0)
    
    Norig = SigHb.shape[1]
    if Norig!=Nsav:
        temp = SigHb.copy()
        SigHb = np.zeros((2,Nsav))
        SigHb[:,:Norig] = temp
        SigHb[:,Norig:] = temp[:,-1:]
        
        temp = SigHm.copy()
        SigHm = np.zeros((2,Nsav))
        SigHm[:,:Norig] = temp
        SigHm[:,Norig:] = temp[:,-1:]
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms["L"],prms["CVL"])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    start = time.process_time()
    
    if which=="base":
        full_rb,full_Crb,convb = gauss_dmft(tau,muWbb,SigWbb,muHb,SigHb,base_M,base_C,Twrm,Tsav,dt)
        full_rm,full_Crm,convm = gauss_dmft(tau,muWmm,SigWmm,muHm,SigHm,base_M,base_C,Twrm,Tsav,dt)
    elif which=="opto":
        full_rb,full_Crb,convb = gauss_dmft(tau,muWbb,SigWbb,muHb,SigHb,opto_M,opto_C,Twrm,Tsav,dt)
        full_rm,full_Crm,convm = gauss_dmft(tau,muWmm,SigWmm,muHm,SigHm,opto_M,opto_C,Twrm,Tsav,dt)
    else:
        raise NotImplementedError("Only implemented options for \"which\" keyword are: \"base\" and \"opto\"")
        
    print("integrating first stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    rb = full_rb[:,-1]
    rm = full_rm[:,-1]
    Crb = full_Crb[:,-1,-1:-Nsav-1:-1]
    Crm = full_Crm[:,-1,-1:-Nsav-1:-1]
    
    if which in ("base","opto"):
        mub = muWbb@rb + muHb
        Sigb = SigWbb@Crb + SigHb
        mum = muWmm@rm + muHm
        Sigm = SigWmm@Crm + SigHm
    else:
        raise NotImplementedError("Only implemented options for \"which\" keyword are: \"base\", \"opto\", and \"both\"")
    
    res_dict = {}
    
    res_dict["rb"] = rb
    res_dict["rm"] = rm
    res_dict["sr"] = sr
    res_dict["Crb"] = Crb
    res_dict["Crm"] = Crm
    res_dict["sCr"] = sCr
    
    res_dict["mub"] = mub
    res_dict["mum"] = mum
    res_dict["Sigb"] = Sigb
    res_dict["Sigm"] = Sigm
    
    res_dict["convb"] = convb
    res_dict["convm"] = convm
    
    if return_full:
        res_dict["full_rb"] = full_rb
        res_dict["full_rm"] = full_rm
        res_dict["full_Crb"] = full_Crb
        res_dict["full_Crm"] = full_Crm
    
    return res_dict

def run_decoupled_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,L=180,Nori=20,
                                struct_dict=None,which="both",return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms["K"]
    SoriE = prms["SoriE"]
    SoriI = prms["SoriI"]
    SoriF = prms["SoriF"]
    J = prms["J"]
    beta = prms["beta"]
    gE = prms["gE"]
    gI = prms["gI"]
    hE = prms["hE"]
    hI = prms["hI"]
    basefrac = prms.get("basefrac",0)
    baseinp = prms.get("baseinp",0)
    baseprob = prms.get("baseprob",0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =    (1-basefrac)*(1-baseprob) *np.array([K,K/4],dtype=np.float32)
    Kbs =(1-(1-basefrac)*(1-baseprob))*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+(1-(1-basefrac)*(1-baseinp))*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hm = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    CVH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sW2 = sW**2
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    sr = struct_dict["sr"]
    sCr = struct_dict["sCr"][:,-1]
    
    sWr = np.sqrt(sW2+sr**2)
    sWCr = np.sqrt(sW2+sCr**2)
    
    muWs = np.zeros((Nori,Nori,2,2))
    SigWs = np.zeros((Nori,Nori,2,2))
    
    oris = np.arange(Nori)/Nori * L
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,:,None,None],sW[None,None,:,:],L)
    
    muWs = (muWb[None,None,:,:] + muW[None,None,:,:]*2*np.pi*kerns) / Nori
    SigWs = (SigWb[None,None,:,:] + SigW[None,None,:,:]*2*np.pi*kerns) / Nori
    
    muWxs = muWs.copy()
    SigWxs = SigWs.copy()
    for i in range(Nori):
        muWxs[i,i] = 0
        SigWxs[i,i] = 0
    
    rs = struct_dict["rb"][None,:] + (struct_dict["rm"]-struct_dict["rb"])[None,:]*\
        basesubwrapnorm(oris[:,None],sr[None,:])
    Crs = struct_dict["Crb"][None,:,:] + (struct_dict["Crm"]-struct_dict["Crb"])[None,:,:]*\
        basesubwrapnorm(oris[:,None,None],sCr[None,:,None])
    
    muHs = tau[None,:]*(Hb[None,:] + (Hm-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:])) +\
        np.einsum("ijkl,jl->ik",muWxs,rs)
    SigHs = ((tau[None,:]*(Hb[None,:] + (Hm-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:]))*CVH)**2)[:,:,None] +\
        np.einsum("ijkl,jlm->ikm",SigWxs,Crs)
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms["L"],prms["CVL"])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    start = time.process_time()
    
    full_rs = [None]*20
    full_Crs = [None]*20
    convs = [None]*20
    
    if which=="base":
        for i in range(Nori):
            full_rs[i],full_Crs[i],convs[i] = gauss_dmft(tau,muWs[i,i],SigWs[i,i],muHs[i],SigHs[i],
                                                         base_M,base_C,Twrm,Tsav,dt)
    elif which=="opto":
        for i in range(Nori):
            full_rs[i],full_Crs[i],convs[i] = gauss_dmft(tau,muWs[i,i],SigWs[i,i],muHs[i],SigHs[i],
                                                         opto_M,opto_C,Twrm,Tsav,dt)
    else:
        raise NotImplementedError("Only implemented options for \"which\" keyword are: \"base\" and \"opto\"")
        
    print("integrating first stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    for i in range(Nori):
        rs[i] = full_rs[i][:,-1]
        Crs[i] = full_Crs[i][:,-1,-1:-Nsav-1:-1]
    
    if which in ("base","opto"):
        mus = np.zeros_like(rs)
        Sigs = np.zeros_like(Crs)
        for i in range(Nori):
            mus[i] = muWs[i,i]@rs[i] + muHs[i]
            Sigs[i] = SigWs[i,i]@Crs[i] + SigHs[i]
    else:
        raise NotImplementedError("Only implemented options for \"which\" keyword are: \"base\", \"opto\", and \"both\"")
    
    res_dict = {}
    
    res_dict["rs"] = rs
    res_dict["sr"] = sr
    res_dict["Crs"] = Crs
    res_dict["sCr"] = sCr
    
    res_dict["mus"] = mus
    res_dict["Sigs"] = Sigs
    
    res_dict["convs"] = convs
    
    if return_full:
        res_dict["full_rs"] = full_rs
        res_dict["full_Crs"] = full_Crs
    
    return res_dict

def run_two_stage_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,sa=15,L=180,return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms["K"]
    SoriE = prms["SoriE"]
    SoriI = prms["SoriI"]
    SoriF = prms["SoriF"]
    J = prms["J"]
    beta = prms["beta"]
    gE = prms["gE"]
    gI = prms["gI"]
    hE = prms["hE"]
    hI = prms["hI"]
    basefrac = prms.get("basefrac",0)
    baseinp = prms.get("baseinp",0)
    baseprob = prms.get("baseprob",0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =    (1-basefrac)*(1-baseprob) *np.array([K,K/4],dtype=np.float32)
    Kbs =(1-(1-basefrac)*(1-baseprob))*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+(1-(1-basefrac)*(1-baseinp))*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hm = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    CVH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sW2 = sW**2
    
    solve_width = get_solve_width(sa,L)
    
    muHb = tau*Hb
    SigHb = (muHb*CVH)**2
    muHa = tau*(Hb+(Hm-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*CVH)**2
    muHm = tau*Hm
    SigHm = (muHm*CVH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms["L"],prms["CVL"])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])
    
    start = time.process_time()
    
    full_rb,full_ra,full_rm,full_Crb,full_Cra,full_Crm,\
        convb,conva,convm = doub_sparse_ring_dmft(tau,W,Ks,Hb,Hm,CVH,sW,sH,sa,[base_M,opto_M],[base_C,opto_C],
                                                  Twrm,Tsav,dt,Kb=Kbs,L=L)

    print("integrating first stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    rb = full_rb[:,-1]
    ra = full_ra[:,-1]
    rm = full_rm[:,-1]
    Crb = full_Crb[:,-1,-1:-Nsav-1:-1]
    Cra = full_Cra[:,-1,-1:-Nsav-1:-1]
    Crm = full_Crm[:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)
    
    sr = solve_width((ra-rb)/(rm-rb))
    sWr = np.sqrt(doub_mat(sW2)+sr**2)
    rmsubb = rm-rb
    sCr = solve_width((Cra-Crb)/(Crm-Crb))
    sWCr = np.sqrt(doub_mat(sW2)[:,:,None]+sCr[None,:,:]**2)
    Crmsubb = Crm-Crb
    
    mub = (doub_muW+doub_muWb)@rb + (unstruct_fact(sr,L)*doub_muWb)@rmsubb + doub_vec(muHb)
    mua = mub + (struct_fact(sa,sWr,sr,L)*doub_muW)@rmsubb + doub_vec(muHa-muHb)
    mum = mub + (struct_fact(0,sWr,sr,L)*doub_muW)@rmsubb + doub_vec(muHm-muHb)
    mub = mub + (struct_fact(L/2,sWr,sr,L)*doub_muW)@rmsubb
    Sigb = (doub_SigW+doub_SigWb)@Crb + each_matmul(unstruct_fact(sCr,L)*doub_SigWb[:,:,None],Crmsubb) +\
        doub_vec(SigHb)[:,None]
    Siga = Sigb + each_matmul(struct_fact(sa,sWCr,sCr,L)*doub_SigW[:,:,None],Crmsubb) + doub_vec(SigHa-SigHb)[:,None]
    Sigm = Sigb + each_matmul(struct_fact(0,sWCr,sCr,L)*doub_SigW[:,:,None],Crmsubb) + doub_vec(SigHm-SigHb)[:,None]
    Sigb = Sigb + each_matmul(struct_fact(L/2,sWCr,sCr,L)*doub_SigW[:,:,None],Crmsubb)

    start = time.process_time()

    full_Cdrb,full_Cdra,full_Cdrm,\
        convdb,convda,convdm = diff_sparse_ring_dmft(tau,W,Ks,Hb,Hm,CVH,sW,sH,sa,diff_R,Twrm,Tsav,dt,
                                                rb,ra,rm,Crb,Cra,Crm,Kb=Kbs,L=L)

    print("integrating second stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    drb = rb[2:] - rb[:2]
    dra = ra[2:] - ra[:2]
    drm = rm[2:] - rm[:2]
    Cdrb = full_Cdrb[:,-1,-1:-Nsav-1:-1]
    Cdra = full_Cdra[:,-1,-1:-Nsav-1:-1]
    Cdrm = full_Cdrm[:,-1,-1:-Nsav-1:-1]

    dmub = mub[2:] - mub[:2]
    dmua = mua[2:] - mua[:2]
    dmum = mum[2:] - mum[:2]

    sCdr = solve_width((Cdra-Cdrb)/(Cdrm-Cdrb))
    sWCdr = np.sqrt(sW2[:,:,None]+sCdr[None,:,:]**2)
    Cdrmsubb = Cdrm - Cdrb
    Sigdb = (SigW+SigWb)@Cdrb + each_matmul(unstruct_fact(sCdr,L)*SigWb[:,:,None],Cdrmsubb)
    Sigda = Sigdb + each_matmul(struct_fact(sa,sWCdr,sCdr,L)*SigW[:,:,None],Cdrmsubb)
    Sigdm = Sigdb + each_matmul(struct_fact(0,sWCdr,sCdr,L)*SigW[:,:,None],Cdrmsubb)
    Sigdb = Sigdb + each_matmul(struct_fact(L/2,sWCdr,sCdr,L)*SigW[:,:,None],Cdrmsubb)
    
    res_dict = {}
    
    res_dict["rb"] = rb
    res_dict["ra"] = ra
    res_dict["rm"] = rm
    res_dict["sr"] = sr
    res_dict["Crb"] = Crb
    res_dict["Cra"] = Cra
    res_dict["Crm"] = Crm
    res_dict["sCr"] = sCr
    res_dict["drb"] = drb
    res_dict["dra"] = dra
    res_dict["drm"] = drm
    res_dict["Cdrb"] = Cdrb
    res_dict["Cdra"] = Cdra
    res_dict["Cdrm"] = Cdrm
    res_dict["sCdr"] = sCdr
    
    res_dict["mub"] = mub
    res_dict["mua"] = mua
    res_dict["mum"] = mum
    res_dict["Sigb"] = Sigb
    res_dict["Siga"] = Siga
    res_dict["Sigm"] = Sigm
    res_dict["dmub"] = dmub
    res_dict["dmua"] = dmua
    res_dict["dmum"] = dmum
    res_dict["Sigdb"] = Sigdb
    res_dict["Sigda"] = Sigda
    res_dict["Sigdm"] = Sigdm

    res_dict["convb"] = convb
    res_dict["conva"] = conva
    res_dict["convm"] = convm
    res_dict["convdb"] = convdb
    res_dict["convda"] = convda
    res_dict["convdm"] = convdm
    
    if return_full:
        res_dict["full_rb"] = full_rb
        res_dict["full_ra"] = full_ra
        res_dict["full_rm"] = full_rm
        res_dict["full_Crb"] = full_Crb
        res_dict["full_Cra"] = full_Cra
        res_dict["full_Crm"] = full_Crm
        res_dict["full_Cdrb"] = full_Cdrb
        res_dict["full_Cdra"] = full_Cdra
        res_dict["full_Cdrm"] = full_Cdrm
    
    return res_dict

def run_two_stage_2feat_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,sa=15,dori=45,L=180,return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms["K"]
    SoriE = prms["SoriE"]
    SoriI = prms["SoriI"]
    SoriF = prms["SoriF"]
    J = prms["J"]
    beta = prms["beta"]
    gE = prms["gE"]
    gI = prms["gI"]
    hE = prms["hE"]
    hI = prms["hI"]
    basefrac = prms.get("basefrac",0)
    baseinp = prms.get("baseinp",0)
    baseprob = prms.get("baseprob",0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =    (1-basefrac)*(1-baseprob) *np.array([K,K/4],dtype=np.float32)
    Kbs =(1-(1-basefrac)*(1-baseprob))*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+(1-(1-basefrac)*(1-baseinp))*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hm = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    CVH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sW2 = sW**2
    
    xmatched = np.array([0,-dori])
    solve_width = get_2feat_solve_width(sa,dori,L)
    
    muHb = tau*Hb
    SigHb = (muHb*CVH)**2
    muHa = tau*(Hb+(Hm-Hb)*(basesubwrapnorm(sa,sH,L)+basesubwrapnorm(dori+sa,sH,L)))
    SigHa = (muHa*CVH)**2
    muHm = tau*(Hm+(Hm-Hb)*basesubwrapnorm(dori,sH,L))
    SigHm = (muHm*CVH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms["L"],prms["CVL"])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])
    
    start = time.process_time()
    
    full_rb,full_ra,full_rm,full_Crb,full_Cra,full_Crm,\
        convb,conva,convm = doub_sparse_2feat_ring_dmft(tau,W,Ks,Hb,Hm,CVH,sW,sH,sa,[base_M,opto_M],[base_C,opto_C],
                                                        Twrm,Tsav,dt,Kb=Kbs,dori=dori,L=L)

    print("integrating first stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    rb = full_rb[:,-1]
    ra = full_ra[:,-1]
    rm = full_rm[:,-1]
    Crb = full_Crb[:,-1,-1:-Nsav-1:-1]
    Cra = full_Cra[:,-1,-1:-Nsav-1:-1]
    Crm = full_Crm[:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)
    
    sr = solve_width((ra-rb)/(rm-rb))
    sWr = np.sqrt(doub_mat(sW2)+sr**2)
    rOinv = np.sum(inv_overlap(xmatched,sr[:,None])[:,:,0],-1)
    rmsubb = (rm-rb)*rOinv
    sCr = solve_width((Cra-Crb)/(Crm-Crb))
    sWCr = np.sqrt(doub_mat(sW2)[:,:,None]+sCr[None,:,:]**2)
    CrOinv = np.sum(inv_overlap(xmatched,sCr.flatten()[:,None])[:,:,0],-1).reshape(-1,Nsav)
    Crmsubb = (Crm-Crb)*CrOinv
    
    mub = (doub_muW+doub_muWb)@rb + (unstruct_fact(sr,L)*doub_muWb)@rmsubb + doub_vec(muHb)
    mua = mub + ((struct_fact(sa,sWr,sr,L)+struct_fact(sa+dori,sWr,sr,L))*doub_muW)@rmsubb + doub_vec(muHa-muHb)
    mum = mub + ((struct_fact(0,sWr,sr,L)+struct_fact(dori,sWr,sr,L))*doub_muW)@rmsubb + doub_vec(muHm-muHb)
    mub = mub + (2*struct_fact(L/2,sWr,sr,L)*doub_muW)@rmsubb
    Sigb = (doub_SigW+doub_SigWb)@Crb + each_matmul(unstruct_fact(sCr,L)*doub_SigWb[:,:,None],Crmsubb) +\
        doub_vec(SigHb)[:,None]
    Siga = Sigb + each_matmul((struct_fact(sa,sWCr,sCr,L)+\
        struct_fact(sa+dori,sWCr,sCr,L))*doub_SigW[:,:,None],Crmsubb) + doub_vec(SigHa-SigHb)[:,None]
    Sigm = Sigb + each_matmul((struct_fact(0,sWCr,sCr,L)+\
        struct_fact(dori,sWCr,sCr,L))*doub_SigW[:,:,None],Crmsubb) + doub_vec(SigHm-SigHb)[:,None]
    Sigb = Sigb + each_matmul(2*struct_fact(L/2,sWCr,sCr,L)*doub_SigW[:,:,None],Crmsubb)

    start = time.process_time()

    full_Cdrb,full_Cdra,full_Cdrm,\
        convdb,convda,convdm = diff_sparse_2feat_ring_dmft(tau,W,Ks,Hb,Hm,CVH,sW,sH,sa,diff_R,Twrm,Tsav,dt,
                                                           rb,ra,rm,Crb,Cra,Crm,Kb=Kbs,dori=dori,L=L)

    print("integrating second stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    drb = rb[2:] - rb[:2]
    dra = ra[2:] - ra[:2]
    drm = rm[2:] - rm[:2]
    Cdrb = full_Cdrb[:,-1,-1:-Nsav-1:-1]
    Cdra = full_Cdra[:,-1,-1:-Nsav-1:-1]
    Cdrm = full_Cdrm[:,-1,-1:-Nsav-1:-1]

    dmub = mub[2:] - mub[:2]
    dmua = mua[2:] - mua[:2]
    dmum = mum[2:] - mum[:2]

    sCdr = solve_width((Cdra-Cdrb)/(Cdrm-Cdrb))
    sWCdr = np.sqrt(sW2[:,:,None]+sCdr[None,:,:]**2)
    CdrOinv = np.sum(inv_overlap(xmatched,sCdr.flatten()[:,None])[:,:,0],-1).reshape(-1,Nsav)
    Cdrmsubb = (Cdrm-Cdrb)*CdrOinv
    Sigdb = (SigW+SigWb)@Cdrb + each_matmul(unstruct_fact(sCdr,L)*SigWb[:,:,None],Cdrmsubb)
    Sigda = Sigdb + each_matmul((struct_fact(sa,sWCdr,sCdr,L)+struct_fact(sa+dori,sWCdr,sCdr,L))*SigW[:,:,None],Cdrmsubb)
    Sigdm = Sigdb + each_matmul((struct_fact(0,sWCdr,sCdr,L)+struct_fact(dori,sWCdr,sCdr,L))*SigW[:,:,None],Cdrmsubb)
    Sigdb = Sigdb + each_matmul(2*struct_fact(L/2,sWCdr,sCdr,L)*SigW[:,:,None],Cdrmsubb)
    
    res_dict = {}
    
    res_dict["rb"] = rb
    res_dict["ra"] = ra
    res_dict["rm"] = rm
    res_dict["sr"] = sr
    res_dict["Crb"] = Crb
    res_dict["Cra"] = Cra
    res_dict["Crm"] = Crm
    res_dict["sCr"] = sCr
    res_dict["drb"] = drb
    res_dict["dra"] = dra
    res_dict["drm"] = drm
    res_dict["Cdrb"] = Cdrb
    res_dict["Cdra"] = Cdra
    res_dict["Cdrm"] = Cdrm
    res_dict["sCdr"] = sCdr
    
    res_dict["mub"] = mub
    res_dict["mua"] = mua
    res_dict["mum"] = mum
    res_dict["Sigb"] = Sigb
    res_dict["Siga"] = Siga
    res_dict["Sigm"] = Sigm
    res_dict["dmub"] = dmub
    res_dict["dmua"] = dmua
    res_dict["dmum"] = dmum
    res_dict["Sigdb"] = Sigdb
    res_dict["Sigda"] = Sigda
    res_dict["Sigdm"] = Sigdm

    res_dict["convb"] = convb
    res_dict["conva"] = conva
    res_dict["convm"] = convm
    res_dict["convdb"] = convdb
    res_dict["convda"] = convda
    res_dict["convdm"] = convdm
    
    if return_full:
        res_dict["full_rb"] = full_rb
        res_dict["full_ra"] = full_ra
        res_dict["full_rm"] = full_rm
        res_dict["full_Crb"] = full_Crb
        res_dict["full_Cra"] = full_Cra
        res_dict["full_Crm"] = full_Crm
        res_dict["full_Cdrb"] = full_Cdrb
        res_dict["full_Cdra"] = full_Cdra
        res_dict["full_Cdrm"] = full_Cdrm
    
    return res_dict

def run_two_stage_full_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,L=180,Nori=20,
                                 which="both",return_full=False):    
    Nsav = round(Tsav/dt)+1
    
    K = prms["K"]
    SoriE = prms["SoriE"]
    SoriI = prms["SoriI"]
    SoriF = prms["SoriF"]
    J = prms["J"]
    beta = prms["beta"]
    gE = prms["gE"]
    gI = prms["gI"]
    hE = prms["hE"]
    hI = prms["hI"]
    basefrac = prms.get("basefrac",0)
    baseinp = prms.get("baseinp",0)
    baseprob = prms.get("baseprob",0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =    (1-basefrac)*(1-baseprob) *np.array([K,K/4],dtype=np.float32)
    Kbs =(1-(1-basefrac)*(1-baseprob))*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+(1-(1-basefrac)*(1-baseinp))*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hm = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    CVH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    oris = np.arange(Nori)/Nori * L
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,None,:,None],sW[None,:,None,:],L)
    
    muHs = tau*(Hb[None,:]+(Hm-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:],L))
    SigHs = (muHs*CVH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms["L"],prms["CVL"])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])
    
    start = time.process_time()
    
    full_rs,full_Crs,convs = doub_sparse_full_ring_dmft(tau,W,Ks,Hb,Hm,CVH,sW,sH,[base_M,opto_M],[base_C,opto_C],
                                                        Twrm,Tsav,dt,Kb=Kbs,L=L,Nori=Nori)
        
    print("integrating first stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    rs = full_rs[:,:,-1]
    Crs = full_Crs[:,:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    SigWs = (SigWb[None,:,None,:] + SigW[None,:,None,:]*2*np.pi*kerns) / Nori
    
    doub_tau = doub_vec(tau)
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)
    doub_sW = np.block([[sW,sW],[sW,sW]])
    doub_Hb = doub_vec(Hb)
    doub_Hm = doub_vec(Hm)
    doub_sH = doub_vec(sH)
    
    doub_kerns = wrapnormdens(doris[:,None,:,None],doub_sW[None,:,None,:],L)

    doub_muWs = (doub_muWb[None,:,None,:] + doub_muW[None,:,None,:]*2*np.pi*doub_kerns) / Nori
    doub_SigWs = (doub_SigWb[None,:,None,:] + doub_SigW[None,:,None,:]*2*np.pi*doub_kerns) / Nori

    doub_muHs = doub_tau*(doub_Hb[None,:]+(doub_Hm-doub_Hb)[None,:]*basesubwrapnorm(oris[:,None],doub_sH[None,:],L))
    doub_SigHs = (doub_muHs*CVH)**2
    
    mus = np.einsum("ijkl,kl->ij",doub_muWs,rs) + doub_muHs
    Sigs = np.einsum("ijkl,klm->ijm",doub_SigWs,Crs) + doub_SigHs[:,:,None]

    start = time.process_time()

    full_Cdrs,convds = diff_sparse_full_ring_dmft(tau,W,Ks,Hb,Hm,CVH,sW,sH,diff_R,Twrm,Tsav,dt,
                                                  rs,Crs,Kb=Kbs,L=L,Nori=Nori)

    print("integrating second stage took",time.process_time() - start,"s")

    # extract predicted moments after long time evolution
    drs = rs[:,2:] - rs[:,:2]
    Cdrs = full_Cdrs[:,:,-1,-1:-Nsav-1:-1]

    dmus = mus[:,2:] - mus[:,:2]

    Sigds = np.einsum("ijkl,klm->ijm",SigWs,Cdrs)
    
    res_dict = {}
    
    res_dict["rs"] = rs
    res_dict["Crs"] = Crs
    res_dict["drs"] = drs
    res_dict["Cdrs"] = Cdrs
    
    res_dict["mus"] = mus
    res_dict["Sigs"] = Sigs
    res_dict["dmus"] = dmus
    res_dict["Sigds"] = Sigds

    res_dict["convs"] = convs
    res_dict["convds"] = convds
    
    if return_full:
        res_dict["full_rs"] = full_rs
        res_dict["full_Crs"] = full_Crs
        res_dict["full_Cdrs"] = full_Cdrs
    
    return res_dict

def lin_resp_mats(tau,muW,SigW,dmuH,dSigH,M_fn,C_fn,Tsav,dt,mu,Sig):
    Ntyp = len(dmuH)
    Nsav = round(Tsav/dt)+1
    
    def Md_fn(mu,Sig,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        M_fn(mu+dmu,Sig,outr)
        M_fn(mu-dmu,Sig,outl)
        out[:] = (outr-outl)/(2*dmu)
    
    def Md2_fn(mu,Sig,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        M_fn(mu,Sig+dmu**2,outr)
        M_fn(mu,Sig-dmu**2,outl)
        out[:] = (outr-outl)/dmu**2
    
    def Rd_fn(mu,Sig,Cov,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        C_fn(mu+dmu,Sig,Cov,outr)
        C_fn(mu-dmu,Sig,Cov,outl)
        out[:] = (outr-outl)/(4*dmu)
    
    def Rd2_fn(mu,Sig,Cov,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        C_fn(mu,np.fmax(Cov,Sig+dmu**2),Cov,outr)
        C_fn(mu,np.fmax(Cov,Sig-dmu**2),Cov,outl)
        out[:] = (outr-outl)/(np.fmax(Cov,Sig+dmu**2)-np.fmax(Cov,Sig-dmu**2))
    
    def Cd_fn(mu,Sig,Cov,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        C_fn(mu,Sig,np.fmin(Sig,Cov+dmu**2),outr)
        C_fn(mu,Sig,np.fmin(Sig,Cov-dmu**2),outl)
        out[:] = (outr-outl)/(np.fmin(Sig,Cov+dmu**2)-np.fmin(Sig,Cov-dmu**2))
    
    Mphi = np.empty((Ntyp,),dtype=np.float32)
    Mdphi = np.empty((Ntyp,),dtype=np.float32)
    Md2phi = np.empty((Ntyp,),dtype=np.float32)
    Rdphi = np.empty((Ntyp,Nsav),dtype=np.float32)
    Rd2phi = np.empty((Ntyp,Nsav),dtype=np.float32)
    Cphi = np.empty((Ntyp,Nsav),dtype=np.float32)
    Cdphi = np.empty((Ntyp,Nsav),dtype=np.float32)
    
    M_fn(mu,Sig[:,0],Mphi)
    Md_fn(mu,Sig[:,0],Mdphi)
    Md2_fn(mu,Sig[:,0],Md2phi)
    for i in range(Nsav):
        Rd_fn(mu,Sig[:,0],Sig[:,i],Rdphi[:,i])
        Rd2_fn(mu,Sig[:,0],Sig[:,i],Rd2phi[:,i])
        C_fn(mu,Sig[:,0],Sig[:,i],Cphi[:,i])
        Cd_fn(mu,Sig[:,0],Sig[:,i],Cdphi[:,i])
    
    Rd2phi = smooth_func(Rd2phi,dt)
    Cdphi = smooth_func(Cdphi,dt)
        
    d2_mat = d2_stencil(Tsav,dt)
    del_vec = np.concatenate(([1],np.zeros(Nsav-1)))
    
    res_dict = {}
    
    res_dict["d2_mat"] = d2_mat
    res_dict["del_vec"] = del_vec
    res_dict["Mphi"] = Mphi
    res_dict["Mdphi"] = Mdphi
    res_dict["Md2phi"] = Md2phi
    res_dict["Rdphi"] = Rdphi
    res_dict["Rd2phi"] = Rd2phi
    res_dict["Cphi"] = Cphi
    res_dict["Cdphi"] = Cdphi
    
    res_dict["A"] = np.eye(Ntyp) - Mdphi[:,None] * muW
    res_dict["B"] = -0.5 * Md2phi[:,None,None] * SigW[:,:,None] * del_vec[None,None,:]
    res_dict["C"] = -2 * Rdphi[:,:,None] * muW[:,None,:]
    res_dict["D"] = np.eye(Ntyp)[:,None,:,None]*np.eye(Nsav)[None,:,None,:] +\
        - (np.diag(tau**2)[:,None,:,None] - dt*np.diag(tau)[:,None,:,None]) * d2_mat[None,:,None,:] +\
        - Cdphi[:,:,None,None] * SigW[:,None,:,None] * np.eye(Nsav)[None,:,None,:] +\
        - Rd2phi[:,:,None,None] * SigW[:,None,:,None] * del_vec[None,None,None,:]
    res_dict["D_mint"] = np.eye(Ntyp)[:,None,:,None]*np.eye(Nsav)[None,:,None,:] +\
        - (np.diag(tau**2)[:,None,:,None] - dt*np.diag(tau)[:,None,:,None]) * d2_mat[None,:,None,:]
    
    if dSigH.ndim==1:
        res_dict["E"] = Mdphi * dmuH + 0.5 * Md2phi * dSigH
        res_dict["F"] = 2 * Rdphi * dmuH[:,None] + (Cdphi + Rd2phi) * dSigH[:,None]
    else:
        res_dict["E"] = Mdphi * dmuH + 0.5 * Md2phi * dSigH[:,0]
        res_dict["F"] = 2 * Rdphi * dmuH[:,None] + Cdphi * dSigH + Rd2phi * dSigH[:,0][:,None]
        
    return res_dict