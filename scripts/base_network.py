from abc import ABC

import numpy as np
from mpmath import fp

jtheta = np.vectorize(fp.jtheta, "D")

def wrapnormdens(x,S):
    return np.real(jtheta(3,x/2,np.exp(-S**2/2)))/(2*np.pi)
        
def make_periodic(x,halfL):
    '''
    Make array x, which is assumed to be in [-L,L], take values in [-L/2,L/2] using periodic boundary conditions.

    Parameters
    ----------
    x : array-like
        Input values for to make periodic, assumed to be in [-L,L].
    L : float
        Size of periodic dimension.

    Returns
    -------
    array-like
        Input values redefined to be in [-L/2,L/2] using periodic boundary conditions.
    '''
    out = np.copy(x);
    out[out >  halfL] =  2*halfL-out[out >  halfL];
    out[out < -halfL] = -2*halfL-out[out < -halfL];
    return out

def apply_kernel(x,S,L,dx=None,kernel="gaussian"):
    '''
    Apply a kernel parameterized by width S to array x, which is assumed to be in [-L/2,L/2].

    Parameters
    ----------
    x : array-like
        Input values for kernel, assumed to be in [-L/2,L/2] .
    S : float
        Width parameter for kernel.
    L : float
        Size of periodic dimension.
    dx : float, optional
        Distance between locations on discretized dimension, which will rescale output to act as an integration measure.
    kernel : str, optional
        Kernel type to apply.

    Returns
    -------
    array-like
        Output values where kernel was applied to input values.
    '''
    if kernel == "gaussian":
        out = np.exp(-x**2/(2*S**2))/np.sqrt(2*np.pi)/S
    elif kernel == "nonnormgaussian":
        out = np.exp(-x**2/(2*S**2))
    elif kernel == "exponential":
        out = np.exp(-np.abs(x)/S)/S
    elif kernel == "nonnormexponential":
        out = np.exp(-np.abs(x)/S)
    elif kernel == "vonmisses":
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = np.exp(np.cos(x_rad)/S_rad)/(2*np.pi*i0(1/S_rad))*d_rad
    elif kernel == "wrapgauss":
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = wrapnormdens(x_rad,S_rad)*d_rad
    elif kernel == "nonnormwrapgauss":
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = wrapnormdens(x_rad,S_rad)/wrapnormdens(0,S_rad)
    elif kernel == "basesubwrapgauss":
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = (wrapnormdens(x_rad,S_rad)-wrapnormdens(np.pi,S_rad))/\
            (wrapnormdens(0,S_rad)-wrapnormdens(np.pi,S_rad))
    else:
        raise Exception("kernel not implemented")
    if dx is None:
        return out
    else:
        return out*dx
    
def bin_corr_bnds(p1,p2):
    '''
    Compute the lower and upper bounds for the correlations between joint Bernoulli variables.

    Parameters
    ----------
    p1 : array-like
        Array of probabilities of first binary variable.
    p2 : array-like
        Array of probabilities of second binary variable.

    Returns
    -------
    lo_bnd: array-like
        Array of lower bounds for each pair.
    up_bnd: array-like 
        Array of upper bounds for each pair.
    '''
    lo_bnd = np.fmax(-np.sqrt(((1-p1)/p1)*((1-p2)/p2)), -1/np.sqrt(((1-p2)/p2)*((1-p1)/p1)))
    up_bnd = np.fmin( np.sqrt(((1-p1)/p1)/((1-p2)/p2)),    np.sqrt(((1-p2)/p2)/((1-p1)/p1)))
    return lo_bnd,up_bnd

class BaseNetwork(ABC):

    """
    Creates a network with n cell types on Nloc sites.
    """
    
    def __init__(self, seed=0, n=None, NC=None, Nloc=1, profile="gaussian"):
        '''
        Construct a BaseNetwork object.

        Parameters
        ----------
        seed : int
            Seed value for random number generator.
        n : int, optional
            Number of cell types. If not provided, defaults to the number of cell types in NC. If neither are provided, defaults to 1.
        NC : int or array-like, optional
            Number of cells of each cell type per site. If a single int is provided, it is used for all cell types. If not provided, the default is one cell per cell type per site.
        Nloc : int
            Number of sites in the network. Defaults to 1.
        profile : str
            Type of kernel to use for generating connections. Defaults to "gaussian".

        Attributes
        ----------
        n : int
            Number of cell types.
        NC : array-like
            Number of cells of each cell type per site.
        Nloc : int
            Number of sites in the network.
        profile : str
            Type of kernel used for generating connections.
        NT : int
            Total number of cells per site.
        N : int
            Total number of neurons in the network (NT * Nloc).
        C_idxs : list of list
            List of indices for each cell type per site.
        C_all : list of array-like
            List of all indices for each cell type across all sites.
        '''
        self.rng = np.random.default_rng(seed=seed)

        if NC is None:
            if n is None:
                self.NC = np.ones(1,dtype=int)
                self.n = 1
            else:
                self.NC = np.ones(n,dtype=int)
                self.n = int(n)
        elif np.isscalar(NC):
            if n is None:
                self.NC = NC*np.ones(1,dtype=int)
                self.n = 1
            else:
                self.NC = NC*np.ones(n,dtype=int)
                self.n = int(n)
        else:
            if n is None:
                self.NC = np.array(NC,dtype=int)
                self.n = len(self.NC)
            else:
                self.NC = np.array(NC,dtype=int)
                self.n = int(n)
                assert self.n == len(self.NC)

        self.NT = np.sum(self.NC)
        self.Nloc = Nloc
        
        # Total number of neurons
        self.N = self.Nloc*self.NT

        # Compute indices for each cell type at each site
        self.C_idxs = []
        self.C_all = []
        prev_NC = 0
        for cidx in range(self.n):
            prev_NC += 0 if cidx == 0 else self.NC[cidx-1]
            this_NC = self.NC[cidx]
            this_C_idxs = [slice(int(loc*self.NT+prev_NC),int(loc*self.NT+prev_NC+this_NC)) for loc in range(self.Nloc)]
            self.C_idxs.append(this_C_idxs)

            this_C_all = np.zeros(0,np.int8)
            for loc in range(self.Nloc):
                this_C_loc_idxs = np.arange(this_C_idxs[loc].start,this_C_idxs[loc].stop,dtype=int)
                this_C_all = np.append(this_C_all,this_C_loc_idxs)
            self.C_all.append(this_C_all)

        self.profile = profile

    def set_seed(self,seed):
        '''
        Reset the random number generator seed.
        
        Parameters
        ----------
        seed : int
            Seed value for random number generator.
        '''
        self.rng = np.random.default_rng(seed=seed)

    def generate_sparse_rec_conn(self,WKern,K):
        '''
        Generate a sparse recurrent connection matrix for the network.
        
        Parameters
        ----------
        WKern : list of list of array-like
            Nested list of kernel applied to distances between sites for each connection type.
        K : float or array-like
            Mean of out-degree per connection type. If a single float is provided, K is set as the mean out-degree of the first cell type and the out-degree of cell type i>0 is scaled by NC[i]/NC[0]. If an nxn 2D array is provided, the elements are used as the out-degree for each connection type.
            
        Returns
        -------
        array-like
            2D Array of the binary connections between all cell pairs.
        '''
        C_full = np.zeros((self.N,self.N),np.uint32)

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            NpstC = self.NC[pstC]
            for preC in range(self.n):
                preC_idxs = self.C_idxs[preC]
                NpreC = self.NC[preC]

                W = WKern[pstC][preC]
                if W is None: continue

                if np.isscalar(K):
                    ps = np.fmax(K/self.NC[0] * W,1e-12)
                else:
                    ps = np.fmax(K[pstC,preC]/self.NC[preC] * W,1e-12)
                if np.any(ps > 1):
                    raise Exception("Error: p > 1, please decrease K or increase NC")

                for pst_loc in range(self.Nloc):
                    pst_idxs = pstC_idxs[pst_loc]
                    for pre_loc in range(self.Nloc):
                        p = ps[pst_loc,pre_loc]
                        pre_idxs = preC_idxs[pre_loc]
                        C_full[pst_idxs,pre_idxs] = self.rng.binomial(1,p,size=(NpstC,NpreC))

        return C_full
    
    def generate_corr_sparse_rec_conn(self,WKern,K,rho):
        '''
        Generate a sparse recurrent connection matrix for the network with correlated connections.
        
        Parameters
        ----------
        WKern : list of list of array-like
            Nested list of kernel applied to distances between sites for each connection type.
        K : float or array-like
            Mean of out-degree per connection type. If a single float is provided, K is set as the mean out-degree of the first cell type and the out-degree of cell type i>0 is scaled by NC[i]/NC[0]. If an nxn 2D array is provided, the elements are used as the out-degree for each connection type.
        rho : float or array-like
            Correlation of the binary connections. If a single float is provided, rho is set as the correlation for all connection types. If an nxn 2D array is provided, the elements are used as the correlation for each connection type.
            
        Returns
        -------
        array-like
            2D Array of the binary connections between all cell pairs.
        '''
        p_full = np.zeros((self.n,self.Nloc,self.n,self.Nloc),np.float32)
        C_full = np.zeros((self.N,self.N),np.uint32)

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            pstC_all = self.C_all[pstC]
            NpstC = self.NC[pstC]
            for preC in range(self.n):
                preC_idxs = self.C_idxs[preC]
                preC_all = self.C_all[preC]
                NpreC = self.NC[preC]

                W = WKern[pstC][preC]
                if W is None: continue

                if np.isscalar(K):
                    ps = np.fmax(K/self.NC[0] * W,1e-12)
                else:
                    ps = np.fmax(K[pstC,preC]/self.NC[preC] * W,1e-12)
                if np.any(ps > 1):
                    raise Exception("Error: p > 1, please decrease K or increase NC")

                for pst_loc in range(self.Nloc):
                    for pre_loc in range(self.Nloc):
                        p_full[pstC,pst_loc,preC,pre_loc] = ps[pst_loc,pre_loc]
        
        lo_bnd,up_bnd = bin_corr_bnds(p_full,p_full.transpose((2,3,0,1)))
        if np.isscalar(rho):
            rho_full = np.clip(rho,0.99*lo_bnd,0.99*up_bnd)
        else:
            rho_full = np.clip(rho[:,None,:,None],0.99*lo_bnd,0.99*up_bnd)
        a_full = np.log(1 + rho_full*np.sqrt((1-p_full)/p_full * ((1-p_full)/p_full).transpose((2,3,0,1))))
        
        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            NpstC = self.NC[pstC]
            for preC in range(self.n):
                preC_idxs = self.C_idxs[preC]
                NpreC = self.NC[preC]
                for pst_loc in range(self.Nloc):
                    pst_idxs = pstC_idxs[pst_loc]
                    for pre_loc in range(self.Nloc):
                        p = p_full[pstC,pst_loc,preC,pre_loc]
                        a = a_full[pstC,pst_loc,preC,pre_loc]
                        pre_idxs = preC_idxs[pre_loc]
                        C_full[pst_idxs,pre_idxs] = self.rng.poisson(-np.log(p)-a,size=(NpstC,NpreC))
                            
        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            NpstC = self.NC[pstC]
            for preC in range(pstC,self.n):
                preC_idxs = self.C_idxs[preC]
                NpreC = self.NC[preC]
                for pst_loc in range(self.Nloc):
                    pst_idxs = pstC_idxs[pst_loc]
                    if pstC==preC:
                        init_loc = pst_loc
                    else:
                        init_loc = 0
                    for pre_loc in range(init_loc,self.Nloc):
                        pre_idxs = preC_idxs[pre_loc]
                        a = a_full[pstC,pst_loc,preC,pre_loc]
                        shared_var = self.rng.poisson(a,size=(NpstC,NpreC)).astype(np.uint32)
                        if pstC==preC and pst_loc==pre_loc:
                            C_full[pst_idxs,pre_idxs] += np.tril(shared_var)
                            C_full[pre_idxs,pst_idxs] += np.tril(shared_var,-1).T
                        else:
                            C_full[pst_idxs,pre_idxs] += shared_var
                            C_full[pre_idxs,pst_idxs] += shared_var.T

        return C_full==0