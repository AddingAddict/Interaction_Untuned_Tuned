import time

import numpy as np
import torch

import ring_network as ring_network
import integrate as integ

def gen_ring_disorder(seed,prms,CVH,vis_ori=None,opto_per_pop=None):
    '''
    Generates a ring network with disorder and returns the network, the connectivity matrix, the mean input vector,
    the optogenetic input strength, and the noisy relative input strength vector.
    
    Parameters
    ----------
    seed : int
        Seed value for random number generator.
    prms : dict
        Dictionary containing parameters for the network.
    CVH : float
        The coefficient of variation of the afferent inputs.
    vis_ori : float, optional
        Visual stimulus orientation.
    opto_per_pop : array-like, optional
        Array containing the optogenetic input multiplier per cell-type. If None, only the excitatory population receives optogenetic stimulation.
        
    Returns
    -------
    RingNetwork
        Ring network object.
    array-like
        2D Array of recurrent weight matrix.
    array-like
        1D Array of structured afferent inputs per cell.
    array-like
        1D Array of baseline afferent inputs per cell.
    array-like
        1D Array of optogenetic input strengths per cell.
    array-like
        1D Array of noisy relative input strengths per cell.
    '''
    net = ring_network.RingNetwork(seed=0,NC=[prms.get('NE',4),prms.get('NI',1)],
        Nori=prms.get('Nori',180))

    K = prms.get('K',500)
    SoriE = prms.get('SoriE',30)
    SoriI = prms.get('SoriI',30)
    SoriF = prms.get('SoriF',30)
    J = prms.get('J',1e-4)
    beta = prms.get('beta',1)
    gE = prms.get('gE',5)
    gI = prms.get('gI',4)
    hE = prms.get('hE',1)
    hI = prms.get('hI',1)
    L = prms.get('L',1)
    CVL = prms.get('CVL',1)

    WMat = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    HVec = K*J*np.array([hE,hI/beta],dtype=np.float32)

    net.set_seed(seed)
    net.generate_disorder(WMat,np.array([[SoriE,SoriI],[SoriE,SoriI]]),HVec,SoriF*np.ones(2),K,
                          baseinp=1-(1-prms.get('baseinp',0))*(1-prms.get('basefrac',0)),
                          baseprob=1-(1-prms.get('baseprob',0))*(1-prms.get('basefrac',0)),
                          rho=prms.get('rho',0),vis_ori=vis_ori)

    B = np.zeros(net.N,dtype=np.float32)
    B[net.C_all[0]] = HVec[0]
    B[net.C_all[1]] = HVec[1]

    if opto_per_pop is None:
        LAS = np.zeros(net.N,dtype=np.float32)
        sigma_l = np.sqrt(np.log(1+CVL**2))
        mu_l = np.log(1e-3*L)-sigma_l**2/2
        LAS_E = np.random.default_rng(seed).lognormal(mu_l, sigma_l, net.NC[0]*net.Nloc).astype(np.float32)
        LAS[net.C_all[0]] = LAS_E
    else:
        LAS = np.zeros(net.N,dtype=np.float32)
        for nc in range(net.n):
            sigma_l = np.sqrt(np.log(1+CVL**2))
            mu_l = np.log(1e-3*L)-sigma_l**2/2
            LAS_P = np.random.default_rng(seed).lognormal(mu_l, sigma_l, net.NC[nc]*net.Nloc).astype(np.float32)
            LAS[net.C_all[nc]] = opto_per_pop[nc]*LAS_P

    shape = 1/CVH**2
    scale = 1/shape
    eps = np.random.default_rng(seed).gamma(shape,scale=scale,size=net.N).astype(np.float32)

    return net,net.M,net.H,B,LAS,eps

def gen_ring_disorder_tensor(seed,prms,CVH,vis_ori=None,opto_per_pop=None):
    '''
    Generates a ring network with disorder and returns the network, the connectivity matrix, the mean input vector,
    the optogenetic input strength, and the noisy relative input strength vector as PyTorch tensors.
    
    Parameters
    ----------
    seed : int
        Seed value for random number generator.
    prms : dict
        Dictionary containing parameters for the network.
    CVH : float
        The coefficient of variation of the afferent inputs.
    vis_ori : float, optional
        Visual stimulus orientation.
    opto_per_pop : array-like, optional
        Array containing the optogenetic input multiplier per cell-type. If None, only the excitatory population receives optogenetic stimulation.
        
    Returns
    -------
    RingNetwork
        Ring network object.
    tensor
        2D Tensor of recurrent weight matrix.
    tensor
        1D Tensor of structured afferent inputs per cell.
    tensor
        1D Tensor of baseline afferent inputs per cell.
    tensor
        1D Tensor of optogenetic input strengths per cell.
    tensor
        1D Tensor of noisy relative input strengths per cell.
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    net = ring_network.RingNetwork(seed=0,NC=[prms.get('NE',4),prms.get('NI',1)],
        Nori=prms.get('Nori',180))

    K = prms.get('K',500)
    SoriE = prms.get('SoriE',30)
    SoriI = prms.get('SoriI',30)
    SoriF = prms.get('SoriF',30)
    J = prms.get('J',1e-4)
    beta = prms.get('beta',1)
    gE = prms.get('gE',5)
    gI = prms.get('gI',4)
    hE = prms.get('hE',1)
    hI = prms.get('hI',1)
    L = prms.get('L',1)
    CVL = prms.get('CVL',1)

    WMat = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    HVec = K*J*np.array([hE,hI/beta],dtype=np.float32)

    net.set_seed(seed)
    net.generate_disorder(WMat,np.array([[SoriE,SoriI],[SoriE,SoriI]]),HVec,SoriF*np.ones(2),K,
                          baseinp=1-(1-prms.get('baseinp',0))*(1-prms.get('basefrac',0)),
                          baseprob=1-(1-prms.get('baseprob',0))*(1-prms.get('basefrac',0)),
                          rho=prms.get('rho',0),vis_ori=vis_ori)
    net.generate_tensors()

    B = torch.where(net.C_conds[0],HVec[0],HVec[1])

    if opto_per_pop is None:
        LAS = torch.zeros(net.N,dtype=torch.float32)
        sigma_l = np.sqrt(np.log(1+CVL**2))
        mu_l = np.log(1e-3*L)-sigma_l**2/2
        LAS_E = np.random.default_rng(seed).lognormal(mu_l, sigma_l, net.NC[0]*net.Nloc).astype(np.float32)
        LAS[net.C_conds[0]] = torch.from_numpy(LAS_E)
    else:
        LAS = torch.zeros(net.N,dtype=torch.float32)
        for nc in range(net.n):
            sigma_l = np.sqrt(np.log(1+CVL**2))
            mu_l = np.log(1e-3*L)-sigma_l**2/2
            LAS_P = np.random.default_rng(seed).lognormal(mu_l, sigma_l, net.NC[nc]*net.Nloc).astype(np.float32)
            LAS[net.C_conds[nc]] = opto_per_pop[nc]*torch.from_numpy(LAS_P)

    B = B.to(device)
    LAS = LAS.to(device)

    shape = 1/CVH**2
    scale = 1/shape
    this_eps = np.random.default_rng(seed).gamma(shape,scale=scale,size=net.N).astype(np.float32)
    eps = torch.from_numpy(this_eps)
    eps = eps.to(device)

    return net,net.M_torch,net.H_torch,B,LAS,eps