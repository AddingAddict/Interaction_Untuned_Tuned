import time

import numpy as np
import torch

import ring_network as ring_network
import integrate as integ

def gen_ring_disorder(seed,prm_dict,eX,vis_ori=None,opto_per_pop=None):
    '''
    Generates a ring network with disorder and returns the network, the connectivity matrix, the mean input vector,
    the optogenetic input strength, and the noisy relative input strength vector.
    
    Parameters
    ----------
    seed : int
        Seed value for random number generator.
    prm_dict : dict
        Dictionary containing parameters for the network.
    eX : float
        The coefficient of variation of the afferent inputs.
    vis_ori : float, optional
        Visual stimulus orientation.
    opto_per_pop : array-like, optional
        Array containing the optogenetic input multiplier per cell-type. If None, only the excitatory population receives optogenetic stimulation
        
    Returns
    -------
    RingNetwork
        Ring network object.
    array-like
        2D Array of recurrent weight matrix
    array-like
        1D Array of structured afferent inputs per cell
    array-like
        1D Array of baseline afferent inputs per cell
    array-like
        1D Array of optogenetic input strengths per cell
    array-like
        1D Array of noisy relative input strengths per cell
    '''
    net = ring_network.RingNetwork(seed=0,NC=[prm_dict.get('NE',4),prm_dict.get('NI',1)],
        Nori=prm_dict.get('Nori',180))

    K = prm_dict.get('K',500)
    SoriE = prm_dict.get('SoriE',30)
    SoriI = prm_dict.get('SoriI',30)
    SoriF = prm_dict.get('SoriF',30)
    J = prm_dict.get('J',1e-4)
    beta = prm_dict.get('beta',1)
    gE = prm_dict.get('gE',5)
    gI = prm_dict.get('gI',4)
    hE = prm_dict.get('hE',1)
    hI = prm_dict.get('hI',1)
    L = prm_dict.get('L',1)
    CVL = prm_dict.get('CVL',1)

    WMat = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    HVec = K*J*np.array([hE,hI/beta],dtype=np.float32)

    net.set_seed(seed)
    net.generate_disorder(WMat,np.array([[SoriE,SoriI],[SoriE,SoriI]]),HVec,SoriF*np.ones(2),K,
                          baseinp=1-(1-prm_dict.get('baseinp',0))*(1-prm_dict.get('basefrac',0)),
                          baseprob=1-(1-prm_dict.get('baseprob',0))*(1-prm_dict.get('basefrac',0)),
                          rho=prm_dict.get('rho',0),vis_ori=vis_ori)

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

    shape = 1/eX**2
    scale = 1/shape
    eps = np.random.default_rng(seed).gamma(shape,scale=scale,size=net.N).astype(np.float32)

    return net,net.M,net.H,B,LAS,eps

def gen_ring_disorder_tensor(seed,prm_dict,eX,vis_ori=None,opto_per_pop=None):
    '''
    Generates a ring network with disorder and returns the network, the connectivity matrix, the mean input vector,
    the optogenetic input strength, and the noisy relative input strength vector as PyTorch tensors.
    
    Parameters
    ----------
    seed : int
        Seed value for random number generator.
    prm_dict : dict
        Dictionary containing parameters for the network.
    eX : float
        The coefficient of variation of the afferent inputs.
    vis_ori : float, optional
        Visual stimulus orientation.
    opto_per_pop : array-like, optional
        Array containing the optogenetic input multiplier per cell-type. If None, only the excitatory population receives optogenetic stimulation
        
    Returns
    -------
    RingNetwork
        Ring network object.
    tensor
        2D Tensor of recurrent weight matrix
    tensor
        1D Tensor of structured afferent inputs per cell
    tensor
        1D Tensor of baseline afferent inputs per cell
    tensor
        1D Tensor of optogenetic input strengths per cell
    tensor
        1D Tensor of noisy relative input strengths per cell
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    net = ring_network.RingNetwork(seed=0,NC=[prm_dict.get('NE',4),prm_dict.get('NI',1)],
        Nori=prm_dict.get('Nori',180))

    K = prm_dict.get('K',500)
    SoriE = prm_dict.get('SoriE',30)
    SoriI = prm_dict.get('SoriI',30)
    SoriF = prm_dict.get('SoriF',30)
    J = prm_dict.get('J',1e-4)
    beta = prm_dict.get('beta',1)
    gE = prm_dict.get('gE',5)
    gI = prm_dict.get('gI',4)
    hE = prm_dict.get('hE',1)
    hI = prm_dict.get('hI',1)
    L = prm_dict.get('L',1)
    CVL = prm_dict.get('CVL',1)

    WMat = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    HVec = K*J*np.array([hE,hI/beta],dtype=np.float32)

    net.set_seed(seed)
    net.generate_disorder(WMat,np.array([[SoriE,SoriI],[SoriE,SoriI]]),HVec,SoriF*np.ones(2),K,
                          baseinp=1-(1-prm_dict.get('baseinp',0))*(1-prm_dict.get('basefrac',0)),
                          baseprob=1-(1-prm_dict.get('baseprob',0))*(1-prm_dict.get('basefrac',0)),
                          rho=prm_dict.get('rho',0),vis_ori=vis_ori)
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

    shape = 1/eX**2
    scale = 1/shape
    this_eps = np.random.default_rng(seed).gamma(shape,scale=scale,size=net.N).astype(np.float32)
    eps = torch.from_numpy(this_eps)
    eps = eps.to(device)

    return net,net.M_torch,net.H_torch,B,LAS,eps

def sim_ring_tensor(prm_dict,eX,bX,aX,ri,T,mask_time,seeds,return_dynas=False,max_min=15):
    net = ring_network.RingNetwork(seed=0,NC=[prm_dict.get('NE',4),prm_dict.get('NI',1)],
        Nori=prm_dict.get('Nori',180))

    rates = np.zeros((2,len(seeds),net.N))
    if return_dynas:
        dynas = np.zeros((2,len(seeds),net.N,len(T)))

    for seed_idx,seed in enumerate(seeds):
        print('Doing seed '+str(seed_idx+1) +' of '+str(len(seeds)))

        start = time.process_time()

        net,M,H,B,LAS,eps = gen_ring_disorder_tensor(seed,prm_dict,eX)

        print("Generating disorder took ",time.process_time() - start," s")
        print('')
        
        start = time.process_time()
        
        sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,M,(bX*B+aX*H)*eps,LAS,net.C_conds[0],
            mult_tau=True,max_min=max_min)
        rates[0,seed_idx,:]=torch.mean(sol[:,mask_time],axis=1).cpu().numpy()
        if return_dynas:
            dynas[0,seed_idx,:,:]=sol.cpu().numpy()

        print("Integrating base network took ",time.process_time() - start," s")
        print('')
        
        start = time.process_time()
        
        sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,M,(bX*B+aX*H)*eps,LAS,net.C_conds[0],
            mult_tau=True,max_min=max_min)
        rates[1,seed_idx,:]=torch.mean(sol[:,mask_time],axis=1).cpu().numpy()
        if return_dynas:
            dynas[1,seed_idx,:,:]=sol.cpu().numpy()

        print("Integrating opto network took ",time.process_time() - start," s")
        print('')
        
    if return_dynas:
        return net,rates.reshape((2,-1)),dynas.reshape((2,-1,len(T))),(base_timeout or opto_timeout)
    else:
        return net,rates.reshape((2,-1)),(base_timeout or opto_timeout)

def sim_ring(prm_dict,eX,bX,aX,ri,T,mask_time,seeds,return_dynas=False,max_min=15,stat_stop=True):
    net = ring_network.RingNetwork(seed=0,NC=[prm_dict.get('NE',4),prm_dict.get('NI',1)],
        Nori=prm_dict.get('Nori',180))
    
    rates = np.zeros((len(seeds),2,net.N))
    if return_dynas:
        dynas = np.zeros((2,len(seeds),net.N,len(T)))

    for seed_idx,seed in enumerate(seeds):
        print('Doing seed '+str(seed_idx+1) +' of '+str(len(seeds)))

        start = time.process_time()

        net,M,H,B,LAS,eps = gen_ring_disorder(seed,prm_dict,eX)

        print("Generating disorder took ",time.process_time() - start," s")
        print('')
        
        start = time.process_time()
        
        sol,base_timeout = integ.sim_dyn(ri,T,0.0,M,(bX*B+aX*H)*eps,LAS,net.C_all[0],net.C_all[1],
            mult_tau=True,max_min=max_min,stat_stop=stat_stop)
        rates[0,seed_idx,:]=torch.mean(sol[:,mask_time],axis=1).cpu().numpy()
        if return_dynas:
            dynas[0,seed_idx,:,:]=sol.cpu().numpy()

        print("Integrating base network took ",time.process_time() - start," s")
        print('')
        
        start = time.process_time()
        
        sol,opto_timeout = integ.sim_dyn(ri,T,1.0,M,(bX*B+aX*H)*eps,LAS,net.C_all[0],net.C_all[1],
            mult_tau=True,max_min=max_min,stat_stop=stat_stop)
        rates[1,seed_idx,:]=torch.mean(sol[:,mask_time],axis=1).cpu().numpy()
        if return_dynas:
            dynas[1,seed_idx,:,:]=sol.cpu().numpy()

        print("Integrating opto network took ",time.process_time() - start," s")
        print('')
        
    if return_dynas:
        return net,rates.reshape((2,-1)),dynas.reshape((2,-1,len(T))),(base_timeout or opto_timeout)
    else:
        return net,rates.reshape((2,-1)),(base_timeout or opto_timeout)
