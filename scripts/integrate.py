'''
Based on code collaboratively written by Alessandro Sanzeni, Agostina Palmigiano, and Tuan Nguyen for Sanzeni et al 2023
'''
import time
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torchdiffeq import odeint

def sim_dyn(ri,T,L,M,H,LAM,E_all,I_all,mult_tau=False,max_min=7.5,stat_stop=True):
    '''
    Simulate rate dynamics
    
    Parameters
    ----------
    ri : Ricciardi
        Ricciardi class object for computing the activation function.
    T : array-like
        1D Array of time-points to save the rates.
    L : array-like
        1D Array of optogenetic input strengths per cell.
    M : array-like
        2D Array of recurrent weight matrix.
    H : array-like
        1D Array of afferent inputs per cell.
    LAM : float
        Factor by which to multiply the optogentic input strengths.
    E_all : array-like
        Indices of excitatory cells.
    I_all : array-like
        Indices of inhibitory cells.
    mult_tau : bool
        Whether to multiply the input by the time constants of the cells.
    max_min : float
        Maximum time to run the simulation in minutes.
    stat_stop : bool
        Whether to stop the simulation when it reaches a stationary state.
    
    Returns
    -------
    array-like
        2D Array of shape (Ncell x Ntime) with the rates of the cells.
    bool
        Whether the simulation reached the maximum time limit.
    '''
    LAS = LAM*L

    if callable(H):
        N = len(H(0))
    else:
        N = len(H)
    F=np.zeros(N)
    start = time.process_time()
    max_time = max_min*60
    timeout = False

    # This function computes the dynamics of the rate model
    if callable(H):
        if mult_tau:
            def ode_fn(t,R):
                MU=np.matmul(M,R)+H(t)
                MU[E_all]=ri.tE*MU[E_all]
                MU[I_all]=ri.tI*MU[I_all]
                MU=MU+LAS
                F[E_all] =(-R[E_all]+ri.phiE(MU[E_all]))/ri.tE;
                F[I_all] =(-R[I_all]+ri.phiI(MU[I_all]))/ri.tI;
                return F
        else:
            def ode_fn(t,R):
                MU=np.matmul(M,R)+H(t)+LAS
                F[E_all] =(-R[E_all]+ri.phiE(MU[E_all]))/ri.tE;
                F[I_all] =(-R[I_all]+ri.phiI(MU[I_all]))/ri.tI;
                return F
    else:
        if mult_tau:
            def ode_fn(t,R):
                MU=np.matmul(M,R)+H
                MU[E_all]=ri.tE*MU[E_all]
                MU[I_all]=ri.tI*MU[I_all]
                MU=MU+LAS
                F[E_all] =(-R[E_all]+ri.phiE(MU[E_all]))/ri.tE;
                F[I_all] =(-R[I_all]+ri.phiI(MU[I_all]))/ri.tI;
                return F
        else:
            def ode_fn(t,R):
                MU=np.matmul(M,R)+H+LAS
                F[E_all] =(-R[E_all]+ri.phiE(MU[E_all]))/ri.tE;
                F[I_all] =(-R[I_all]+ri.phiI(MU[I_all]))/ri.tI;
                return F

    # This function determines if the system is stationary or not
    def stat_event(t,R):
        meanF = np.mean(np.abs(F)/np.maximum(R,1e-1)) - 5e-3
        if meanF < 0: meanF = 0
        return meanF
    stat_event.terminal = True

    # This function forces the integration to stop after 15 minutes
    def time_event(t,R):
        int_time = (start + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True

    rates=np.zeros((N,len(T)));
    if stat_stop:
        sol = solve_ivp(ode_fn,[np.min(T),np.max(T)],rates[:,0], method="RK45", t_eval=T, events=[stat_event,time_event])
    else:
        sol = solve_ivp(ode_fn,[np.min(T),np.max(T)],rates[:,0], method="RK45", t_eval=T, events=[time_event])
    if sol.t.size < len(T):
        print("      Integration stopped after " + str(np.around(T[sol.t.size-1],2)) + "s of simulation time")
        if time.process_time() - start > max_time:
            print("            Integration reached time limit")
            timeout = True
        rates[:,0:sol.t.size] = sol.y
        rates[:,sol.t.size:] = sol.y[:,-1:]
    else:
        rates=sol.y
    
    return rates,timeout

def sim_dyn_tensor(ri,T,L,M,H,LAM,E_cond,mult_tau=False,max_min=30,method=None):
    '''
    Simulate rate dynamics using PyTorch, allowing for GPU acceleration.
    
    Parameters
    ----------
    ri : Ricciardi
        Ricciardi class object for computing the activation function.
    T : tensor
        1D Tensor of time-points to save the rates.
    L : tensor
        1D Tensor of optogenetic input strengths per cell.
    M : tensor
        2D Tensor of recurrent weight matrix.
    H : tensor
        1D Tensor of afferent inputs per cell.
    LAM : float
        Factor by which to multiply the optogentic input strengths.
    E_cond : tensor
        1D Boolean tensor indicating excitatory cells.
    mult_tau : bool
        Whether to multiply the input by the time constants of the cells.
    max_min : float
        Maximum time to run the simulation in minutes.
    method : str, optional
        Integration method to use (default is "dopri5").
    
    Returns
    -------
    tensor
        2D Tensor of shape (Ncell x Ntime) with the rates of the cells.
    bool
        Whether the simulation reached the maximum time limit.
    '''
    if callable(H):
        N = len(H(0))
    else:
        N = len(H)
    MU = torch.zeros(N,dtype=torch.float32)
    F = torch.ones(N,dtype=torch.float32)
    LAS = LAM*L

    start = time.process_time()
    max_time = max_min*60

    # This function computes the dynamics of the rate model
    if callable(H):
        if mult_tau:
            def ode_fn(t,R):
                MU=torch.matmul(M,R)
                MU=torch.add(MU,H(t))
                MU=torch.where(E_cond,ri.tE*MU,ri.tI*MU)
                MU=torch.add(MU,LAS)
                F=torch.where(E_cond,(-R+ri.phiE_tensor(MU))/ri.tE,(-R+ri.phiI_tensor(MU))/ri.tI)
                if time.process_time() - start > max_time: raise Exception("Timeout")
                return F
        else:
            def ode_fn(t,R):
                MU=torch.matmul(M,R)
                MU=torch.add(MU,H(t) + LAS)
                F=torch.where(E_cond,(-R+ri.phiE_tensor(MU))/ri.tE,(-R+ri.phiI_tensor(MU))/ri.tI)
                if time.process_time() - start > max_time: raise Exception("Timeout")
                return F
    else:
        if mult_tau:
            def ode_fn(t,R):
                MU=torch.matmul(M,R)
                MU=torch.add(MU,H)
                MU=torch.where(E_cond,ri.tE*MU,ri.tI*MU)
                MU=torch.add(MU,LAS)
                F=torch.where(E_cond,(-R+ri.phiE_tensor(MU))/ri.tE,(-R+ri.phiI_tensor(MU))/ri.tI)
                if time.process_time() - start > max_time: raise Exception("Timeout")
                return F
        else:
            def ode_fn(t,R):
                MU=torch.matmul(M,R)
                MU=torch.add(MU,H + LAS)
                F=torch.where(E_cond,(-R+ri.phiE_tensor(MU))/ri.tE,(-R+ri.phiI_tensor(MU))/ri.tI)
                if time.process_time() - start > max_time: raise Exception("Timeout")
                return F

    try:
        rates = odeint(ode_fn,torch.zeros_like(H,dtype=torch.float32),T,method=method)
        timeout = False
    except:
        rates = torch.randn((len(T),N),dtype=torch.float32)*1e4+1e4
        timeout = True
    return torch.transpose(rates,0,1),timeout

def calc_lyapunov_exp(ri,T,L,M,H,LAM,E_all,I_all,rates,NLE,TWONS,TONS,mult_tau=False,save_time=False,return_Q=False):
    '''
    Calculate top Lyapunov exponents
    
    Parameters
    ----------
    ri : Ricciardi
        Ricciardi class object for computing the activation function.
    T : array-like
        1D Array of time-points to save the rates.
    L : array-like
        1D Array of optogenetic input strengths per cell.
    M : array-like
        2D Array of recurrent weight matrix.
    H : array-like
        1D Array of afferent inputs per cell.
    LAM : float
        Factor by which to multiply the optogentic input strengths.
    E_all : array-like
        Indices of excitatory cells.
    I_all : array-like
        Indices of inhibitory cells.
    rates : array-like
        2D Array of shape (Ncell x Ntime) with the rates of the cells.
    NLE : int
        Number of Lyapunov exponents to calculate.
    TWONS : float
        Time for warmup prior to computing Lyapunov exponents.
    TONS : float
        Time between measuring Lyapunov exponents.
    mult_tau : bool
        Whether to multiply the input by the time constants of the cells.
    save_time : bool
        Whether to save the Lyapunov exponents at each time step.
    return_Q : bool
        Whether to return the Q matrix used for the calculation.
    
    Returns
    -------
    array-like
        Top NLE Lyapunov exponents. If save_time is True, then Ls is a 2D array of shape (NLE x Ntime) with the Lyapunov exponents at each time step.
    array-like
        Q matrix used for the calculation. Only returned if return_Q is True.
    '''
    if len(H) != rates.shape[0]:
        raise("Rates must have shape Ncell x Ntime")

    LAS = LAM*L

    dt = T[1] - T[0]
    dt_tau_inv = dt*np.ones_like(H)
    dt_tau_inv[E_all]=dt_tau_inv[E_all]/ri.tE
    dt_tau_inv[I_all]=dt_tau_inv[I_all]/ri.tI
    NT = len(T) - 1
    NWONS = int(np.round(TWONS/dt))
    NONS = int(np.round(TONS/dt))

    print("NWONS =",NWONS)
    print("NT =",NT)
    print("NONS =",NONS)

    if mult_tau:
        def calc_mu(RATE,MU):
            MU=np.matmul(M,RATE)+H
            MU[E_all]=ri.tE*MU[E_all]
            MU[I_all]=ri.tI*MU[I_all]
            MU=MU+LAS
    else:
        def calc_mu(RATE,MU):
            MU=np.matmul(M,RATE)+H+LAS

    start = time.process_time()

    # Initialize Q
    Q,_ = np.linalg.qr(np.random.rand(len(H),len(H)))
    Q = Q[:,:NLE]
    R = np.zeros((NLE,NLE))
    G = np.zeros(len(H))
    MU = np.zeros(len(H))

    print("Initializing Q took",time.process_time() - start,"s\n")

    if save_time:
        Ls = np.zeros((NLE,(NT-NWONS)//NONS))
    else:
        Ls = np.zeros((NLE))

    start = time.process_time()

    # Evolve Q
    for i in range(NT):
        calc_mu(rates[:,i+1],MU)
        G[E_all] = ri.dphiE(MU[E_all])
        G[I_all] = ri.dphiE(MU[I_all])
        Q += (-Q*dt_tau_inv[:,None] + G[:,None]*(M@Q)*dt)
        # Reorthogonalize Q
        if (i+1) % NONS == 0:
            Q,R = np.linalg.qr(Q,out=(Q,R))
            # After warming up, use R to calculate Lyapunov exponents
            if i > NWONS:
                if save_time:
                    Ls[:,(i-NWONS+1)//NONS-1] = np.log(np.abs(np.diag(R)))
                else:
                    Ls += np.log(np.abs(np.diag(R)))
                if np.any(np.isnan(Ls)):
                    print(R)
        if i+1 == NWONS:
            print("Warmup took",time.process_time() - start,"s\n")
            start = time.process_time()

    print("Full Q evolution took",time.process_time() - start,"s\n")

    if save_time:
        Ls /= NONS*dt
    else:
        Ls /= (NT-NWONS)*dt

    if return_Q:
        return Ls,Q
    else:
        return Ls

def calc_lyapunov_exp_tensor(ri,T,L,M,H,LAM,E_cond,rates,NLE,TWONS,TONS,mult_tau=False,save_time=False,return_Q=False):
    '''
    Calculate top Lyapunov exponents using PyTorch, allowing for GPU acceleration.
    
    Parameters
    ----------
    ri : Ricciardi
        Ricciardi class object for computing the activation function.
    T : tensor
        1D Tensor of time-points to save the rates.
    L : tensor
        1D Tensor of optogenetic input strengths per cell.
    M : tensor
        2D Tensor of recurrent weight matrix.
    H : tensor
        1D Tensor of afferent inputs per cell.
    LAM : float
        Factor by which to multiply the optogentic input strengths.
    E_cond : tensor
        1D Boolean tensor indicating excitatory cells.
    rates : array-like
        2D Array of shape (Ncell x Ntime) with the rates of the cells.
    NLE : int
        Number of Lyapunov exponents to calculate.
    TWONS : float
        Time for warmup prior to computing Lyapunov exponents.
    TONS : float
        Time between measuring Lyapunov exponents.
    mult_tau : bool
        Whether to multiply the input by the time constants of the cells.
    save_time : bool
        Whether to save the Lyapunov exponents at each time step.
    return_Q : bool
        Whether to return the Q matrix used for the calculation.
    
    Returns
    -------
    array-like
        Top NLE Lyapunov exponents. If save_time is True, then Ls is a 2D array of shape (NLE x Ntime) with the Lyapunov exponents at each time step.
    array-like
        Q matrix used for the calculation. Only returned if return_Q is True.
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if len(H) != rates.shape[0]:
        raise("Rates must have shape Ncell x Ntime")

    LAS = LAM*L

    dt = T[1] - T[0]
    dt_tau_inv = (dt / torch.where(E_cond,ri.tE,ri.tI)).to(device)
    NT = len(T) - 1
    NWONS = int(np.round(TWONS/dt))
    NONS = int(np.round(TONS/dt))

    print("NWONS =",NWONS)
    print("NT =",NT)
    print("NONS =",NONS)

    if mult_tau:
        def calc_mu(RATE,MU):
            MU=torch.matmul(M,RATE)
            MU=torch.add(MU,H)
            MU=torch.where(E_cond,ri.tE*MU,ri.tI*MU)
            MU=torch.add(MU,LAS)
    else:
        def calc_mu(RATE,MU):
            MU=torch.matmul(M,RATE)
            MU=torch.add(MU,H + LAS)

    start = time.process_time()

    # Initialize Q
    Q,_ = torch.linalg.qr(torch.rand(len(H),len(H),dtype=torch.float32))
    Q = Q[:,:NLE].to(device)
    R = torch.zeros((NLE,NLE),dtype=torch.float32).to(device)
    G = torch.zeros(len(H),dtype=torch.float32).to(device)
    MU = torch.zeros(len(H),dtype=torch.float32).to(device)

    print("Initializing Q took",time.process_time() - start,"s\n")

    if save_time:
        Ls = np.zeros((NLE,(NT-NWONS)//NONS))
    else:
        Ls = np.zeros((NLE))

    start = time.process_time()

    # Evolve Q
    for i in range(NT):
        calc_mu(rates[:,i+1],MU)
        G=torch.where(E_cond,ri.dphiE_tensor(MU),ri.dphiI_tensor(MU))
        Q += (-Q*dt_tau_inv[:,None] + G[:,None]*torch.matmul(M,Q)*dt)
        # Reorthogonalize Q
        if (i+1) % NONS == 0:
            torch.linalg.qr(Q,out=(Q,R))
            # After warming up, use R to calculate Lyapunov exponents
            if i > NWONS:
                if save_time:
                    Ls[:,(i-NWONS+1)//NONS-1] = np.log(np.abs(np.diag(R.cpu().numpy())))
                else:
                    Ls += np.log(np.abs(np.diag(R.cpu().numpy())))
                if np.any(np.isnan(Ls)):
                    print(R)
        if i+1 == NWONS:
            print("Warmup took",time.process_time() - start,"s\n")
            start = time.process_time()

    print("Full Q evolution took",time.process_time() - start,"s\n")

    if save_time:
        Ls /= NONS*dt
    else:
        Ls /= (NT-NWONS)*dt

    if return_Q:
        return Ls,Q
    else:
        return Ls