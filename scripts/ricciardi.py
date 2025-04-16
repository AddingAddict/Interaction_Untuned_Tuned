'''
Based on code collaboratively written by Alessandro Sanzeni, Agostina Palmigiano, and Tuan Nguyen for Sanzeni et al 2023
'''
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
import torch_interpolations
from scipy.special import erfi
from scipy.integrate import  quad
from scipy.interpolate import interp1d,interpn
from mpmath import fp

sr2 = np.sqrt(2)
sr2pi = np.sqrt(2*np.pi)
srpi = np.sqrt(np.pi)

def int_dawsni_scal(x):
    return -0.5*x**2*fp.hyp2f2(1.0,1.0,1.5,2.0,x**2)
int_dawsni = np.vectorize(int_dawsni_scal)

def expval(fun,us,sigs):
    '''
    Compute expectation value of a given function over normally distributed inputs.
    
    Parameters
    ----------
    fun : function
        Function to be integrated.
    us : float or array-like
        Mean of the normal distribution.
    sigs : float or array-like
        Standard deviation of the normal distribution.
        
    Returns
    -------
    float or array-like
        Expectation value of the function.
    '''
    if np.isscalar(us):
        return quad(lambda z: fun(us+sigs*z)*np.exp(-z**2/2)/sr2pi,-8,8)[0]
    else:
        return [quad(lambda z: fun(us[i]+sigs[i]*z)*np.exp(-z**2/2)/sr2pi,-8,8)[0]
                for i in range(len(us))]

du = 1e-4

def d(fun,u):
    '''
    Compute the derivative of a function using central differences.
    
    Parameters
    ----------
    fun : function
        Function to be differentiated.
    u : float
        Value at which to compute the second derivative.
    '''
    return (fun(u+du)-fun(u-du))/(2*du)

def d2(fun,u):
    '''
    Compute the second derivative of a function using central differences.
    
    Parameters
    ----------
    fun : function
        Function to be differentiated.
    u : float
        Value at which to compute the second derivative.
    '''
    return (fun(u+du)-2*fun(u)+fun(u-du))/du**2
    
class Ricciardi(object):
    
    '''
    Creates an object to compute the Ricciardi nonlinearity.
    '''
    
    def __init__(self,tE=0.02,tI=0.01,trp=0.002,tht=0.02,Vr=0.01,st=0.01):
        '''
        Construct a Ricciardi object.

        Parameters
        ----------
        tE : float
            Excitatory synaptic time constant. Defaults to 0.02.
        tI : float
            Inhibitory synaptic time constant. Defaults to 0.01.
        trp : float
            Time constant for the refractory period. Defaults to 0.002.
        tht : float
            Membrane potential spiking threshold. Defaults to 0.02.
        Vr : float
            Rest potential. Defaults to 0.01.
        st : float
            White noise standard deviation. Defaults to 0.01.
        '''
        # Parameters defined by ale
        self.tE = tE
        self.tI = tI
        self.trp = trp
        self.tht = tht
        self.Vr = Vr
        self.st = st

    def calc_phi(self,u,t):
        '''
        Compute the Ricciardi nonlinearity.
        
        Parameters
        ----------
        u : float
            Membrane potential.
        t : float
            Synaptic time constant.
            
        Returns
        -------
        float
            Firing rate.
        '''
        min_u = (self.Vr-u)/self.st
        max_u = (self.tht-u)/self.st
        r = np.zeros_like(u);

        if np.isscalar(u):
            if(min_u>3):
                r=max_u/t/srpi*np.exp(-max_u**2)
            elif(min_u>-5):
                r=1.0/(self.trp+t*(0.5*np.pi*(erfi(max_u[idx]) - erfi(min_u[idx])) -\
                    2*(int_dawsni(max_u[idx]) - int_dawsni(min_u[idx]))))
            else:
                r=1.0/(self.trp+t*(np.log(abs(min_u)) - np.log(abs(max_u)) +
                                   (0.25*min_u**-2 - 0.1875*min_u**-4 + 0.3125*min_u**-6 -
                                    0.8203125*min_u**-8 + 2.953125*min_u**-10) -
                                   (0.25*max_u**-2 - 0.1875*max_u**-4 + 0.3125*max_u**-6 -
                                    0.8203125*max_u**-8 + 2.953125*max_u**-10)))
        else:
            for idx in range(len(u)):
                if(min_u[idx]>3):
                    r[idx]=max_u[idx]/t/srpi*np.exp(-max_u[idx]**2)
                elif(min_u[idx]>-5):
                    r[idx]=1.0/(self.trp+t*(0.5*np.pi*(erfi(max_u[idx]) - erfi(min_u[idx])) -\
                        2*(int_dawsni(max_u[idx]) - int_dawsni(min_u[idx]))))
                else:
                    r[idx]=1.0/(self.trp+t*(np.log(abs(min_u[idx])) -
                                            np.log(abs(max_u[idx])) +
                                       (0.25*min_u[idx]**-2 - 0.1875*min_u[idx]**-4 +
                                        0.3125*min_u[idx]**-6 - 0.8203125*min_u[idx]**-8 +
                                        2.953125*min_u[idx]**-10) -
                                       (0.25*max_u[idx]**-2 - 0.1875*max_u[idx]**-4 +
                                        0.3125*max_u[idx]**-6 - 0.8203125*max_u[idx]**-8 +
                                        2.953125*max_u[idx]**-10)))
        return r

    def calc_phi_tensor(self,u,t,out=None):
        '''
        Compute the Ricciardi nonlinearity using PyTorch.
        
        Parameters
        ----------
        u : float
            Membrane potential.
        t : float
            Synaptic time constant.
        out : tensor, optional
            Output tensor to store the result. If not provided, a new tensor is created.
            
        Returns
        -------
        tensor
            Firing rate.
        '''
        if not out:
            out = torch.zeros_like(u)
        min_u = (self.Vr-u)/self.st
        max_u = (self.tht-u)/self.st
        r_low = max_u/t/srpi*torch.exp(-max_u**2)
        r_mid = 1.0/(self.trp+t*(srpi*(1+(0.5641895835477563*max_u-0.07310176646978049*max_u**3+
                                        0.019916897946949282*max_u**5-0.001187484601754455*max_u**7+
                                        0.00014245755084666304*max_u**9-4.208652789675569e-6*max_u**11+
                                        2.8330406295105274e-7*max_u**13-3.2731460579579614e-9*max_u**15+
                                        1.263640520928807e-10*max_u**17)/(1.-0.12956950748735815*max_u**2+
                                        0.046412893575273534*max_u**4-0.002486221791373608*max_u**6+
                                        0.000410629108366176*max_u**8-9.781058014448444e-6*max_u**10+
                                        1.0371239952922995e-6*max_u**12-7.166099219321984e-9*max_u**14+
                                        6.85317470793816e-10*max_u**16+1.932753647574705e-12*max_u**18+
                                        4.121310879310989e-14*max_u**20))*
                                    torch.exp(max_u**2)*max_u*(654729075+252702450*max_u**2+79999920*max_u**4+20386080*max_u**6+
                                        4313760*max_u**8+784320*max_u**10+126720*max_u**12+18944*max_u**14+2816*max_u**16+512*max_u**18)/
                                    (654729075+689188500*max_u**2+364864500*max_u**4+129729600*max_u**6+34927200*max_u**8+
                                        7620480*max_u**10+1411200*max_u**12+230400*max_u**14+34560*max_u**16+5120*max_u**18+1024*max_u**20) -
                                srpi*(1+(0.5641895835477563*min_u-0.07310176646978049*min_u**3+
                                        0.019916897946949282*min_u**5-0.001187484601754455*min_u**7+
                                        0.00014245755084666304*min_u**9-4.208652789675569e-6*min_u**11+
                                        2.8330406295105274e-7*min_u**13-3.2731460579579614e-9*min_u**15+
                                        1.263640520928807e-10*min_u**17)/(1.-0.12956950748735815*min_u**2+
                                        0.046412893575273534*min_u**4-0.002486221791373608*min_u**6+
                                        0.000410629108366176*min_u**8-9.781058014448444e-6*min_u**10+
                                        1.0371239952922995e-6*min_u**12-7.166099219321984e-9*min_u**14+
                                        6.85317470793816e-10*min_u**16+1.932753647574705e-12*min_u**18+
                                        4.121310879310989e-14*min_u**20))*
                                    torch.exp(min_u**2)*min_u*(654729075+252702450*min_u**2+79999920*min_u**4+20386080*min_u**6+
                                        4313760*min_u**8+784320*min_u**10+126720*min_u**12+18944*min_u**14+2816*min_u**16+512*min_u**18)/
                                    (654729075+689188500*min_u**2+364864500*min_u**4+129729600*min_u**6+34927200*min_u**8+
                                        7620480*min_u**10+1411200*min_u**12+230400*min_u**14+34560*min_u**16+5120*min_u**18+1024*min_u**20)))
        r_hgh = 1.0/(self.trp+t*(torch.log(torch.abs(min_u)) - torch.log(torch.abs(max_u)) +
                                   (0.25*min_u**-2 - 0.1875*min_u**-4 + 0.3125*min_u**-6 -
                                    0.8203125*min_u**-8 + 2.953125*min_u**-10) -
                                   (0.25*max_u**-2 - 0.1875*max_u**-4 + 0.3125*max_u**-6 -
                                    0.8203125*max_u**-8 + 2.953125*max_u**-10)))
        torch.where(min_u>-3,r_mid,out,out=out)
        torch.where(min_u> 3,r_low,out,out=out)
        torch.where(min_u<=-3,r_hgh,out,out=out)
        return out
        
    def set_up_nonlinearity(self,nameout=None):
        '''
        Computes or loads precomputed excitatory and inhibitory activation functions over a predefined range of membrane potentials and computes a linear interpolation for faster computation.
        
        Parameters
        ----------
        nameout : str, optional
            Name of the output file to save/load the precomputed values. If None, the precomputed values are not saved. Defaults to None.
        '''
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+".pkl", "rb") as handle:
                    out_dict = pickle.load(handle)
                self.phi_int_E=out_dict["phi_int_E"]
                self.phi_int_I=out_dict["phi_int_I"]
                print("Loading previously saved nonlinearity")
                return None
            except:
                print("Calculating nonlinearity")
                save_file = True

        u_tab_max=10.0;
        u_tab=np.linspace(-u_tab_max/5,u_tab_max,int(200000*1.2+1))
        u_tab=np.concatenate(([-10000],u_tab))
        u_tab=np.concatenate((u_tab,[10000]))

        phi_tab_E,phi_tab_I=u_tab*0,u_tab*0;
        # phi_der_tab_E,phi_der_tab_I=u_tab*0,u_tab*0;

        for idx in range(len(phi_tab_E)):
            phi_tab_E[idx]=self.calc_phi(u_tab[idx],self.tE)
            phi_tab_I[idx]=self.calc_phi(u_tab[idx],self.tI)

        self.phi_int_E=interp1d(u_tab, phi_tab_E, kind="linear", fill_value="extrapolate")
        self.phi_int_I=interp1d(u_tab, phi_tab_I, kind="linear", fill_value="extrapolate")

        if save_file:
            out_dict = {"phi_int_E":self.phi_int_E,
                        "phi_int_I":self.phi_int_I}
            with open(nameout+".pkl", 'wb') as handle:
                pickle.dump(out_dict,handle)
        
    def set_up_nonlinearity_tensor(self,nameout=None):
        '''
        Computes or loads precomputed excitatory and inhibitory activation functions over a predefined range of membrane potentials and computes a PyTorch linear interpolation for faster computation.
        
        Parameters
        ----------
        nameout : str, optional
            Name of the output file to save/load the precomputed values. If None, the precomputed values are not saved. Defaults to None.
        '''
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+".pkl", "rb") as handle:
                    out_dict = pickle.load(handle)
                self.phi_int_tensor_E=out_dict["phi_int_tensor_E"]
                self.phi_int_tensor_I=out_dict["phi_int_tensor_I"]
                print("Loading previously saved nonlinearity")
                return None
            except:
                print("Calculating nonlinearity")
                save_file = True

        if hasattr(self,"phi_int_E"):
            u_tab=self.phi_int_E.x
            phi_tab_E=self.phi_int_E.y
            phi_tab_I=self.phi_int_I.y
        else:
            u_tab_max=10.0;
            u_tab=np.linspace(-u_tab_max/5,u_tab_max,int(200000*1.2+1))
            u_tab=np.concatenate(([-10000],u_tab))
            u_tab=np.concatenate((u_tab,[10000]))

            phi_tab_E,phi_tab_I=u_tab*0,u_tab*0;
            # phi_der_tab_E,phi_der_tab_I=u_tab*0,u_tab*0;

            for idx in range(len(phi_tab_E)):
                phi_tab_E[idx]=self.calc_phi(u_tab[idx],self.tE)
                phi_tab_I[idx]=self.calc_phi(u_tab[idx],self.tI)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print("Using",device)

        u_tab_tensor = torch.from_numpy(u_tab.astype(np.float32)).to(device)
        phi_tab_tensor_E = torch.from_numpy(phi_tab_E.astype(np.float32)).to(device)
        phi_tab_tensor_I = torch.from_numpy(phi_tab_I.astype(np.float32)).to(device)

        self.phi_int_tensor_E=torch_interpolations.RegularGridInterpolator((u_tab_tensor,), phi_tab_tensor_E)
        self.phi_int_tensor_I=torch_interpolations.RegularGridInterpolator((u_tab_tensor,), phi_tab_tensor_I)

        if save_file:
            out_dict = {"phi_int_tensor_E":self.phi_int_tensor_E,
                        "phi_int_tensor_I":self.phi_int_tensor_I}
            with open(nameout+".pkl", 'wb') as handle:
                pickle.dump(out_dict,handle)

    def phiE_tensor(self,u):
        # return self.calc_phi_tensor(u,self.tE)
        return self.phi_int_tensor_E(u[None,:])
    def phiI_tensor(self,u):
        # return self.calc_phi_tensor(u,self.tI)
        return self.phi_int_tensor_I(u[None,:])

    def dphiE_tensor(self,u):
        return d(self.phiE_tensor,u)
    def dphiI_tensor(self,u):
        return d(self.phiI_tensor,u)

    def phiE(self,u):
        return self.phi_int_E(u)
    def phiI(self,u):
        return self.phi_int_I(u)

    def dphiE(self,u):
        return d(self.phiE,u)
    def dphiI(self,u):
        return d(self.phiI,u)
    
    