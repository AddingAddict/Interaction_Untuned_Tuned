import argparse
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
import torch_interpolations as torchitp
from torchquad import Simpson, set_up_backend

import time

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
set_up_backend("torch", data_type="float32")
simp = Simpson()

parser = argparse.ArgumentParser(description=("Computes the rate moments over evenly spaced samples of 3D grid of input means, input standard deviations, and input correlations for given optogenetic parameters"))

parser.add_argument("--L", "-L",  help="Laser strength", type=float, default=1.0)
parser.add_argument("--CVL", "-CVL",  help="Laser strength", type=float, default=1.0)
args = vars(parser.parse_args())
print(parser.parse_args())
L= args["L"]
CVL= args["CVL"]

resultsdir="./"
print("Saving all results in "+  resultsdir)
print(" ")

if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)

lamL = 1e-3*L
LNSig = np.log(1+CVL**2)
LNsig = np.sqrt(LNSig)
LNmu = np.log(lamL)-0.5*LNSig

start = time.process_time()

with open(resultsdir+"itp_ranges"+".pkl", "rb") as handle:
    ranges_dict = pickle.load(handle)

phixrange = ranges_dict["Ph"]["xrange"]
phixs = np.linspace(phixrange[0],phixrange[1],round(phixrange[2])).astype(np.float32)
phiEs = np.load(resultsdir+"PhE_itp.npy").astype(np.float32)
phiIs = np.load(resultsdir+"PhI_itp.npy").astype(np.float32)
phixs_torch = torch.from_numpy(phixs).to(device)
phiEs_torch = torch.from_numpy(phiEs).to(device)
phiIs_torch = torch.from_numpy(phiIs).to(device)

Mxrange = ranges_dict["M"]["xrange"]
Mxs = np.linspace(Mxrange[0],Mxrange[1],round(Mxrange[2])).astype(np.float32)
Msigrange = ranges_dict["M"]["sigrange"]
Msigs = np.linspace(Msigrange[0],Msigrange[1],round(Msigrange[2])).astype(np.float32)
MEs = np.load(resultsdir+"ME_itp.npy").astype(np.float32)
MIs = np.load(resultsdir+"MI_itp.npy").astype(np.float32)
Mxs_torch = torch.from_numpy(Mxs).to(device)
Msigs_torch = torch.from_numpy(Msigs).to(device)
MEs_torch = torch.from_numpy(MEs).to(device)
MIs_torch = torch.from_numpy(MIs).to(device)

Cxrange = ranges_dict["C"]["xrange"]
Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
Csigrange = ranges_dict["C"]["sigrange"]
Csigs = np.linspace(Csigrange[0],Csigrange[1],round(Csigrange[2])).astype(np.float32)
Ccrange = ranges_dict["C"]["crange"]
Ccs = np.linspace(Ccrange[0],Ccrange[1],round(Ccrange[2])).astype(np.float32)
CEs = np.load(resultsdir+"CE_itp.npy").astype(np.float32)
CIs = np.load(resultsdir+"CI_itp.npy").astype(np.float32)
Cxs_torch = torch.from_numpy(Cxs).to(device)
Csigs_torch = torch.from_numpy(Csigs).to(device)
Ccs_torch = torch.from_numpy(Ccs).to(device)
CEs_torch = torch.from_numpy(CEs).to(device)
CIs_torch = torch.from_numpy(CIs).to(device)

phiE_itp = torchitp.RegularGridInterpolator((phixs_torch,),phiEs_torch)
phiI_itp = torchitp.RegularGridInterpolator((phixs_torch,),phiIs_torch)

ME_itp = torchitp.RegularGridInterpolator((Msigs_torch,Mxs_torch),MEs_torch)
MI_itp = torchitp.RegularGridInterpolator((Msigs_torch,Mxs_torch),MIs_torch)

CE_itp = torchitp.RegularGridInterpolator((Ccs_torch,Csigs_torch,Cxs_torch),CEs_torch)
CI_itp = torchitp.RegularGridInterpolator((Ccs_torch,Csigs_torch,Cxs_torch),CIs_torch)

sr2 = np.sqrt(2)
sr2π = np.sqrt(2*np.pi)

def mutox(mu):
    return np.sign(mu/100-0.2)*np.abs(mu/100-0.2)**0.5

def xtomu(x):
    return 100*(np.sign(x)*np.abs(x)**2.0+0.2)

def mutox_torch(mu):
    return torch.sign(mu/100-0.2)*torch.abs(mu/100-0.2)**0.5

def xtomu_torch(x):
    return 100*(torch.sign(x)*torch.abs(x)**2.0+0.2)

def phiE(mu):
    try:
        return phiE_itp(mutox_torch(1e3*mu)[None,:])
    except:
        return phiE_itp(torch.tensor([mutox_torch(1e3*mu)]))
def phiI(mu):
    try:
        return phiI_itp(mutox_torch(1e3*mu)[None,:])
    except:
        return phiI_itp(torch.tensor([mutox_torch(1e3*mu)]))

def ME(mu,Sig):
    try:
        return ME_itp(torch.row_stack(torch.broadcast_tensors(1e3*torch.sqrt(Sig),mutox_torch(1e3*mu))))
    except:
        return ME_itp(torch.tensor([[1e3*torch.sqrt(Sig)],[mutox_torch(1e3*mu)]]))
def MI(mu,Sig):
    try:
        return MI_itp(torch.row_stack(torch.broadcast_tensors(1e3*torch.sqrt(Sig),mutox_torch(1e3*mu))))
    except:
        return MI_itp(torch.tensor([[1e3*torch.sqrt(Sig)],[mutox_torch(1e3*mu)]]))
def ME_vecmu(mu,Sig):
    return ME_itp(torch.stack((1e3*torch.sqrt(Sig)*torch.ones_like(mu),mutox_torch(1e3*mu)),dim=0))
def MI_vecmu(mu,Sig):
    return MI_itp(torch.stack((1e3*torch.sqrt(Sig)*torch.ones_like(mu),mutox_torch(1e3*mu)),dim=0))

def CE(mu,Sig,k):
    c = torch.sign(k)*torch.fmin(torch.abs(k)/Sig,torch.tensor(1))
    try:
        return CE_itp(torch.row_stack(torch.broadcast_tensors(c,1e3*torch.sqrt(Sig),mutox_torch(1e3*mu))))
    except:
        return CE_itp(torch.tensor([[c],[1e3*torch.sqrt(Sig)],[mutox_torch(1e3*mu)]]))
def CI(mu,Sig,k):
    c = torch.sign(k)*torch.fmin(torch.abs(k)/Sig,torch.tensor(1))
    try:
        return CI_itp(torch.row_stack(torch.broadcast_tensors(c,1e3*torch.sqrt(Sig),mutox_torch(1e3*mu))))
    except:
        return CI_itp(torch.tensor([[c],[1e3*torch.sqrt(Sig)],[mutox_torch(1e3*mu)]]))
def CE_vecmu(mu,Sig,k):
    c = torch.sign(k)*torch.fmin(torch.abs(k)/Sig,torch.tensor(1))
    return CE_itp(torch.stack((c*torch.ones_like(mu),1e3*torch.sqrt(Sig)*torch.ones_like(mu),mutox_torch(1e3*mu)),dim=0))
def CI_vecmu(mu,Sig,k):
    c = torch.sign(k)*torch.fmin(torch.abs(k)/Sig,torch.tensor(1))
    return CI_itp(torch.stack((c*torch.ones_like(mu),1e3*torch.sqrt(Sig)*torch.ones_like(mu),mutox_torch(1e3*mu)),dim=0))

Nint = 500001
# Nint = 101

def phiLint(mu):
    return simp.integrate(lambda x: torch.exp(-0.5*((torch.log(x)-LNmu)/LNsig)**2)/(sr2π*LNsig*x)*phiE(mu+x),
        dim=1,N=Nint,integration_domain=[[1e-12,100*lamL]],backend="torch").cpu().numpy()

def MLint(mu,Sig):
    return simp.integrate(lambda x: torch.exp(-0.5*((torch.log(x)-LNmu)/LNsig)**2)/(sr2π*LNsig*x)*ME_vecmu(mu+x,Sig),
        dim=1,N=Nint,integration_domain=[[1e-12,100*lamL]],backend="torch").cpu().numpy()

def CLint(mu,Sig,k):
    return simp.integrate(lambda x: torch.exp(-0.5*((torch.log(x)-LNmu)/LNsig)**2)/(sr2π*LNsig*x)*CE_vecmu(mu+x,Sig,k),
        dim=1,N=Nint,integration_domain=[[1e-12,100*lamL]],backend="torch").cpu().numpy()

print("Interpolating base moments took ",time.process_time() - start," s")
print("")

start = time.process_time()
    
phixrange = ranges_dict["PhL"]["xrange"]
phixs = np.linspace(phixrange[0],phixrange[1],round(phixrange[2])).astype(np.float32)
phixs_torch = torch.from_numpy(phixs).to(device)

try:
    phiLs = np.load(resultsdir+"PhL_itp"+"_L={:.2f}".format(L)+"_CVL={:.2f}".format(CVL)+".npy").astype(np.float32)
except:
    phiLs = np.zeros((len(phixs)),dtype=np.float32)

    for x_idx,x in enumerate(phixs_torch):
        phiLs[x_idx] = phiLint(xtomu_torch(x)*1e-3)
        
    # phiL_itp = torchitp.RegularGridInterpolator((phixs,),phiLs)

    np.save(resultsdir+"PhL_itp"+"_L={:.2f}".format(L)+"_CVL={:.2f}".format(CVL),phiLs)

print("Interpolating phiL took ",time.process_time() - start," s")
print("")

start = time.process_time()

Mxrange = ranges_dict["ML"]["xrange"]
Mxs = np.linspace(Mxrange[0],Mxrange[1],round(Mxrange[2])).astype(np.float32)
Msigrange = ranges_dict["ML"]["sigrange"]
Msigs = np.linspace(Msigrange[0],Msigrange[1],round(Msigrange[2])).astype(np.float32)
Mxs_torch = torch.from_numpy(Mxs).to(device)
Msigs_torch = torch.from_numpy(Msigs).to(device)

try:
    MLs = np.load(resultsdir+"ML_itp"+"_L={:.2f}".format(L)+"_CVL={:.2f}".format(CVL)+".npy").astype(np.float32)
except:
    MLs = np.zeros((len(Msigs),len(Mxs)),dtype=np.float32)

    for sig_idx,sig in enumerate(Msigs_torch):
        for x_idx,x in enumerate(Mxs_torch):
            MLs[sig_idx,x_idx] = MLint(xtomu_torch(x)*1e-3,(sig*1e-3)**2)

    np.save(resultsdir+"ML_itp"+"_L={:.2f}".format(L)+"_CVL={:.2f}".format(CVL),MLs)

print("Interpolating ML took ",time.process_time() - start," s")
print("")

start = time.process_time()

Cxrange = ranges_dict["CL"]["xrange"]
Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
Csigrange = ranges_dict["CL"]["sigrange"]
Csigs = np.linspace(Csigrange[0],Csigrange[1],round(Csigrange[2])).astype(np.float32)
Ccrange = ranges_dict["CL"]["crange"]
Ccs = np.linspace(Ccrange[0],Ccrange[1],round(Ccrange[2])).astype(np.float32)
Cxs_torch = torch.from_numpy(Cxs).to(device)
Csigs_torch = torch.from_numpy(Csigs).to(device)
Ccs_torch = torch.from_numpy(Ccs).to(device)

try:
    CLs = np.load(resultsdir+"CL_itp"+"_L={:.2f}".format(L)+"_CVL={:.2f}".format(CVL)+".npy").astype(np.float32)
except:
    CLs = np.zeros((len(Ccs),len(Csigs),len(Cxs)),dtype=np.float32)

    for c_idx,c in enumerate(Ccs_torch):
        for sig_idx,sig in enumerate(Csigs_torch):
            for x_idx,x in enumerate(Cxs_torch):
                CLs[c_idx,sig_idx,x_idx] = CLint(xtomu_torch(x)*1e-3,(sig*1e-3)**2,c*(sig*1e-3)**2)

    np.save(resultsdir+"CL_itp"+"_L={:.2f}".format(L)+"_CVL={:.2f}".format(CVL),CLs)

print("Interpolating CL took ",time.process_time() - start," s")
print("")
