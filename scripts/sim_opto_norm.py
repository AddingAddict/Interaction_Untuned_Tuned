import argparse
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
import time

import sim_util as su
import ricciardi as ric
import integrate as integ

parser = argparse.ArgumentParser()

parser.add_argument("--c1_idx", "-c1",  help="which contrast for peak 1", type=int, default=0)
parser.add_argument("--c2_idx", "-c2",  help="which contrast for peak 2", type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
c1_idx= args["c1_idx"]
c2_idx= args["c2_idx"]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


with open("./../model_data/best_fit.pkl", "rb") as handle:
    res_dict = pickle.load(handle)
prms = res_dict["prms"]
CVh = res_dict["best_monk_eX"]
bX = res_dict["best_monk_bX"]
aXs = res_dict["best_monk_aXs"]
K = prms["K"]
SoriE = prms["SoriE"]
SoriI = prms["SoriI"]

ri = ric.Ricciardi()
ri.set_up_nonlinearity("./phi_int")
ri.set_up_nonlinearity_tensor()

NtE = 50
Nt = NtE*ri.tE
dt = ri.tI/5
T = torch.linspace(0,5*Nt,round(5*Nt/dt)+1)
mask_time = T>(4*Nt)
T_mask = T.cpu().numpy()[mask_time]

N = 10000
Nori = 20
NE = 4*(N//Nori)//5
NI = 1*(N//Nori)//5

prms["Nori"] = Nori
prms["NE"] = NE
prms["NI"] = NI

seeds = np.arange(200)

print("simulating contrast # "+str(c1_idx+1)+" for peak 1, contrast # "+str(c2_idx+1)+" for peak 2")
print("")
cA1 = aXs[c1_idx]/bX
cA2 = aXs[c2_idx]/bX
rX = bX

μrEs = np.zeros((len(seeds),3,Nori))
μrIs = np.zeros((len(seeds),3,Nori))
ΣrEs = np.zeros((len(seeds),4,Nori))
ΣrIs = np.zeros((len(seeds),4,Nori))
μhEs = np.zeros((len(seeds),3,Nori))
μhIs = np.zeros((len(seeds),3,Nori))
ΣhEs = np.zeros((len(seeds),4,Nori))
ΣhIs = np.zeros((len(seeds),4,Nori))
balEs = np.zeros((len(seeds),2,Nori))
balIs = np.zeros((len(seeds),2,Nori))
Lexps = np.zeros((len(seeds),2))
timeouts = np.zeros((len(seeds),2)).astype(bool)

def simulate_networks(prms,rX,cA1,cA2,CVh):
    N = prms.get("Nori",180) * (prms.get("NE",4) + prms.get("NI",1))
    rs = np.zeros((len(seeds),2,N))
    mus = np.zeros((len(seeds),2,N))
    muEs = np.zeros((len(seeds),2,N))
    muIs = np.zeros((len(seeds),2,N))
    Ls = np.zeros((len(seeds),2))
    TOs = np.zeros((len(seeds),2))

    for seed_idx,seed in enumerate(seeds):
        print("simulating seed # "+str(seed_idx+1))
        print("")
        
        start = time.process_time()

        net,this_M,this_H1,this_B,this_LAS,this_EPS = su.gen_ring_disorder_tensor(seed,prms,CVh)
        this_H2 = torch.roll(this_H1,N//4).to(device)
        M = this_M.cpu().numpy()
        H = (rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS).cpu().numpy()
        LAS = this_LAS.cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print("")

        start = time.process_time()
        
        base_sol,base_timeout = integ.sim_dyn_tensor(ri,T,0,this_M,rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,0] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],0,this_M,
                                                               rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS,this_LAS,
                                                               net.C_conds[0],base_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,0] = np.mean(base_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,0] = base_timeout

        print("Integrating base network took ",time.process_time() - start," s")
        print("")

        start = time.process_time()
        
        opto_sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1,this_M,rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,1] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],1,this_M,
                                                               rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS,this_LAS,
                                                               net.C_conds[0],opto_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,1] = np.mean(opto_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,1] = opto_timeout

        print("Integrating opto network took ",time.process_time() - start," s")
        print("")

        start = time.process_time()

        muEs[seed_idx] = (M[:,net.C_all[0]]@rs[seed_idx,:,net.C_all[0]]).T + H[None,:]
        muIs[seed_idx] = (M[:,net.C_all[1]]@rs[seed_idx,:,net.C_all[1]]).T
        muEs[seed_idx,:,net.C_all[0]] *= ri.tE
        muEs[seed_idx,:,net.C_all[1]] *= ri.tI
        muIs[seed_idx,:,net.C_all[0]] *= ri.tE
        muIs[seed_idx,:,net.C_all[1]] *= ri.tI
        muEs[seed_idx,1] = muEs[seed_idx,1] + LAS
        mus[seed_idx] = muEs[seed_idx] + muIs[seed_idx]

        print("Calculating statistics took ",time.process_time() - start," s")
        print("")

    return net,rs,mus,muEs,muIs,Ls,TOs

# Simulate network
this_prms = prms.copy()

net,rs,mus,muEs,muIs,Ls,TOs = simulate_networks(this_prms,rX,cA1,cA2,CVh)

start = time.process_time()

for nloc in range(Nori):
    μrEs[:,:2,nloc] = np.mean(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    μrIs[:,:2,nloc] = np.mean(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    ΣrEs[:,:2,nloc] = np.var(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    ΣrIs[:,:2,nloc] = np.var(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    μhEs[:,:2,nloc] = np.mean(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    μhIs[:,:2,nloc] = np.mean(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    ΣhEs[:,:2,nloc] = np.var(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    ΣhIs[:,:2,nloc] = np.var(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    balEs[:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[0][nloc]])/muEs[:,:,net.C_idxs[0][nloc]],axis=-1)
    balIs[:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[1][nloc]])/muEs[:,:,net.C_idxs[1][nloc]],axis=-1)

    μrEs[:,2,nloc] = np.mean(rs[:,1,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
    μrIs[:,2,nloc] = np.mean(rs[:,1,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
    ΣrEs[:,2,nloc] = np.var(rs[:,1,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
    ΣrIs[:,2,nloc] = np.var(rs[:,1,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
    μhEs[:,2,nloc] = np.mean(mus[:,1,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
    μhIs[:,2,nloc] = np.mean(mus[:,1,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)
    ΣhEs[:,2,nloc] = np.var(mus[:,1,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
    ΣhIs[:,2,nloc] = np.var(mus[:,1,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)

    for seed_idx in range(len(seeds)):
        ΣrEs[seed_idx,3,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[0][nloc]],
                                       rs[seed_idx,1,net.C_idxs[0][nloc]]-rs[seed_idx,0,net.C_idxs[0][nloc]])[0,1]
        ΣrIs[seed_idx,3,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[1][nloc]],
                                       rs[seed_idx,1,net.C_idxs[1][nloc]]-rs[seed_idx,0,net.C_idxs[1][nloc]])[0,1]
Lexps[:,:] = Ls
timeouts[:,:] = TOs

seed_mask = np.logical_not(np.any(timeouts,axis=-1))
vsm1_mask = net.get_oriented_neurons(delta_ori=4.5,)[0]
vsm2_mask = net.get_oriented_neurons(delta_ori=4.5,vis_ori=45)[0]

base_rates = rs[:,0,:]
opto_rates = rs[:,1,:]
diff_rates = opto_rates - base_rates

all_base_means = np.mean(base_rates[seed_mask,:])
all_base_stds = np.std(base_rates[seed_mask,:])
all_opto_means = np.mean(opto_rates[seed_mask,:])
all_opto_stds = np.std(opto_rates[seed_mask,:])
all_diff_means = np.mean(diff_rates[seed_mask,:])
all_diff_stds = np.std(diff_rates[seed_mask,:])
all_norm_covs = np.cov(base_rates[seed_mask,:].flatten(),
    diff_rates[seed_mask,:].flatten())[0,1] / all_diff_stds**2

vsm1_base_means = np.mean(base_rates[seed_mask,:][:,vsm1_mask])
vsm1_base_stds = np.std(base_rates[seed_mask,:][:,vsm1_mask])
vsm1_opto_means = np.mean(opto_rates[seed_mask,:][:,vsm1_mask])
vsm1_opto_stds = np.std(opto_rates[seed_mask,:][:,vsm1_mask])
vsm1_diff_means = np.mean(diff_rates[seed_mask,:][:,vsm1_mask])
vsm1_diff_stds = np.std(diff_rates[seed_mask,:][:,vsm1_mask])
vsm1_norm_covs = np.cov(base_rates[seed_mask,:][:,vsm1_mask].flatten(),
    diff_rates[seed_mask,:][:,vsm1_mask].flatten())[0,1] / vsm1_diff_stds**2

vsm2_base_means = np.mean(base_rates[seed_mask,:][:,vsm2_mask])
vsm2_base_stds = np.std(base_rates[seed_mask,:][:,vsm2_mask])
vsm2_opto_means = np.mean(opto_rates[seed_mask,:][:,vsm2_mask])
vsm2_opto_stds = np.std(opto_rates[seed_mask,:][:,vsm2_mask])
vsm2_diff_means = np.mean(diff_rates[seed_mask,:][:,vsm2_mask])
vsm2_diff_stds = np.std(diff_rates[seed_mask,:][:,vsm2_mask])
vsm2_norm_covs = np.cov(base_rates[seed_mask,:][:,vsm2_mask].flatten(),
    diff_rates[seed_mask,:][:,vsm2_mask].flatten())[0,1] / vsm2_diff_stds**2

print("Saving statistics took ",time.process_time() - start," s")
print("")

res_dict = {}
res_dict["prms"] = this_prms
res_dict["μrEs"] = μrEs
res_dict["μrIs"] = μrIs
res_dict["ΣrEs"] = ΣrEs
res_dict["ΣrIs"] = ΣrIs
res_dict["μhEs"] = μhEs
res_dict["μhIs"] = μhIs
res_dict["ΣhEs"] = ΣhEs
res_dict["ΣhIs"] = ΣhIs
res_dict["balEs"] = balEs
res_dict["balIs"] = balIs
res_dict["Lexps"] = Lexps
res_dict["all_base_means"] = all_base_means
res_dict["all_base_stds"] = all_base_stds
res_dict["all_opto_means"] = all_opto_means
res_dict["all_opto_stds"] = all_opto_stds
res_dict["all_diff_means"] = all_diff_means
res_dict["all_diff_stds"] = all_diff_stds
res_dict["all_norm_covs"] = all_norm_covs
res_dict["vsm1_base_means"] = vsm1_base_means
res_dict["vsm1_base_stds"] = vsm1_base_stds
res_dict["vsm1_opto_means"] = vsm1_opto_means
res_dict["vsm1_opto_stds"] = vsm1_opto_stds
res_dict["vsm1_diff_means"] = vsm1_diff_means
res_dict["vsm1_diff_stds"] = vsm1_diff_stds
res_dict["vsm1_norm_covs"] = vsm1_norm_covs
res_dict["vsm2_base_means"] = vsm2_base_means
res_dict["vsm2_base_stds"] = vsm2_base_stds
res_dict["vsm2_opto_means"] = vsm2_opto_means
res_dict["vsm2_opto_stds"] = vsm2_opto_stds
res_dict["vsm2_diff_means"] = vsm2_diff_means
res_dict["vsm2_diff_stds"] = vsm2_diff_stds
res_dict["vsm2_norm_covs"] = vsm2_norm_covs
res_dict["timeouts"] = timeouts

res_file = "./../results/opto_norm_c1_{:d}_c2_{:d}".format(c1_idx,c2_idx)

with open(res_file+".pkl", "wb") as handle:
    pickle.dump(res_dict,handle)
