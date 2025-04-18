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
import dmft as dmft

parser = argparse.ArgumentParser()

parser.add_argument('--width_idx', '-w',  help='which width', type=int, default=0)
parser.add_argument('--J_idx', '-j',  help='which J', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
width_idx= args['width_idx']
J_idx= args['J_idx']

with open("./../model_data/best_fit.pkl", "rb") as handle:
    res_dict = pickle.load(handle)
prms = res_dict['prms']
CVh = res_dict['best_monk_eX']
bX = res_dict['best_monk_bX']
aXs = res_dict['best_monk_aXs']
K = prms['K']
SoriE = prms['SoriE']
SoriI = prms['SoriI']
J = prms['J']
L = prms['L']

ri = ric.Ricciardi()
ri.set_up_nonlinearity('./phi_int')
ri.set_up_nonlinearity_tensor()

NtE = 50
Nt = NtE*ri.tE
dt = ri.tI/5
T = torch.linspace(0,5*Nt,round(5*Nt/dt)+1)
mask_time = T>(4*Nt)
T_mask = T.cpu().numpy()[mask_time]

N = 10000
Nori = [80,50,40,20,16,10, 8][width_idx]
NE = 4*(N//Nori)//5
NI = 1*(N//Nori)//5

prms['Nori'] = Nori
prms['NE'] = NE
prms['NI'] = NI

seeds = np.arange(100)

widths = 4**(2*np.arange(0,6+1)/6 - 1)
Js = J*8**(2*np.arange(0,6+1)/6 - 2/3)

print('simulating width # '+str(width_idx+1))
print('')
width = widths[width_idx]

print('simulating J # '+str(J_idx+1))
print('')
newJ = Js[J_idx]

cA = aXs[-1]/bX
rX = bX

μrEs = np.zeros((len(seeds),3,Nori))
μrIs = np.zeros((len(seeds),3,Nori))
ΣrEs = np.zeros((len(seeds),4,Nori))
ΣrIs = np.zeros((len(seeds),4,Nori))
μhEs = np.zeros((len(seeds),3,Nori))
μhIs = np.zeros((len(seeds),3,Nori))
ΣhEs = np.zeros((len(seeds),4,Nori))
ΣhIs = np.zeros((len(seeds),4,Nori))
μhEEs = np.zeros((len(seeds),2,Nori))
μhEIs = np.zeros((len(seeds),2,Nori))
μhIEs = np.zeros((len(seeds),2,Nori))
μhIIs = np.zeros((len(seeds),2,Nori))
ΣhEEs = np.zeros((len(seeds),2,Nori))
ΣhEIs = np.zeros((len(seeds),2,Nori))
ΣhIEs = np.zeros((len(seeds),2,Nori))
ΣhIIs = np.zeros((len(seeds),2,Nori))
balEs = np.zeros((len(seeds),2,Nori))
balIs = np.zeros((len(seeds),2,Nori))
Lexps = np.zeros((len(seeds),2))
timeouts = np.zeros((len(seeds),2)).astype(bool)

if J_idx < 4:
    method = None
else:
    method = 'rk4'

def simulate_networks(prms,rX,cA,CVh):
    N = prms.get('Nori',180) * (prms.get('NE',4) + prms.get('NI',1))
    rs = np.zeros((len(seeds),2,N))
    mus = np.zeros((len(seeds),2,N))
    muEs = np.zeros((len(seeds),2,N))
    muIs = np.zeros((len(seeds),2,N))
    Ls = np.zeros((len(seeds),2))
    TOs = np.zeros((len(seeds),2))

    for seed_idx,seed in enumerate(seeds):
        print('simulating seed # '+str(seed_idx+1))
        print('')
        
        start = time.process_time()

        net,this_M,this_H,this_B,this_LAS,this_EPS = su.gen_ring_disorder_tensor(seed,prms,CVh)
        M = this_M.cpu().numpy()
        H = (rX*(this_B+cA*this_H)*this_EPS).cpu().numpy()
        LAS = this_LAS.cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        base_sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30,method=method)
        Ls[seed_idx,0] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],0.0,this_M,
                                                               rX*(this_B+cA*this_H)*this_EPS,this_LAS,
                                                               net.C_conds[0],base_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,0] = np.mean(base_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,0] = base_timeout

        print("Integrating base network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()
        
        opto_sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30,method=method)
        Ls[seed_idx,1] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],1.0,this_M,
                                                               rX*(this_B+cA*this_H)*this_EPS,this_LAS,
                                                               net.C_conds[0],opto_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,1] = np.mean(opto_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,1] = opto_timeout

        print("Integrating opto network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        muEs[seed_idx,0] = M[:,net.C_all[0]]@rs[seed_idx,0,net.C_all[0]] + H
        muIs[seed_idx,0] = M[:,net.C_all[1]]@rs[seed_idx,0,net.C_all[1]]
        muEs[seed_idx,0,net.C_all[0]] *= ri.tE
        muEs[seed_idx,0,net.C_all[1]] *= ri.tI
        muIs[seed_idx,0,net.C_all[0]] *= ri.tE
        muIs[seed_idx,0,net.C_all[1]] *= ri.tI
        mus[seed_idx,0] = muEs[seed_idx,0] + muIs[seed_idx,0]

        muEs[seed_idx,1] = M[:,net.C_all[0]]@rs[seed_idx,1,net.C_all[0]] + H
        muIs[seed_idx,1] = M[:,net.C_all[1]]@rs[seed_idx,1,net.C_all[1]]
        muEs[seed_idx,1,net.C_all[0]] *= ri.tE
        muEs[seed_idx,1,net.C_all[1]] *= ri.tI
        muIs[seed_idx,1,net.C_all[0]] *= ri.tE
        muIs[seed_idx,1,net.C_all[1]] *= ri.tI
        muEs[seed_idx,1] = muEs[seed_idx,1] + LAS
        mus[seed_idx,1] = muEs[seed_idx,1] + muIs[seed_idx,1]

        print("Calculating statistics took ",time.process_time() - start," s")
        print('')

    return net,rs,mus,muEs,muIs,Ls,TOs

# Simulate network where widthure is removed by increasing baseline fraction
print('simulating baseline fraction network')
print('')
this_prms = prms.copy()
this_prms['J'] = newJ
this_prms['SoriF'] *= width
this_prms['baseinp'] = dmft.wrapnormdens(90,this_prms['SoriF']) / dmft.wrapnormdens(0,this_prms['SoriF'])
# this_prms['basefrac'] = 1-width

net,rs,mus,muEs,muIs,Ls,TOs = simulate_networks(this_prms,rX,cA,CVh)

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
    μhEEs[:,:2,nloc] = np.mean(muEs[:,:,net.C_idxs[0][nloc]],axis=-1)
    μhIEs[:,:2,nloc] = np.mean(muEs[:,:,net.C_idxs[1][nloc]],axis=-1)
    μhEIs[:,:2,nloc] = np.mean(muIs[:,:,net.C_idxs[0][nloc]],axis=-1)
    μhIIs[:,:2,nloc] = np.mean(muIs[:,:,net.C_idxs[1][nloc]],axis=-1)
    ΣhEEs[:,:2,nloc] = np.var(muEs[:,:,net.C_idxs[0][nloc]],axis=-1)
    ΣhIEs[:,:2,nloc] = np.var(muEs[:,:,net.C_idxs[1][nloc]],axis=-1)
    ΣhEIs[:,:2,nloc] = np.var(muIs[:,:,net.C_idxs[0][nloc]],axis=-1)
    ΣhIIs[:,:2,nloc] = np.var(muIs[:,:,net.C_idxs[1][nloc]],axis=-1)
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
vsm_mask = net.get_oriented_neurons(delta_ori=4.5*width)[0]
osm_mask = net.get_oriented_neurons(delta_ori=4.5*width,vis_ori=90)[0]

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

vsm_base_means = np.mean(base_rates[seed_mask,:][:,vsm_mask])
vsm_base_stds = np.std(base_rates[seed_mask,:][:,vsm_mask])
vsm_opto_means = np.mean(opto_rates[seed_mask,:][:,vsm_mask])
vsm_opto_stds = np.std(opto_rates[seed_mask,:][:,vsm_mask])
vsm_diff_means = np.mean(diff_rates[seed_mask,:][:,vsm_mask])
vsm_diff_stds = np.std(diff_rates[seed_mask,:][:,vsm_mask])
vsm_norm_covs = np.cov(base_rates[seed_mask,:][:,vsm_mask].flatten(),
    diff_rates[seed_mask,:][:,vsm_mask].flatten())[0,1] / vsm_diff_stds**2

osm_base_means = np.mean(base_rates[seed_mask,:][:,osm_mask])
osm_base_stds = np.std(base_rates[seed_mask,:][:,osm_mask])
osm_opto_means = np.mean(opto_rates[seed_mask,:][:,osm_mask])
osm_opto_stds = np.std(opto_rates[seed_mask,:][:,osm_mask])
osm_diff_means = np.mean(diff_rates[seed_mask,:][:,osm_mask])
osm_diff_stds = np.std(diff_rates[seed_mask,:][:,osm_mask])
osm_norm_covs = np.cov(base_rates[seed_mask,:][:,osm_mask].flatten(),
    diff_rates[seed_mask,:][:,osm_mask].flatten())[0,1] / osm_diff_stds**2

print("Saving statistics took ",time.process_time() - start," s")
print('')

res_dict = {}
res_dict['prms'] = this_prms
res_dict['μrEs'] = μrEs
res_dict['μrIs'] = μrIs
res_dict['ΣrEs'] = ΣrEs
res_dict['ΣrIs'] = ΣrIs
res_dict['μhEs'] = μhEs
res_dict['μhIs'] = μhIs
res_dict['ΣhEs'] = ΣhEs
res_dict['ΣhIs'] = ΣhIs
res_dict['μhEEs'] = μhEEs
res_dict['μhEIs'] = μhEIs
res_dict['μhIEs'] = μhIEs
res_dict['μhIIs'] = μhIIs
res_dict['ΣhEEs'] = ΣhEEs
res_dict['ΣhEIs'] = ΣhEIs
res_dict['ΣhIEs'] = ΣhIEs
res_dict['ΣhIIs'] = ΣhIIs
res_dict['balEs'] = balEs
res_dict['balIs'] = balIs
res_dict['Lexps'] = Lexps
res_dict['all_base_means'] = all_base_means
res_dict['all_base_stds'] = all_base_stds
res_dict['all_opto_means'] = all_opto_means
res_dict['all_opto_stds'] = all_opto_stds
res_dict['all_diff_means'] = all_diff_means
res_dict['all_diff_stds'] = all_diff_stds
res_dict['all_norm_covs'] = all_norm_covs
res_dict['vsm_base_means'] = vsm_base_means
res_dict['vsm_base_stds'] = vsm_base_stds
res_dict['vsm_opto_means'] = vsm_opto_means
res_dict['vsm_opto_stds'] = vsm_opto_stds
res_dict['vsm_diff_means'] = vsm_diff_means
res_dict['vsm_diff_stds'] = vsm_diff_stds
res_dict['vsm_norm_covs'] = vsm_norm_covs
res_dict['osm_base_means'] = osm_base_means
res_dict['osm_base_stds'] = osm_base_stds
res_dict['osm_opto_means'] = osm_opto_means
res_dict['osm_opto_stds'] = osm_opto_stds
res_dict['osm_diff_means'] = osm_diff_means
res_dict['osm_diff_stds'] = osm_diff_stds
res_dict['osm_norm_covs'] = osm_norm_covs
res_dict['timeouts'] = timeouts

with open('./../results/vary_ffwidth_{:d}_J_{:d}'.format(width_idx,J_idx)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
