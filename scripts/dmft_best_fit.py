import argparse
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import time

import ricciardi as ric
import dmft

parser = argparse.ArgumentParser()

parser.add_argument("--c_idx", "-c",  help="which contrast", type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
c_idx= args["c_idx"]

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

Twrm = 1.2
Tsav = 0.4
Tsim = 1.0
dt = 0.01/5

Nori = 20

print("simulating contrast # "+str(c_idx+1))
print("")
aXs = np.concatenate([aXs,aXs[-1]*np.arange(1.0+0.2,2.0+0.2,0.2)])
aX = aXs[c_idx]

cA = aX/bX
rX = bX

μrEs = np.zeros((3,Nori))
μrIs = np.zeros((3,Nori))
ΣrEs = np.zeros((4,Nori))
ΣrIs = np.zeros((4,Nori))
μhEs = np.zeros((3,Nori))
μhIs = np.zeros((3,Nori))
ΣhEs = np.zeros((4,Nori))
ΣhIs = np.zeros((4,Nori))
balEs = np.zeros((2,Nori))
balIs = np.zeros((2,Nori))
normCEs = np.zeros((3,Nori))
normCIs = np.zeros((3,Nori))
convs = np.zeros((2,3)).astype(bool)

def predict_networks(prms,rX,cA,CVh):
    tau = np.array([ri.tE,ri.tI],dtype=np.float32)
    W = prms["J"]*np.array([[1,-prms["gE"]],[1./prms["beta"],-prms["gI"]/prms["beta"]]],dtype=np.float32)
    Ks = (1-prms.get("basefrac",0))*np.array([prms["K"],prms["K"]/4],dtype=np.float32)
    Kbs =   prms.get("basefrac",0) *np.array([prms["K"],prms["K"]/4],dtype=np.float32)
    Hb = rX*(1+prms.get("basefrac",0)*cA)*prms["K"]*prms["J"]*\
        np.array([prms["hE"],prms["hI"]/prms["beta"]],dtype=np.float32)
    Hp = rX*(1+                       cA)*prms["K"]*prms["J"]*\
        np.array([prms["hE"],prms["hI"]/prms["beta"]],dtype=np.float32)
    eH = CVh
    sW = np.array([[prms["SoriE"],prms["SoriI"]],[prms["SoriE"],prms["SoriI"]]],dtype=np.float32)
    sH = np.array([prms["SoriF"],prms["SoriF"]],dtype=np.float32)
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    sW2 = sW**2
    sH2 = sH**2
    
    muHb = tau*Hb
    muHp = tau*Hp
    smuH = sH
    SigHb = (muHb*eH)**2
    SigHp = (muHp*eH)**2
    sSigH = 2*sH

    μrs = np.zeros((2,3,Nori))
    Σrs = np.zeros((2,4,Nori))
    μmuEs = np.zeros((2,3,Nori))
    ΣmuEs = np.zeros((2,4,Nori))
    μmuIs = np.zeros((2,3,Nori))
    ΣmuIs = np.zeros((2,4,Nori))
    normC = np.zeros((2,3,Nori))
    conv = np.zeros((2,3)).astype(bool)
    
    oris = np.arange(Nori)*180/Nori
    oris[oris > 90] = 180 - oris[oris > 90]
        
    def gauss(x,b,p,s):
        return b + (p-b)*dmft.basesubwrapnorm(x,s)
    
    if cA == 0 or prms.get("basefrac",0)==1:
        res_dict = dmft.run_two_stage_dmft(prms,rX*(1+cA),CVh,"./../results",ri,Twrm,Tsav,dt)
        rvb = res_dict["r"][:2]
        rvp = res_dict["r"][:2]
        srv = 1e2*np.ones(2)
        rob = res_dict["r"][2:]
        rop = res_dict["r"][2:]
        sro = 1e2*np.ones(2)
        Crvb = dmft.grid_stat(np.mean,res_dict["Cr"][:2],Tsim,dt)
        Crvp = dmft.grid_stat(np.mean,res_dict["Cr"][:2],Tsim,dt)
        sCrv = 1e2*np.ones(2)
        Crob = dmft.grid_stat(np.mean,res_dict["Cr"][2:],Tsim,dt)
        Crop = dmft.grid_stat(np.mean,res_dict["Cr"][2:],Tsim,dt)
        sCro = 1e2*np.ones(2)
        Cdrb = dmft.grid_stat(np.mean,res_dict["Cdr"],Tsim,dt)
        Cdrm = dmft.grid_stat(np.mean,res_dict["Cdr"],Tsim,dt)
        sCdr = 1e2*np.ones(2)
        normC[:,0,:] = (res_dict["Cr"][:2,-1]/res_dict["Cr"][:2,0])[:,None]
        normC[:,1,:] = (res_dict["Cr"][2:,-1]/res_dict["Cr"][2:,0])[:,None]
        normC[:,2,:] = (res_dict["Cdr"][:,-1]/res_dict["Cdr"][:,0])[:,None]
        conv[:,0] = res_dict["conv"][:2]
        conv[:,1] = res_dict["conv"][2:]
        conv[:,2] = res_dict["convd"]
        dmft_res = res_dict.copy()
    else:
        res_dict = dmft.run_two_stage_ring_dmft(prms,rX,cA,CVh,"./../results",ri,Twrm,Tsav,dt)
        rvb = res_dict["rb"][:2]
        rvp = res_dict["rm"][:2]
        srv = res_dict["sr"][:2]
        rob = res_dict["rb"][2:]
        rop = res_dict["rm"][2:]
        sro = res_dict["sr"][2:]
        Crvb = dmft.grid_stat(np.mean,res_dict["Crb"][:2],Tsim,dt)
        Crvp = dmft.grid_stat(np.mean,res_dict["Crm"][:2],Tsim,dt)
        sCrv = dmft.grid_stat(np.mean,res_dict["sCr"][:2],Tsim,dt)
        Crob = dmft.grid_stat(np.mean,res_dict["Crb"][2:],Tsim,dt)
        Crop = dmft.grid_stat(np.mean,res_dict["Crm"][2:],Tsim,dt)
        sCro = dmft.grid_stat(np.mean,res_dict["sCr"][2:],Tsim,dt)
        Cdrb = dmft.grid_stat(np.mean,res_dict["Cdrb"],Tsim,dt)
        Cdrm = dmft.grid_stat(np.mean,res_dict["Cdrm"],Tsim,dt)
        sCdr = dmft.grid_stat(np.mean,res_dict["sCdr"],Tsim,dt)
        normC[:,0] = gauss(oris[None,:],res_dict["Crb"][:2,-1,None],res_dict["Crm"][:2,-1,None],
                           res_dict["sCr"][:2,-1,None]) /\
                     gauss(oris[None,:],res_dict["Crb"][:2, 0,None],res_dict["Crm"][:2, 0,None],
                           res_dict["sCr"][:2, 0,None])
        normC[:,1] = gauss(oris[None,:],res_dict["Crb"][2:,-1,None],res_dict["Crm"][2:,-1,None],
                           res_dict["sCr"][2:,-1,None]) /\
                     gauss(oris[None,:],res_dict["Crb"][2:, 0,None],res_dict["Crm"][2:, 0,None],
                           res_dict["sCr"][2:, 0,None])
        normC[:,2] = gauss(oris[None,:],res_dict["Cdrb"][:,-1,None],res_dict["Cdrm"][:,-1,None],
                           res_dict["sCdr"][:,-1,None]) /\
                     gauss(oris[None,:],res_dict["Cdrb"][:, 0,None],res_dict["Cdrm"][:, 0,None],
                           res_dict["sCdr"][:, 0,None])
        conv[:,0] = res_dict["convp"][:2]
        conv[:,1] = res_dict["convp"][2:]
        conv[:,2] = res_dict["convdp"]
        dmft_res = res_dict.copy()
        
    sWrv = np.sqrt(sW2+srv**2)
    sWCrv = np.sqrt(sW2+sCrv**2)
    sWro = np.sqrt(sW2+sro**2)
    sWCro = np.sqrt(sW2+sCro**2)
    sWCdr = np.sqrt(sW2+sCdr**2)
    
    muvb = (muW+dmft.unstruct_fact(srv)*muWb)*rvb
    muvp = muvb + dmft.struct_fact(0,sWrv,srv)*muW*(rvp-rvb)
    muvb = muvb + dmft.struct_fact(90,sWrv,srv)*muW*(rvp-rvb)
    smuv = sWrv
    muob = (muW+dmft.unstruct_fact(sro)*muWb)*rob
    muop = muob + dmft.struct_fact(0,sWro,sro)*muW*(rop-rob)
    muob = muob + dmft.struct_fact(90,sWro,sro)*muW*(rop-rob)
    smuo = sWro
    
    Sigvb = (SigW+dmft.unstruct_fact(sCrv)*SigWb)*Crvb
    Sigvp = Sigvb + dmft.struct_fact(0,sWCrv,sCrv)*SigW*(Crvp-Crvb)
    Sigvb = Sigvb + dmft.struct_fact(90,sWCrv,sCrv)*SigW*(Crvp-Crvb)
    sSigv = sWCrv
    Sigob = (SigW+dmft.unstruct_fact(sCro)*SigWb)*Crob
    Sigop = Sigob + dmft.struct_fact(0,sWCro,sCro)*SigW*(Crop-Crob)
    Sigob = Sigob + dmft.struct_fact(90,sWCro,sCro)*SigW*(Crop-Crob)
    sSigo = sWCro
    Sigdb = (SigW+dmft.unstruct_fact(sCdr)*SigWb)*Cdrb
    Sigdp = Sigdb + dmft.struct_fact(0,sWCdr,sCdr)*SigW*(Cdrm-Cdrb)
    Sigdb = Sigdb + dmft.struct_fact(90,sWCdr,sCdr)*SigW*(Cdrm-Cdrb)
    sSigd = sWCdr
    
    for i in range(2):
        μrs[i,0] = gauss(oris,rvb[i],rvp[i],srv[i])
        μrs[i,1] = gauss(oris,rob[i],rop[i],sro[i])
        μrs[i,2] = μrs[i,1] - μrs[i,0]
        Σrs[i,0] = np.fmax(gauss(oris,Crvb[i],Crvp[i],sCrv[i]) - μrs[i,0]**2,0)
        Σrs[i,1] = np.fmax(gauss(oris,Crob[i],Crop[i],sCro[i]) - μrs[i,1]**2,0)
        Σrs[i,2] = np.fmax(gauss(oris,Cdrb[i],Cdrm[i],sCdr[i]) - μrs[i,2]**2,0)
        Σrs[i,3] = 0.5*(Σrs[i,1] - Σrs[i,0] - Σrs[i,2])
        μmuEs[i,0] = gauss(oris,muvb[i,0],muvp[i,0],smuv[i,0]) + gauss(oris,muHb[i],muHp[i],smuH[i])
        μmuEs[i,1] = gauss(oris,muob[i,0],muop[i,0],smuo[i,0]) + gauss(oris,muHb[i],muHp[i],smuH[i])
        μmuEs[i,2] = μmuEs[i,1] - μmuEs[i,0]
        ΣmuEs[i,0] = gauss(oris,Sigvb[i,0],Sigvp[i,0],sSigv[i,0]) + (gauss(oris,muHb[i],muHp[i],smuH[i])*eH)**2
        ΣmuEs[i,1] = gauss(oris,Sigob[i,0],Sigop[i,0],sSigo[i,0]) + (gauss(oris,muHb[i],muHp[i],smuH[i])*eH)**2
        ΣmuEs[i,2] = gauss(oris,Sigdb[i,0],Sigdp[i,0],sSigd[i,0])
        ΣmuEs[i,3] =  0.5*(ΣmuEs[i,1] - ΣmuEs[i,0] - ΣmuEs[i,2])
        μmuIs[i,0] = gauss(oris,muvb[i,1],muvp[i,1],smuv[i,1])
        μmuIs[i,1] = gauss(oris,muob[i,1],muop[i,1],smuo[i,1])
        μmuIs[i,2] = μmuIs[i,1] - μmuIs[i,0]
        ΣmuIs[i,0] = gauss(oris,Sigvb[i,1],Sigvp[i,1],sSigv[i,1])
        ΣmuIs[i,1] = gauss(oris,Sigob[i,1],Sigop[i,1],sSigo[i,1])
        ΣmuIs[i,2] = gauss(oris,Sigdb[i,1],Sigdp[i,1],sSigd[i,1])
        ΣmuIs[i,3] =  0.5*(ΣmuIs[i,1] - ΣmuIs[i,0] - ΣmuIs[i,2])
    μmus = μmuEs + μmuIs
    Σmus = ΣmuEs + ΣmuIs

    return μrs,Σrs,μmus,Σmus,μmuEs,ΣmuEs,μmuIs,ΣmuIs,normC,conv,dmft_res

def calc_bal(μmuE,μmuI,ΣmuE,ΣmuI,N=20000,seed=0):
    rng = np.random.default_rng(seed)
    muEs = np.fmax(μmuE + np.sqrt(ΣmuE)*rng.normal(size=N),1e-12)
    muIs = np.fmin(μmuI + np.sqrt(ΣmuI)*rng.normal(size=N),-1e-12)
    return np.mean(np.abs(muEs+muIs)/muEs)

def calc_opto_bal(μmuE,μmuI,ΣmuE,ΣmuI,L,CVL,N=20000,seed=0):
    sigma_l = np.sqrt(np.log(1+CVL**2))
    mu_l = np.log(1e-3*L)-sigma_l**2/2
    rng = np.random.default_rng(seed)
    muEs = np.fmax(μmuE + np.sqrt(ΣmuE)*rng.normal(size=N) +\
        rng.lognormal(mu_l, sigma_l, N),1e-12)
    muIs = np.fmin(μmuI + np.sqrt(ΣmuI)*rng.normal(size=N),-1e-12)
    return np.mean(np.abs(muEs+muIs)/muEs)

# Simulate zero and full contrast networks with ring connectivity
print("simulating baseline fraction network")
print("")

μrs,Σrs,μmus,Σmus,μmuEs,ΣmuEs,μmuIs,ΣmuIs,normC,conv,dmft_res = predict_networks(prms,rX,cA,CVh)

start = time.process_time()

μrEs[:] = μrs[0]
μrIs[:] = μrs[1]
ΣrEs[:] = Σrs[0]
ΣrIs[:] = Σrs[1]
μhEs[:] = μmus[0]
μhIs[:] = μmus[1]
ΣhEs[:] = Σmus[0]
ΣhIs[:] = Σmus[1]
for nloc in range(Nori):
    balEs[0,nloc] = calc_bal(μmuEs[0,0,nloc],μmuIs[0,0,nloc],ΣmuEs[0,0,nloc],ΣmuIs[0,0,nloc])
    balIs[0,nloc] = calc_bal(μmuEs[1,0,nloc],μmuIs[1,0,nloc],ΣmuEs[1,0,nloc],ΣmuIs[1,0,nloc])
    balEs[1,nloc] = calc_opto_bal(μmuEs[0,1,nloc],μmuIs[0,1,nloc],ΣmuEs[0,1,nloc],ΣmuIs[0,1,nloc],prms["L"],prms["CVL"])
    balIs[1,nloc] = calc_bal(μmuEs[1,1,nloc],μmuIs[1,1,nloc],ΣmuEs[1,1,nloc],ΣmuIs[1,1,nloc])
normCEs[:] = normC[0]
normCIs[:] = normC[1]
convs[:] = conv

oris = np.arange(Nori)*180/Nori
oris[oris > 90] = 180 - oris[oris > 90]
vsm_mask = np.abs(oris) < 4.5
oris = np.abs(np.arange(Nori)*180/Nori - 90)
oris[oris > 90] = 180 - oris[oris > 90]
osm_mask = np.abs(oris) < 4.5

all_base_means = 0.8*np.mean(μrEs[0]) + 0.2*np.mean(μrIs[0])
all_base_stds = np.sqrt(0.8*np.mean(ΣrEs[0]+μrEs[0]**2) + 0.2*np.mean(ΣrIs[0]+μrIs[0]**2) - all_base_means**2)
all_opto_means = 0.8*np.mean(μrEs[1]) + 0.2*np.mean(μrIs[1])
all_opto_stds = np.sqrt(0.8*np.mean(ΣrEs[1]+μrEs[1]**2) + 0.2*np.mean(ΣrIs[1]+μrIs[1]**2) - all_opto_means**2)
all_diff_means = all_opto_means - all_base_means
all_diff_stds = np.sqrt(0.8*np.mean(ΣrEs[2]+μrEs[2]**2) + 0.2*np.mean(ΣrIs[2]+μrIs[2]**2) - all_diff_means**2)
all_norm_covs = (0.8*np.mean(ΣrEs[3]+μrEs[0]*μrEs[2]) + 0.2*np.mean(ΣrIs[3]+μrIs[0]*μrIs[2]) -\
    all_base_means*all_diff_means) / all_diff_stds**2

vsm_base_means = 0.8*np.mean(μrEs[0,vsm_mask]) + 0.2*np.mean(μrIs[0,vsm_mask])
vsm_base_stds = np.sqrt(0.8*np.mean(ΣrEs[0,vsm_mask]+μrEs[0,vsm_mask]**2) +\
    0.2*np.mean(ΣrIs[0,vsm_mask]+μrIs[0,vsm_mask]**2) - vsm_base_means**2)
vsm_opto_means = 0.8*np.mean(μrEs[1,vsm_mask]) + 0.2*np.mean(μrIs[1,vsm_mask])
vsm_opto_stds = np.sqrt(0.8*np.mean(ΣrEs[1,vsm_mask]+μrEs[1,vsm_mask]**2) +\
    0.2*np.mean(ΣrIs[1,vsm_mask]+μrIs[1,vsm_mask]**2) - vsm_opto_means**2)
vsm_diff_means = vsm_opto_means - vsm_base_means
vsm_diff_stds = np.sqrt(0.8*np.mean(ΣrEs[2,vsm_mask]+μrEs[2,vsm_mask]**2) +\
    0.2*np.mean(ΣrIs[2,vsm_mask]+μrIs[2,vsm_mask]**2) - vsm_diff_means**2)
vsm_norm_covs = (0.8*np.mean(ΣrEs[3,vsm_mask]+μrEs[0,vsm_mask]*μrEs[2,vsm_mask]) +\
    0.2*np.mean(ΣrIs[3,vsm_mask]+μrIs[0,vsm_mask]*μrIs[2,vsm_mask]) -\
    vsm_base_means*vsm_diff_means) / vsm_diff_stds**2

osm_base_means = 0.8*np.mean(μrEs[0,osm_mask]) + 0.2*np.mean(μrIs[0,osm_mask])
osm_base_stds = np.sqrt(0.8*np.mean(ΣrEs[0,osm_mask]+μrEs[0,osm_mask]**2) +\
    0.2*np.mean(ΣrIs[0,osm_mask]+μrIs[0,osm_mask]**2) - osm_base_means**2)
osm_opto_means = 0.8*np.mean(μrEs[1,osm_mask]) + 0.2*np.mean(μrIs[1,osm_mask])
osm_opto_stds = np.sqrt(0.8*np.mean(ΣrEs[1,osm_mask]+μrEs[1,osm_mask]**2) +\
    0.2*np.mean(ΣrIs[1,osm_mask]+μrIs[1,osm_mask]**2) - osm_opto_means**2)
osm_diff_means = osm_opto_means - osm_base_means
osm_diff_stds = np.sqrt(0.8*np.mean(ΣrEs[2,osm_mask]+μrEs[2,osm_mask]**2) +\
    0.2*np.mean(ΣrIs[2,osm_mask]+μrIs[2,osm_mask]**2) - osm_diff_means**2)
osm_norm_covs = (0.8*np.mean(ΣrEs[3,osm_mask]+μrEs[0,osm_mask]*μrEs[2,osm_mask]) +\
    0.2*np.mean(ΣrIs[3,osm_mask]+μrIs[0,osm_mask]*μrIs[2,osm_mask]) -\
    osm_base_means*osm_diff_means) / osm_diff_stds**2

print("Saving statistics took ",time.process_time() - start," s")
print("")

res_dict = {}

res_dict["prms"] = prms
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
res_dict["normCEs"] = normCEs
res_dict["normCIs"] = normCIs
res_dict["convs"] = convs
res_dict["all_base_means"] = all_base_means
res_dict["all_base_stds"] = all_base_stds
res_dict["all_opto_means"] = all_opto_means
res_dict["all_opto_stds"] = all_opto_stds
res_dict["all_diff_means"] = all_diff_means
res_dict["all_diff_stds"] = all_diff_stds
res_dict["all_norm_covs"] = all_norm_covs
res_dict["vsm_base_means"] = vsm_base_means
res_dict["vsm_base_stds"] = vsm_base_stds
res_dict["vsm_opto_means"] = vsm_opto_means
res_dict["vsm_opto_stds"] = vsm_opto_stds
res_dict["vsm_diff_means"] = vsm_diff_means
res_dict["vsm_diff_stds"] = vsm_diff_stds
res_dict["vsm_norm_covs"] = vsm_norm_covs
res_dict["osm_base_means"] = osm_base_means
res_dict["osm_base_stds"] = osm_base_stds
res_dict["osm_opto_means"] = osm_opto_means
res_dict["osm_opto_stds"] = osm_opto_stds
res_dict["osm_diff_means"] = osm_diff_means
res_dict["osm_diff_stds"] = osm_diff_stds
res_dict["osm_norm_covs"] = osm_norm_covs
res_dict["dmft_res"] = dmft_res

res_file = "./../results/dmft_best_fit_c_{:d}".format(c_idx)

with open(res_file+".pkl", "wb") as handle:
    pickle.dump(res_dict,handle)
