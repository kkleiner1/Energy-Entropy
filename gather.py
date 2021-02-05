import h5py
import numpy as np
from scipy import stats
import pyqmc.obdm
import itertools
import pandas as pd
import glob 

def calculate_entropy(rdm1):
    a=np.linalg.eigvals(rdm1)
    x=np.real(a)
    lam = x[x>0]
    return -np.sum(lam*np.log(lam))

def extract_from_pathname(pathname):
    spl = pathname.split('/')
    bond_length=float(spl[0].split('_')[-1])
    basis = spl[2]
    return {"basis":basis,
            "bond length": bond_length}

def read_fci(pathname)
    with h5py.File(pathname,'r') as f:
        e_tot = f['ci']['energy'][()][0]
        rdm1 = np.array(f['ci']['rdm'])
        return e_tot,rdm1

def read_hci(pathname):
    with h5py.File(pathname,'r') as f:
        e_tot = f['ci']['energy'][()][0]
        rdm1 = np.array(f['ci']['rdm'])
        return e_tot,rdm1

def create_hci(pathname):
    e_tot,rdm1 = read_hci(pathname)
    record = extract_from_pathname(pathname)
    ent = calculate_entropy(rdm1)
    record.update({'method':'hci',
           "energy":e_tot,
           "error":0.0, "entropy":ent
    })
    return record

def read_RHF(pathname):
    with h5py.File(pathname,'r') as f:
        e_tot = f['scf']['e_tot'][()]
        return e_tot

def create_RHF(pathname):
    e_tot = read_RHF(pathname)
    record = extract_from_pathname(pathname)
    record.update({'method':'ROHF',
           "energy":e_tot,
           "error":0.0,'entropy':0
    })
    return record

def read_ccsd(filename):
    with h5py.File(filename) as f:
        return f['ccsd']['energy'][()],np.array((f['ccsd']['rdm']))/2

def create_ccsd(pathname):
    record=extract_from_pathname(pathname)
    e_tot,rdm1 = read_ccsd(pathname)
    ent = calculate_entropy(rdm1)
    record.update({'method':'CCSD',
           "energy":e_tot,
           "error":0.0, "entropy":ent
    })
    return record

def avg(vec):
    nblock = vec.shape[0]
    avg = np.mean(vec,axis=0)
    std = np.std(vec,axis=0)
    return avg, std/np.sqrt(nblock)
    
def read_vmc(pathname):
    with h5py.File(pathname,'r') as f:
        en = f['energytotal'][1:]
        nblocks=len(en)
        enavg = np.mean(en)
        enerr=stats.sem(en)
        warmup=2
        en_vec = f['energytotal'][warmup:,...]
        en, en_err = avg(en_vec)
        rdm1, rdm1_err=avg(f['rdm1_upvalue'][warmup:,...])
        rdm1norm, rdm1norm_err = avg(f['rdm1_upnorm'][warmup:,...])
        rdm1=pyqmc.obdm.normalize_obdm(rdm1,rdm1norm)
        rdm1_err=pyqmc.obdm.normalize_obdm(rdm1_err,rdm1norm)
        return enavg,enerr,rdm1,rdm1_err

def create_vmc(pathname):
    record=extract_from_fname(pathname)
    e_tot, err ,rdm1,rdm1_err = read_vmc(pathname)
    ent = calculate_entropy(rdm1)
    fname = pathname.split('/')[-1]
    fwords = fname.split('_')
    method = fwords[0]+'_'+fwords[1]
    record.update({'method':method,
           "energy":e_tot,
           "error":err, "entropy":ent
    })
    return record

if __name__=="__main__":
    df_rhf = pd.DataFrame([create_RHF(pathname) for pathname in glob.glob('h2*/hf/*/mf.chk')])
    df_ccsd = pd.DataFrame([create_ccsd(pathname) for pathname in glob.glob('h2*/hf/*/cc.chk')])
    df_vmc = pd.DataFrame([create_vmc(pathname) for pathname in glob.glob("h2*/hf/*/vmc_*.chk")])
    df_hci = pd.DataFrame([create_hci(pathname) for pathname in glob.glob("h2*/hf/*/hci*.chk")])
    df = pd.concat([df_rhf, df_ccsd, df_vmc,df_hci])
    df.to_csv("energies.csv", index=False)
    
