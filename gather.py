import h5py
import numpy as np
from scipy import stats
import pyqmc.obdm
import itertools
import pandas as pd
import glob 

def read_RHF(fname):
    with h5py.File(fname,'r') as f:
       # print(list(f['scf'].keys()))
        e_tot = f['scf']['e_tot'][()]
        return e_tot

def extract_from_fname(fname):
    is_opt="True"
    spl = fname.split('_')
    bond_length=float(spl[3])
    basis = spl[5].replace(".chkfile","")
    if 'orbitalopt' in spl:
        is_opt = spl[7].replace(".chkfile","")
        
    print(is_opt)   
    return {"basis":basis,
            "bond length": bond_length,"orbital_optimize":is_opt}


def create_RHF(fname):
    e_tot = read_RHF(fname)
    record = extract_from_fname(fname)
    record.update({'method':'RHF',
           "energy":e_tot,
           "error":0.0,
    })
    return record

def read_ccsd(filename):
    with h5py.File(filename) as f:
        return f['ccsd']['energy'][()],list(f['ccsd']['rdm'])

def create_ccsd(fname):
    record=extract_from_fname(fname)
    e_tot = read_ccsd(fname)[0]
    record.update({'method':'CCSD',
           "energy":e_tot,
           "error":0.0
    })
    return record


def avg(vec):
    nblock = vec.shape[0]
    avg = np.mean(vec,axis=0)
    std = np.std(vec,axis=0)
    return avg, std/np.sqrt(nblock)
    
def read_vmc(fname):
    with h5py.File(fname,'r') as f:
        en = f['energytotal'][1:]
        nblocks=len(en)

        enavg = np.mean(en)
        enerr=stats.sem(en)
        
        warmup=2
        en_vec = f['energytotal'][warmup:,...]
        en, en_err = avg(en_vec)
        rdm1, rdm1_err=avg(f['rdm1value'][warmup:,...])
        rdm1norm, rdm1norm_err = avg(f['rdm1norm'][warmup:,...])
        rdm1=pyqmc.obdm.normalize_obdm(rdm1,rdm1norm)
        rdm1_err=pyqmc.obdm.normalize_obdm(rdm1_err,rdm1norm)

        return enavg,enerr,rdm1,rdm1_err


def create_vmc(fname):
    record=extract_from_fname(fname)
    e_tot, err = read_vmc(fname)[0:2]
    record.update({'method':'VMC',
           "energy":e_tot,
           "error":err,
    })
    return record



if __name__=="__main__":
    df_rhf = pd.DataFrame([create_RHF(fname) for fname in glob.glob("RHF*")])
    df_ccsd = pd.DataFrame([create_ccsd(fname) for fname in glob.glob("ccsd*")])
    df_vmc = pd.DataFrame([create_vmc(fname) for fname in glob.glob("vmc*.chkfile")])
    df = pd.concat([df_rhf, df_ccsd, df_vmc])
    df.to_csv("energies.csv", index=False)
    