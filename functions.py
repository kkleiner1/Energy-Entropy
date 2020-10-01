import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import pyscf
import pyqmc.recipes
import h5py
import matplotlib.pyplot as plt
import concurrent.futures 

def mean_field(chkfile,length_factor,basis):
    print(chkfile)
    z=1
    mol = pyscf.gto.M(atom = "H 0 0 0; H 0 0 {0}".format(z*length_factor), basis=basis, unit='bohr')
#ecp='ccecp'
#'ccpvdz'
    mf = pyscf.scf.RHF(mol)
    mf.chkfile = chkfile
    mf.kernel()

def opt(infile,outfile):
    pyqmc.recipes.OPTIMIZE(infile,outfile,nconfig=200)

def opt_orbitals(infile,outfile,startfile):
    pyqmc.recipes.OPTIMIZE(infile,outfile,start_from = startfile,nconfig=200, jastrow_kws = {},slater_kws={"optimize_orbitals":True})


def vmc(RHF_chkfile,opt_chkfile,output_chkfile):
    npartitions =2

    with concurrent.futures.ProcessPoolExecutor(max_workers=npartitions) as client:
        pyqmc.recipes.VMC(RHF_chkfile,output_chkfile,
         start_from=opt_chkfile, client=client,
         npartitions=npartitions,accumulators=dict(rdm1=True),vmc_kws=dict(nblocks=100))

def run_ccsd(hf_chkfile, chkfile):
    mol, mf = pyqmc.recover_pyscf(hf_chkfile)
    mycc = pyscf.cc.CCSD(mf).run(verbose=0)
    dm1 = mycc.make_rdm1()
    from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
    from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
    eris = mycc.ao2mo()
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2)
    dm1_t = ccsd_t_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2, eris=eris)
    pyscf.lib.chkfile.save(chkfile,'ccsd',
        {
        'energy':mycc.e_tot,
        'rdm':dm1
        })
    pyscf.lib.chkfile.save(chkfile,'ccsdt',
        {
        'energy':mycc.ccsd_t(),
        'rdm':dm1_t
        })

#run_ccsd('h2_length_1_RHF.chkfile','h2_length_1_ccsd.chkfile')




# molecule = 'h2'
# for length in  [1]:
# 	mean_field("{0}_length_{1}_UHF.hdf5".format(molecule,length),length)

