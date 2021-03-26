import snakemake
import numpy as np
import itertools


def generate_hci_targets(dir,nconfig,tol,orbitals,determinant_cutoff=0,tstep=0.02):
    hci_target, vmc_hci, dmc_hci=[], [], []
    prod=itertools.product(nconfig,tol,orbitals)
    for nconfig,tol,orbitals in prod:
        if tol==0.02: 
            orbitals='large'
        hci_target.append(f"{molecule}_{length}/hf/{basis}/hci{tol}.chk")
        dmc_hci.append(f"{dir}/dmc_hci{tol}_{determinant_cutoff}_{orbitals}_0_400_{tstep}.chk")
        if nconfig>=3200:
            orbitals='large'
        vmc_hci.append(f"{dir}/vmc_hci{tol}_{determinant_cutoff}_{orbitals}_0_{nconfig}.chk")
    return vmc_hci, dmc_hci

def generate_vmc_targets(dir,nconfig,orbitals):
    vmc_target=[]
    prod=itertools.product(nconfig,orbitals)
    for nconfig,orbitals in prod:
        if nconfig>=3200:
            orbitals='large'
        vmc_target.append(f"{dir}/vmc_mf_{orbitals}_0_{nconfig}.chk") 
    return vmc_target

def generate_dmc_targets(dir,tstep=0.02):
    dmc_target=[]
    dmc_target.append(f"{dir}/dmc_mf_orbitals_0_400_{tstep}.chk")
    dmc_target.append(f"{dir}/dmc_mf_fixed_0_400_{tstep}.chk")
    return dmc_target


molecule = ['h4'] #'h2',,'h6'
basis = ['vtz'] #'vdz',,'vqz'
bond_lengths = [1.2,1.4,1.6,1.8,2.0,3.0,4.0,4.4,4.8,5.0,5.2]
bond_lengths = [1.0,1.4,2.0,3.0,4.0,5.0]
bond_lengths = [2.0]

prod = itertools.product(molecule,basis,bond_lengths)

hf_target, uhf_target, cc_target, fci_target = [],[],[],[]
vmc_target, dmc_target = [],[]
vmc_hci, dmc_hci = [],[]
vmc_uhf = []

tstep = 0.02
determinant_cutoff = 0
tol = [0.1,0.08,0.05] #,0.02
orbitals = ['orbitals','fixed']
nconfig = [1600,3200] #400,

for molecule,basis,length in prod:

    dir_hf = f"{molecule}_{length}/hf/{basis}"
    hf_target.append(f"{dir_hf}/mf.chk")
    cc_target.append(f"{dir_hf}/cc.chk")

    vmc_target += generate_vmc_targets(dir_hf,nconfig,orbitals)
    dmc_target += generate_dmc_targets(dir_hf,tstep)

    hci = generate_hci_targets(dir_hf,nconfig,tol,orbitals,determinant_cutoff,tstep)
    vmc_hci += hci[0]
    dmc_hci += hci[1]

    if molecule=='h2' or molecule=='h4':
        fci_target.append(f"{dir_hf}/fci.chk")
        dir_uhf = f"{molecule}_{length}/uhf/{basis}"
        fci_target.append(f"{dir_uhf}/fci.chk")
        uhf_target.append(f"{dir_uhf}/mf.chk")
        vmc_uhf += generate_vmc_targets(dir_uhf,nconfig,orbitals)


targets = vmc_target + dmc_target + vmc_hci + dmc_hci # + cc_target + fci_target  #+vmc_uhf
files = " ".join(targets)
print(files)


# print(len(targets))
# snakemake.snakemake("Snakefile",cores=1,targets=targets)  #,forcetargets=True
# snakemake.snakemake("Snakefile",cores=1,targets=["h2_1.4/hf/vtz/mf.chk"])
# CCSD & HCI fail for uhf
# FCI for h4 fails for uhf
# large: h2_2.0/hf/vtz/vmc_hci0.02_0_large_0_3200.chk
