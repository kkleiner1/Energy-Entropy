import snakemake
import numpy as np
import itertools

molecule = 'h2'
basis = ['vtz','vqz']
bond_lengths = [1.2,1.4,1.6,1.8,2.0,3.0,4.0,4.4,4.8,5.0,5.2]

prod = itertools.product(basis,bond_lengths)

cc_target,vmc_target,dmc_target = [],[],[]
fci_target,vmc_hci,dmc_hci = [],[],[]

for basis,length in prod:
    #hf_target.append(f"{molecule}_{length}/hf/{basis}/mf.chk")
    #uhf_target.append(f"{molecule}_{length}/uhf/{basis}/mf.chk")
    cc_target.append(f"{molecule}_{length}/hf/{basis}/mf.chk")
    fci_target.append(f"{molecule}_{length}/uhf/{basis}/fci.chk")

    determinant_cutoff=0
    tstep=0.02
    for tol in [0.1]: #,0.08,0.05,0.02
        #hci_target.append(f"{molecule}_{length}/hf/{basis}/hci{tol}.chk")
        vmc_hci.append(f"{molecule}_{length}/hf/{basis}/vmc_hci{tol}_{determinant_cutoff}_fixed_0_400.chk")
        vmc_hci.append(f"{molecule}_{length}/hf/{basis}/vmc_hci{tol}_{determinant_cutoff}_orbitals_0_400.chk")
        dmc_hci.append(f"{molecule}_{length}/hf/{basis}/dmc_hci{tol}_{determinant_cutoff}_fixed_0_400_{tstep}.chk")
        dmc_hci.append(f"{molecule}_{length}/hf/{basis}/dmc_hci{tol}_{determinant_cutoff}_orbitals_0_400_{tstep}.chk")

    dmc_target.append(f"{molecule}_{length}/hf/{basis}/dmc_mf_fixed_0_400_{tstep}.chk")
    dmc_target.append(f"{molecule}_{length}/hf/{basis}/dmc_mf_orbitals_0_400_{tstep}.chk")
    vmc_target.append(f"{molecule}_{length}/hf/{basis}/vmc_mf_orbitals_0_400.chk")  #actually nconfig=8000 in rules
    vmc_target.append(f"{molecule}_{length}/hf/{basis}/vmc_mf_fixed_0_400.chk")
    #vmc_target.append(f"{molecule}_{length}/uhf/{basis}/vmc_mf_orbitals_0_400.chk")
    #vmc_target.append(f"{molecule}_{length}/uhf/{basis}/vmc_mf_fixed_0_400.chk")

targets = cc_target+vmc_target+dmc_target+fci_target+vmc_hci+dmc_hci
files = " ".join(targets)
print(targets)
snakemake.snakemake("Snakefile",cores=1,targets=targets,forcetargets=True)
