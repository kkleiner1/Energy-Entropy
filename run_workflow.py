import snakemake
import itertools

basis = ['ccpvdz','ccpvtz','ccpvqz','ccpv5z']
bond_lengths = [1,2,3,4,5,6]


prod = itertools.product(basis,bond_lengths)


molecule = 'h2'

ccsd_target=[]
vmc_target=[]
for basis,length in prod:
    print(length,"length",basis,"basis")
    ccsd_target.append(f"ccsd_{molecule}_length_{length}_basis_{basis}.chkfile")
    vmc_target.append(f"vmc_{molecule}_length_{length}_basis_{basis}_orbitalopt_True.chkfile")
#print(c)
snakemake.snakemake("Snakefile",cores=1,targets=vmc_target)