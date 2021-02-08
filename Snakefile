import functions
import concurrent
import numpy as np
qmc_threads=40
import json

#The sequence of numbers of configurations to use in optimization. 
#You should check that the energy does not change as you increase this 
#number.
nconfigs = [400,1600,3200,6400,12800,25600] 

# This is how many excited states you would like to access
nroots = 4   

rule DEFAULT_SETTINGS:
    input:
    output: "{dir}/settings.json"
    run:
        json.dump(dict(spin=0), open(output[0],'w'))

rule MEAN_FIELD:
    input: "{dir}/geom.xyz", "{dir}/settings.json"
    output: "{dir}/{functional}/{basis}/mf.chk"
    resources:
        walltime="4:00:00", partition="qmchamm"
    run:
        with open(input[0]) as f:
            xyz=f.read()
        functions.mean_field(xyz, output[0], settings=json.load(open(input[1])), basis=wildcards.basis, functional=wildcards.functional)

rule HCI:
    input: "{dir}/mf.chk"
    output: "{dir}/hci{tol}.chk"
    resources:
        walltime="4:00:00", partition="qmchamm"
    run:
        functions.run_hci(input[0],output[0], float(wildcards.tol), nroots=nroots)

rule CC:
    input: "{dir}/mf.chk"
    output: "{dir}/cc.chk"
    threads: 40
    resources:
        walltime="48:00:00", partition="qmchamm"
    run:
        functions.run_ccsd(input[0],output[0])

rule FCI:
    input: "{dir}/mf.chk"
    output: "{dir}/fci.chk"
    run:
        functions.fci(input[0], output[0], nroots=nroots)

def opt_dependency(wildcards):
    d={}
    basedir = f"{wildcards.dir}/"
    nconfig = int(wildcards.nconfig)
    ind = nconfigs.index(nconfig)
    if hasattr(wildcards,'hci_tol'):
        startingwf = f'hci{wildcards.hci_tol}'
    else:
        startingwf = "mf"

    if hasattr(wildcards, 'hci_tol'):
        basefile = basedir+f"opt_hci{wildcards.hci_tol}_{wildcards.determinant_cutoff}_{wildcards.orbitals}_"
    else: 
        basefile = basedir+f"opt_mf_{wildcards.orbitals}_"

    if ind > 0:
        d['start_from'] = basefile+f"{wildcards.statenumber}_{nconfigs[ind-1]}.chk"
    for i in range(int(wildcards.statenumber)):
        d[f'anchor_wf{i}'] = basefile+f"{i}_{nconfigs[-1]}.chk"
    return d

rule OPTIMIZE_MF:
    input: unpack(opt_dependency), mf = "{dir}/mf.chk"
    output: "{dir}/opt_mf_{orbitals}_{statenumber}_{nconfig}.chk"
    threads: qmc_threads
    resources:
        walltime="72:00:00", partition="qmchamm"
    run:
        n = int(wildcards.statenumber)
        start_from = None
        if hasattr(input, 'start_from'):
            start_from=input.start_from
        if wildcards.orbitals=='orbitals':
            slater_kws={'optimize_orbitals':True}
        elif wildcards.orbitals=='fixed':
            slater_kws={'optimize_orbitals':False}
        elif wildcards.orbitals=='large':
            slater_kws={'optimize_orbitals':True, 'optimize_zeros':False}
        else:
            raise Exception("Did not expect",wildcards.orbitals)
        if n==0:
            with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
                functions.optimize_gs(input.mf, None, output[0], start_from=start_from, nconfig = int(wildcards.nconfig), slater_kws=slater_kws, 
                    client=client, npartitions=qmc_threads)
        if n > 0:
            raise Exception("Don't support excited states just yet for mean-field wfs")


rule OPTIMIZE_HCI:
    input: unpack(opt_dependency), mf = "{dir}/mf.chk", hci="{dir}/hci{hci_tol}.chk"
    output: "{dir}/opt_hci{hci_tol}_{determinant_cutoff}_{orbitals}_{statenumber}_{nconfig}.chk"
    threads: qmc_threads
    resources:
        walltime="72:00:00", partition="qmchamm"
    run:
        n = int(wildcards.statenumber)
        start_from = None
        if hasattr(input, 'start_from'):
            start_from=input.start_from
        if wildcards.orbitals=='orbitals':
            slater_kws={'optimize_orbitals':True}
        elif wildcards.orbitals=='fixed':
            slater_kws={'optimize_orbitals':False}
        elif wildcards.orbitals=='large':
            slater_kws={'optimize_orbitals':True, 'optimize_zeros':False}
        else:
            raise Exception("Did not expect",wildcards.orbitals)

        slater_kws['tol'] = float(wildcards.determinant_cutoff)

        if n==0:
            with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
                functions.optimize_gs(input.mf, input.hci, output[0], start_from=start_from, 
                                  nconfig = int(wildcards.nconfig), slater_kws=slater_kws, client=client, npartitions=qmc_threads)
        if n > 0:
            anchor_wfs = [input[f'anchor_wf{i}'] for i in range(n)]
            functions.orthogonal_opt(input.mf, input.hci, anchor_wfs, output[0], 
                                     slater_kws=slater_kws, nconfig=int(wildcards.nconfig) )

rule VMC:
    input: mf = "{dir}/mf.chk", opt = "{dir}/opt_{variables}.chk"
    output: "{dir}/vmc_{variables}.chk"
    threads: qmc_threads
    resources:
        walltime="24:00:00", partition="qmchamm"
    run:
        multideterminant = None
        startingwf = input.opt.split('/')[-1].split('_')[1]
        if 'hci' in startingwf:
            multideterminant = wildcards.dir+"/"+startingwf+".chk"

        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            functions.evaluate_vmc(input.mf, multideterminant, input.opt, output[0], nconfig=8000, nblocks=60, client=client, npartitions=qmc_threads)


rule DMC:
    input: mf = "{dir}/mf.chk", opt = "{dir}/opt_{variables}.chk"
    output: "{dir}/dmc_{variables}_{tstep}.chk"
    threads: qmc_threads
    resources:
        walltime="24:00:00", partition="qmchamm"
    run:
        multideterminant = None
        startingwf = input.opt.split('/')[-1].split('_')[1]
        if 'hci' in startingwf:
            multideterminant = wildcards.dir+"/"+startingwf+".chk"
        tstep = float(wildcards.tstep)
        nsteps = int(30/tstep)
        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            functions.evaluate_dmc(input.mf, multideterminant, input.opt, output[0], tstep=tstep, nsteps=nsteps, nconfig=8000, client=client, npartitions=qmc_threads)
