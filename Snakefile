
import functions
import numpy as np


#Sequence of configurations to optimize over
nconfigs = [400,1600,3200]

rule HARTREE_FOCK:
    input:  "{dir}/geom.xyz"
    output: "{dir}/hf/{basis}/mf.chk"
    run:
        functions.hartree_fock(open(input[0],'r').read(), output[0], basis=wildcards.basis)

rule UNRESTRICTED_HARTREE_FOCK:
    input:  "{dir}/geom.xyz"
    output: "{dir}/uhf/{basis}/mf.chk"
    run:
        functions.unrestricted_hartree_fock(open(input[0],'r').read(), output[0], basis=wildcards.basis)


rule HCI:
    input: "{dir}/mf.chk"
    output: "{dir}/hci{tol}.chk"
    run:
        functions.run_hci(input[0],output[0], float(wildcards.tol))

rule CC:
    input: "{dir}/mf.chk"
    output: "{dir}/cc.chk"
    run:
        functions.run_ccsd(input[0],output[0])


rule FCI:
    input: "{dir}/mf.chk"
    output: "{dir}/fci.chk"
    run:
        functions.fci(input[0], output[0])

def opt_dependency(wildcards):
    basedir = f"{wildcards.dir}/"
    d={'hf':basedir+"mf.chk"}
    if 'hci' in wildcards.startingwf: 
        d['multideterminant']=basedir+wildcards.startingwf+".chk"
    nconfig = int(wildcards.nconfig)
    ind = nconfigs.index(nconfig)
    if ind > 0:
        d['start_from'] = basedir+f"opt_{wildcards.startingwf}_{wildcards.statenumber}_{nconfigs[ind-1]}.chk"
    for i in range(int(wildcards.statenumber)):
        d[f'anchor_wf{i}'] = basedir + f"opt_{wildcards.startingwf}_{i}_{nconfigs[-1]}.chk"
    return d

rule OPTIMIZE:
    input: unpack(opt_dependency)
    output: "{dir}/opt_{startingwf}_{statenumber}_{nconfig}.chk"
    run:
        n = int(wildcards.statenumber)
        start_from = None
        multideterminant = None
        if hasattr(input, 'start_from'):
            start_from=input.start_from
        if hasattr(input, 'multideterminant'):
            multideterminant = input.multideterminant 
        if n==0:
            functions.optimize_gs(input.hf, multideterminant, output[0], start_from=start_from, nconfig = int(wildcards.nconfig), slater_kws={'optimize_orbitals':True})
        if n > 0:
            anchor_wfs = [input[f"anchor_wf{i}"] for i in range(n)] 
            weights=[0.0, 0.0, 0.0, 0.0]
            weights[n] = 1.0
            forcing = 2.0*np.ones(n)
            functions.orthogonal_opt(input.hf, multideterminant, anchor_wfs, output[0], weights, nconfig=int(wildcards.nconfig), forcing=forcing, tstep=0.5)



rule VMC:
    input:hf="{dir}/{startingwf}.chk", opt = "{dir}/opt_{startingwf}_{statenumber}_{fname}.chk"
    output: "{dir}/vmc_{startingwf}_{statenumber}_{fname}.chk"
    run:
        multideterminant = None
        mf = input.hf+".chk"
        if 'hci' in wildcards.startingwf:
            multideterminant = input.hf
            mf = wildcards.dir +"/mf.chk"

        functions.evaluate_vmc(mf, multideterminant, input.opt, output[0], nconfig=8000, nblocks=60)
