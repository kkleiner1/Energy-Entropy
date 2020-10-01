
import functions

rule:
    input:
    output: 'RHF_{molecule}_length_{length}_basis_{basis}.chkfile'
    run: 
        functions.mean_field(output[0],wildcards.length,wildcards.basis)

rule: 
    input: "RHF_{molecule}_length_{length}_basis_{basis}.chkfile"
    output: "optimized_sj_{molecule}_length_{length}_basis_{basis}_orbitalopt_False.chkfile"
    run:
        functions.opt(input[0],output[0])

rule:
    input: RHF = "RHF_{molecule}_length_{length}_basis_{basis}.chkfile", sj = "optimized_sj_{molecule}_length_{length}_basis_{basis}_orbitalopt_False.chkfile"
    output: "optimized_sj_{molecule}_length_{length}_basis_{basis}_orbitalopt_True.chkfile"
    run:
        functions.opt_orbitals(input[0],output[0],input[1])
rule:
    input: RHF = "RHF_{molecule}_length_{length}_basis_{basis}.chkfile" , opt = "optimized_sj_{molecule}_length_{length}_basis_{basis}_orbitalopt_{is_opt}.chkfile"
    output: "vmc_{molecule}_length_{length}_basis_{basis}_orbitalopt_{is_opt}.chkfile"
    run:
        functions.vmc(input[0],input[1],output[0])
rule:
    input: "RHF_{molecule}_length_{length}_basis_{basis}.chkfile"
    output: "ccsd_{molecule}_length_{length}_basis_{basis}.chkfile"
    run:
        functions.run_ccsd(input[0],output[0])
    