import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pyscf
import pyqmc.recipes
import h5py
import pyqmc.multislater
import pyqmc.tbdm
import pyscf.hci


def hartree_fock(xyz, chkfile, spin=0, basis='vtz'):
    mol = pyscf.gto.M(atom = xyz, basis=f'ccecpccp{basis}', ecp='ccecp', unit='bohr', charge=0, spin=spin)
    mf = pyscf.scf.ROHF(mol)
    mf.chkfile=chkfile
    mf.kernel()


def run_hci(hf_chkfile, chkfile, select_cutoff=0.1):
    mol, mf = pyqmc.recover_pyscf(hf_chkfile, cancel_outputs=False)
    cisolver = pyscf.hci.SCI(mol)
    cisolver.select_cutoff=select_cutoff
    nmo = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    h2 = pyscf.ao2mo.full(mol, mf.mo_coeff)
    e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=0)
    cisolver.ci= civec[0]
    rdm1,rdm2 = cisolver.make_rdm12s(civec[0], nmo, nelec)
    pyscf.lib.chkfile.save(chkfile,'ci',
        {'ci':cisolver.ci,
        'nmo':nmo,
        'nelec':nelec,
        '_strs':cisolver._strs,
        'select_cutoff':select_cutoff,
        'energy':e+mol.energy_nuc(),
        'rdm':np.array(rdm1)
        })


def fci(hf_chkfile, fci_chkfile):
    mol, mf = pyqmc.recover_pyscf(hf_chkfile, cancel_outputs=False)
    cisolver = pyscf.fci.FCI(mf)
    cisolver.nroots = 4
    cisolver.kernel()
    print(dir(cisolver))
    with h5py.File(fci_chkfile, "w") as f:
        f["e_tot"] = cisolver.e_tot


def recover_hci(hf_chkfile, ci_chkfile):
    mol, mf = pyqmc.recover_pyscf(hf_chkfile)
    cisolver = pyscf.hci.SCI(mol)
    cisolver.__dict__.update(pyscf.lib.chkfile.load(ci_chkfile,'ci'))
    return mol, mf, cisolver

def generate_wf(hf_chkfile, ci_chkfile, slater_kws=None):
    if ci_chkfile is not None:
        mol, mf, cisolver = recover_hci(hf_chkfile, ci_chkfile)
        wf, to_opt = pyqmc.generate_wf(mol, mf, mc=cisolver, slater_kws=slater_kws)
    else: 
        mol, mf = pyqmc.recover_pyscf(hf_chkfile)
        wf, to_opt = pyqmc.generate_wf(mol, mf, slater_kws=slater_kws)
    return mol, mf, wf, to_opt

def optimize_gs(hf_chkfile, ci_chkfile, opt_chkfile, start_from=None, slater_kws=None, nconfig=1000, **kwargs):
    mol, mf, wf, to_opt = generate_wf(hf_chkfile, ci_chkfile, slater_kws)
    if start_from is not None:
        print("reading from", start_from)
        pyqmc.read_wf(wf, start_from)

    configs = pyqmc.initial_guess(mol, nconfig)
    acc = pyqmc.gradient_generator(mol, wf, to_opt)
    pyqmc.line_minimization(wf, configs, acc, verbose=True, hdf_file = opt_chkfile, **kwargs)

def transform_ci(ci, coeffs):
    return np.sum([ci[i] * coeffs[i] for i in range(len(coeffs))], axis=0)


def generate_wfs(hf_chkfile, cas_chkfile, Starget, anchor_wfs):
    mol, mf, cas = recover_pyscf_all(hf_chkfile, cas_chkfile)
    cas.ci = transform_ci(cas.ci, Starget)

    wf_anchor = []
    for anchor in anchor_wfs:
        wf, to_opt = pyqmc.generate_wf(
            mol,
            mf,
            mc=cas,
            slater_kws={"optimize_orbitals": True},
            jastrow_kws={"ion_cusp": True, "na": 1},
        )
        pyqmc.read_wf(wf, anchor)
        wf_anchor.append(wf)

    wf_es, to_opt = pyqmc.generate_wf(
        mol,
        mf,
        mc=cas,
        slater_kws={"optimize_orbitals": True},
        jastrow_kws={"ion_cusp": True, "na": 1},
    )

    for k in wf_es.parameters.keys():
        if "wf2" in k:
            wf_es.parameters[k] = wf_anchor[0].parameters[k].copy()
            print(wf_es.parameters[k])
    return mol, wf_anchor, wf_es, to_opt


def orthogonal_opt(
    hf_chkfile, cas_chkfile, anchor_wfs, ortho_chkfile, Starget, nconfig=500, start_from=None, **kws
):
    mol, wf_anchor, wf_es, to_opt = generate_wfs(
        hf_chkfile, cas_chkfile, Starget, anchor_wfs
    )
    for k in ["wf1det_coeff", "wf1mo_coeff_alpha", "wf1mo_coeff_beta"]:
        to_opt[k] = np.ones_like(to_opt[k], dtype="bool")
    if start_from is not None:
        pyqmc.read_wf(wf_es, start_from)
    configs = pyqmc.initial_guess(mol, nconfig)
    acc = pyqmc.gradient_generator(mol, wf_es, to_opt)
    pyqmc.optimize_orthogonal(
        wf_anchor+[wf_es],
        configs,
        acc,
        Starget[0:len(anchor_wfs)],
        hdf_file=ortho_chkfile,
        **kws,
    )


def generate_accumulators(mol, mf):
    mo_coeff2rdm = np.array([mf.mo_coeff, mf.mo_coeff])
    return {
        'energy':pyqmc.EnergyAccumulator(mol),
        'rdm1_up':pyqmc.obdm.OBDMAccumulator(mol, orb_coeff=mf.mo_coeff, spin=0),
        'rdm1_down': pyqmc.obdm.OBDMAccumulator(mol, orb_coeff=mf.mo_coeff, spin=1),
        #'rdm2_updown': pyqmc.tbdm.TBDMAccumulator(mol, orb_coeff=mo_coeff2rdm, spin=(0,1)),
        #'rdm2_upup': pyqmc.tbdm.TBDMAccumulator(mol, orb_coeff=mo_coeff2rdm, spin=(0,0))
    }

def evaluate_vmc(hf_chkfile, ci_chkfile, opt_chkfile, vmc_chkfile, slater_kws=None, nconfig=1000, **kwargs):
    mol, mf, wf, to_opt = generate_wf(hf_chkfile, ci_chkfile, slater_kws)
    configs = pyqmc.initial_guess(mol, nconfig)
    pyqmc.read_wf(wf, opt_chkfile)
    acc=generate_accumulators(mol, mf)
    pyqmc.vmc(wf, configs, accumulators=acc, verbose=True, hdf_file = vmc_chkfile, **kwargs)


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

