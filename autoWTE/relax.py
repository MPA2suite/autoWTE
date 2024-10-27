import warnings, os

import numpy as np

from ase.io import read, write
from ase.io.vasp import read_vasp
from ase import Atoms
from ase.calculators.lammpslib import LAMMPSlib
from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter, ExpCellFilter, StrainFilter,FrechetCellFilter
from ase.optimize import BFGS, FIRE, MDMin, GPMin
from ase.spacegroup import get_spacegroup

from autoWTE.utils import *


from pathlib import Path



convert_ase_to_bar=1.602176634e-19/1e-30/1e5
Optimizer_default=BFGS


def mutlistage_relax(
        atoms,
        calculator = None,
        fmax = 1e-4,
        fmax_step2 = None,
        fmax_init = 1e-3,
        allow_tilt = False,
        Optimizer = Optimizer_default,
        Filter  = FrechetCellFilter,
        filter_kwargs = None,
        force_symmetry=True,
        log : str | Path | bool = True, # NOT WORKING FOR FILES FOR SYMMETRIES
        position_optim_kwargs : dict | None = None,
        cell_params_optim_kwargs : dict | None = None,
        optim_kwargs : dict | None = None,
        joint_relax : bool = False,
        steps : int = 300,
        symprec : float = 1e-5
        ):
    

    if fmax_step2 is None:
        fmax_step2=fmax

    if calculator is not None:
        atoms.calc=calculator

    if filter_kwargs is None:
        _filter_kwargs = {}
    else:
        _filter_kwargs = filter_kwargs

    if position_optim_kwargs is None and optim_kwargs is None:
        _position_optim_kwargs = {}
    else:
        if optim_kwargs is None:
            _position_optim_kwargs = position_optim_kwargs
        else:
            _cell_params_optim_kwargs = optim_kwargs
    
    if cell_params_optim_kwargs is None and optim_kwargs is None:
        _cell_params_optim_kwargs = {}
    else:
        if optim_kwargs is None:
            _cell_params_optim_kwargs = cell_params_optim_kwargs
        else:
            _cell_params_optim_kwargs = optim_kwargs

    if optim_kwargs is None:
        _optim_kwargs = {}
    else:
        _optim_kwargs = optim_kwargs

    
    if log == False:
        ase_logfile=None
    elif log==True:
        ase_logfile='-'
    else:
        ase_logfile = log
    
    if "name" in atoms.info.keys():
        mat_name = atoms.info["name"]
    else:
        mat_name = f'{atoms.get_chemical_formula(mode="metal",empirical=True)}-{get_spacegroup(atoms,symprec=symprec).no}'


    NO_TILT_MASK=[True,True,True,False,False,False]

    tilt_mask=None
    if not allow_tilt:
        tilt_mask=NO_TILT_MASK

    input_cellpar=atoms.cell.cellpar().copy()

    log_message(f"Relaxing {mat_name}\n",output=log)
    log_message(f"Initial Energy {atoms.get_potential_energy()} ev",output=log)
    log_message(f"Initial Stress {atoms.get_stress()*convert_ase_to_bar} bar",output=log)
    log_message("Initial symmetry:",output=log)
    sym_before_5=log_symmetry(atoms, symprec, output=log)

    atoms.set_constraint(FixSymmetry(atoms))

    if joint_relax:
        cell_filter=StrainFilter(atoms,mask=tilt_mask,**_filter_kwargs)
        total_filter=Filter(atoms,mask=NO_TILT_MASK,**_filter_kwargs)

        dyn_cell = Optimizer(cell_filter,**_cell_params_optim_kwargs,logfile=ase_logfile)
        dyn_atoms_only = Optimizer(atoms,**_position_optim_kwargs,logfile=ase_logfile)
        dyn_total=Optimizer(total_filter,**_optim_kwargs,logfile=ase_logfile)

        # Run a optimisation for atomic positions 
        # with every step rescaling the cell to minimise stress
        dyn_total.run(fmax=fmax_init,steps=100)
        
        
        for _ in dyn_atoms_only.irun(fmax=fmax,steps=500):
            dyn_cell.run(fmax=fmax,steps=50)
        dyn_atoms_only.run(fmax=fmax_step2,steps=500)

    else:
        total_filter=Filter(atoms,mask=tilt_mask,**_filter_kwargs)
        dyn_total=Optimizer(total_filter,**_optim_kwargs,logfile=ase_logfile)
        dyn_total.run(fmax=fmax,steps=steps)



    log_message(f"After keeping symmetry stage 1 relax, energy {atoms.get_potential_energy()} ev",output=log)
    log_message(f"After keeping symmetry stage 1 relax, stress {atoms.get_stress()*convert_ase_to_bar} bar",output=log)

    cell_diff = (atoms.cell.cellpar() / input_cellpar - 1.0) * 100
    log_message("Optimized Cell         :", atoms.cell.cellpar(),output=log)
    log_message("Optimized Cell diff (%):", cell_diff,output=log)

    # We print out the initial symmetry groups
    log_message("After keeping symmetry stage 1 relax, symmetry:",output=log)
    sym_middle_5=log_symmetry(atoms, symprec, output=log)

    if sym_middle_5['number']!=sym_before_5['number']:
        warnings.warn(f"Symmetry is not kept during FixSymmetry relaxation of material {mat_name} in folder {os.getcwd()}")
        log_message(f"Symmetry is not kept during FixSymmetry relaxation of material {mat_name} in folder {os.getcwd()}",output=log)

    # delete constrainsts and run a optimisation for atomic positions 
    # with every step rescaling the cell to minimise stress
    atoms_symmetry=atoms.copy()

    atoms.constraints = None

    

    #log_message("Right after deleting symmetry VC/FC relax Energy", atoms.get_potential_energy()," ev",output=log)
    #log_message("Right after deleting symmetry VC/FC relax Stress",atoms.get_stress()*convert_ase_to_bar," bar",output=log)

    # We print out the initial symmetry groups
    #log_message("Right after deleting symmetry VC/FC relax symmetry:",output=log)
    #log_symmetry(atoms, symprec, output=log)

    if joint_relax : 
        dyn_atoms_only.run(fmax=fmax_step2,steps=200)
        for _ in dyn_atoms_only.irun(fmax=fmax,steps=200):
            dyn_cell.run(fmax=fmax,steps=25)
        dyn_atoms_only.run(fmax=fmax_step2,steps=200)
    else:
        dyn_total.run(fmax=fmax,steps=steps)

    log_message("Final Energy", atoms.get_potential_energy()," ev",output=log)
    log_message("Final Stress",atoms.get_stress()*convert_ase_to_bar," bar",output=log)

    #log_message("Final Symmetry:",output=log)
    #log_symmetry(atoms, symprec, output=log)
    log_message("Final Symmetry:",output=log)
    sym_after_5=log_symmetry(atoms, symprec, output=log)
    

    # compare symmetries
    redirected_to_symm = False
    if sym_middle_5['number']!=sym_after_5['number'] and force_symmetry:
        redirected_to_symm = True
        atoms=atoms_symmetry
        warnings.warn(f"Symmetry is not kept after deleting FixSymmetry constraint, redirecting to structure with symmetry of material {mat_name}, in folder {os.getcwd()}")
        log_message(f"Symmetry is not kept after deleting FixSymmetry constraint, redirecting to structure with symmetry of material {mat_name}, in folder {os.getcwd()}",output=log)


    cell_diff = (atoms.cell.cellpar() / input_cellpar - 1.0) * 100
    log_message(f"Optimized Cell         : {atoms.cell.cellpar()}",output=log)
    log_message(f"Optimized Cell diff (%): {cell_diff}\n",output=log)

    return atoms

