import numpy as np
from ase.io import read, write
from ase.io.vasp import read_vasp
from ase import Atoms
from ase.calculators.lammpslib import LAMMPSlib
from ase.constraints import FixSymmetry
from ase.filters import UnitCellFilter, ExpCellFilter, StrainFilter,FrechetCellFilter
from ase.optimize import BFGS, FIRE, MDMin, GPMin
from ase.spacegroup.symmetrize import check_symmetry
from autoWTE.load import *
import warnings, os

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
        Filter  = ExpCellFilter,
        filter_kwargs = None,
        force_symmetry=True,
        log : str | Path | bool = True, # NOT WORKING FOR FILES FOR SYMMETRIES
        position_optim_kwargs : dict | None = None,
        cell_params_optim_kwargs : dict | None = None,
        ):
    

    if fmax_step2 is None:
        fmax_step2=fmax

    if calculator is not None:
        atoms.calc=calculator

    if filter_kwargs is None:
        _filter_kwargs = {}
    else:
        _filter_kwargs = filter_kwargs

    if position_optim_kwargs is None:
        _position_optim_kwargs = {}
    else:
        _position_optim_kwargs = position_optim_kwargs
    
    if cell_params_optim_kwargs is None:
        _cell_params_optim_kwargs = {}
    else:
        _cell_params_optim_kwargs = cell_params_optim_kwargs

    if log == False:
        ase_logfile=None
    elif log==True:
        ase_logfile='-'
    else:
        ase_logfile = log
    
    NO_TILT_MASK=[True,True,True,False,False,False]

    tilt_mask=None
    if not allow_tilt:
        tilt_mask=NO_TILT_MASK

    input_cellpar=atoms.cell.cellpar().copy()
    
    log_message(f"Initial Energy {atoms.get_potential_energy()} ev",output=log)
    log_message(f"Initial Stress {atoms.get_stress()*convert_ase_to_bar} bar",output=log)

    log_message("Initial symmetry at precision 1e-5",output=log)
    sym_before_5=check_symmetry(atoms, 1.0e-5, verbose=bool(log))

    atoms.set_constraint(FixSymmetry(atoms))

    cell_filter=StrainFilter(atoms,mask=tilt_mask,**_filter_kwargs)
    exp_filter=Filter(atoms,mask=NO_TILT_MASK,**_filter_kwargs)

    dyn_cell = Optimizer(cell_filter,**_cell_params_optim_kwargs,logfile=ase_logfile)
    dyn_atoms_only = Optimizer(atoms,**_position_optim_kwargs,logfile=ase_logfile)
    dyn_total=Optimizer(exp_filter,**_position_optim_kwargs,logfile=ase_logfile)



    # Run a optimisation for atomic positions 
    # with every step rescaling the cell to minimise stress
    dyn_total.run(fmax=fmax_init,steps=100)
    
    
    for _ in dyn_atoms_only.irun(fmax=fmax,steps=500):
        dyn_cell.run(fmax=fmax,steps=50)
    dyn_atoms_only.run(fmax=fmax_step2,steps=500)

    log_message(f"After keeping symmetry VC/FC relax Energy {atoms.get_potential_energy()} ev",output=log)
    log_message(f"After keeping symmetry VC/FC relax Stress {atoms.get_stress()*convert_ase_to_bar} bar",output=log)

    cell_diff = (atoms.cell.cellpar() / input_cellpar - 1.0) * 100
    log_message("Optimized Cell         :", atoms.cell.cellpar())
    log_message("Optimized Cell diff (%):", cell_diff)

    # We print out the initial symmetry groups at two different precision levels
    if log:
        log_message("After keeping symmetry VC/FC relax symmetry at precision 1e-5",output=log)
    sym_middle_5=check_symmetry(atoms, 1.0e-5, verbose=bool(log))

    if sym_middle_5['number']!=sym_before_5['number']:
        warnings.warn(f"SYMMETRY IS NOT KEPT DURING FxSymmetry RELAXTION in folder {os.getcwd()}")

    # delete constrainsts and run a optimisation for atomic positions 
    # with every step rescaling the cell to minimise stress
    atoms_symmetry=atoms.copy()

    atoms.constraints = None

    dyn_atoms_only.run(fmax=fmax_step2,steps=200)

    log_message("Right after deleting symmetry VC/FC relax Energy", atoms.get_potential_energy()," ev",output=log)
    log_message("Right after deleting symmetry VC/FC relax Stress",atoms.get_stress()*convert_ase_to_bar," bar",output=log)

    # We print out the initial symmetry groups at two different precision levels
    log_message("Right after deleting symmetry VC/FC relax symmetry at precision 1e-5",output=log)
    check_symmetry(atoms, 1.0e-5, verbose=bool(log))


    for _ in dyn_atoms_only.irun(fmax=fmax,steps=200):
        dyn_cell.run(fmax=fmax,steps=25)
    dyn_atoms_only.run(fmax=fmax_step2,steps=200)

    log_message("Final Energy", atoms.get_potential_energy()," ev",output=log)
    log_message("Final Stress",atoms.get_stress()*convert_ase_to_bar," bar",output=log)

    log_message("Final symmetry at precision 1e-4",output=log)
    check_symmetry(atoms, 1.0e-4, verbose=bool(log))
    log_message("Final symmetry at precision 1e-5",output=log)
    sym_after_5=check_symmetry(atoms, 1.0e-5, verbose=bool(log))
    

    # compare symmetries
    
    if sym_middle_5['number']!=sym_after_5['number'] and force_symmetry:
        atoms=atoms_symmetry
        warnings.warn(f"SYMMETRY IS NOT KEPT AFTER DELETING CONSTRAINT, redirecting to structure with symmetry, in folder {os.getcwd()}")

    cell_diff = (atoms.cell.cellpar() / input_cellpar - 1.0) * 100
    log_message("Optimized Cell         :", atoms.cell.cellpar(),output=log)
    log_message("Optimized Cell diff (%):", cell_diff,output=log)

    return atoms

