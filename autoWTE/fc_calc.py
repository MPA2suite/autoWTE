from pathlib import Path

from numpy.typing import ArrayLike
from typing import Tuple, Any


from ase import Atoms
from ase.calculators.calculator import Calculator
import numpy as np
from tqdm import tqdm


from phono3py.api_phono3py import Phono3py

from autoWTE.utils import aseatoms2phono3py, log_message, get_chemical_formula, ImaginaryFrequencyError
from autoWTE.benchmark import are_frequencies_positive, FREQUENCY_THRESHOLD



def calculate_fc2_set(ph3,calculator,log=True, pbar_kwargs={}):
    # calculate FC2 force set


    log_message(f"Computing FC2 force set in {get_chemical_formula(ph3)=}.", output=log)

    forces = []
    nat = len(ph3.phonon_supercells_with_displacements[0])

    for sc in tqdm(ph3.phonon_supercells_with_displacements,desc = f"FC2 calculation: {get_chemical_formula(ph3)}",**pbar_kwargs):
        if sc is not None:
            atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
            atoms.calc =calculator
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    ph3.phonon_forces = force_set
    return force_set

def calculate_fc3_set(ph3,calculator,log=True, pbar_kwargs={}):
    # calculate FC3 force set

    log_message(f"Computing FC3 force set in {get_chemical_formula(ph3)=}.", output=log)

    forces = []
    nat = len(ph3.supercells_with_displacements[0])

    for sc in tqdm(ph3.supercells_with_displacements,desc = f"FC3 calculation: {get_chemical_formula(ph3)}",**pbar_kwargs):
        if sc is not None:
            atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
            atoms.calc =calculator
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    ph3.forces = np.array(forces)
    return force_set



def get_force_sets(
    atoms : Atoms,
    calculator : Calculator | None = None,
    fc2_supercell : ArrayLike | None = None,
    fc3_supercell : ArrayLike | None = None,
    primitive_matrix : ArrayLike | str | None = "auto",
    q_mesh : ArrayLike | None = None,
    cutoff_pair_distance : float | None = None,
    disp_kwargs : dict[str, Any] = {},
    log : str | Path | bool = True,
    displacement_output : str | bool = False,
    pbar_kwargs : dict = {"leave" : False},
    check_frequencies : bool = False,
    **kwargs
    ) -> Tuple[Phono3py,list, list]:
    """Calculate fc2 and fc3 force lists from phono3py.

    Args:
        

    Raises:
        

    Returns:
        
    """

    if fc2_supercell is not None :
        _fc2_supercell = fc2_supercell
    else:
        if "fc2_supercell" in atoms.info.keys() :
            _fc2_supercell = atoms.info["fc2_supercell"]
        else:
            raise ValueError(f'{atoms.get_chemical_formula(mode="metal")=} "fc2_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.')
    
    if fc3_supercell is not None :
        _fc3_supercell = fc3_supercell
    else:
        if "fc3_supercell" in atoms.info.keys() :
            _fc3_supercell = atoms.info["fc3_supercell"]
        else:
            raise ValueError(f'{atoms.get_chemical_formula(mode="metal")=} "fc3_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.')
    
    if calculator is not None :
        _calculator = calculator
    else:
        if getattr(atoms, "calc", None) is not None :
            _calculator = atoms.calc
        else:
            raise ValueError(f'{atoms.get_chemical_formula(mode="metal")=} "calculator" was not found in atoms object and was not provided as an argument when calculating force sets.')
    
    if primitive_matrix is not None :
        _primitive_matrix = primitive_matrix
    elif "primitive_matrix" in atoms.info.keys() :
            _primitive_matrix = atoms.info["primitive_matrix"]
    else :
        _primitive_matrix = primitive_matrix

    if q_mesh is not None :
        _q_mesh = q_mesh
    else:
        if "q_mesh" in atoms.info.keys():
            _q_mesh = atoms.info["q_mesh"]
        


    # Initialise Phono3py object
    ph3 = aseatoms2phono3py(atoms,
                  fc2_supercell=_fc2_supercell,
                  fc3_supercell=_fc3_supercell,
                  primitive_matrix=_primitive_matrix,
                  **kwargs)
    
    if _q_mesh is not None:
        ph3.mesh_numbers = _q_mesh
    
    if disp_kwargs is None : 
        disp_kwargs = {"distance" : 0.03,
                       "cutoff_pair_distance" : cutoff_pair_distance}
    ph3.generate_displacements(**disp_kwargs)
    
    if displacement_output != False :
        if displacement_output == True :
            ph3.save()
        else:
            ph3.save(displacement_output)
    
    fc2_set = calculate_fc2_set(ph3, calculator,log=log,pbar_kwargs=pbar_kwargs)

    if check_frequencies:
        frequencies = get_phonon_freqs_ph3(ph3)

        if not are_frequencies_positive(frequencies) :
            raise ImaginaryFrequencyError(frequencies[frequencies<FREQUENCY_THRESHOLD])
        
        
    fc3_set = calculate_fc3_set(ph3, calculator,log=log,pbar_kwargs=pbar_kwargs)

    return ph3, fc2_set, fc3_set

def get_phonon_freqs_ph3(ph3, q_mesh=None):
    
    if q_mesh is None:
        if ph3.mesh_numbers is None:
            raise ValueError(f'"mesh_number" was not found in phono3py object and was not provided as an argument when calculating phonons from phono3py object.')
    else :
        ph3.mesh_numbers = q_mesh    

    

    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.init_phph_interaction()
    ph3.run_phonon_solver()


    freqs, eigvecs, grid = ph3.get_phonon_data()

    return freqs