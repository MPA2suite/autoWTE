import os
from ase import Atoms
from numpy.typing import ArrayLike
from typing import List, Tuple, Any, Union, Literal
from copy import deepcopy
from pathlib import Path
from copy import deepcopy
import warnings

from phonopy.structure.atoms import PhonopyAtoms
from phono3py.api_phono3py import Phono3py
from phono3py.cui.load import load
from phono3py.cui.phono3py_script import run_isotope_then_exit, init_phph_interaction
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from phono3py.cui.settings import Phono3pyConfParser, Phono3pySettings

from autoWTE.utils import aseatoms2phono3py, log_message
from autoWTE import BENCHMARK_TEMPERATURES

KAPPA_OUTPUT_NAME_MAP = {
    "weights" : "grid_weights",
    "heat_capacity" : "mode_heat_capacities",
}

CONDUCTIVITY_SIGMAS_LIST = [
    "kappa",
    "mode_kappa",
    "kappa_TOT_RTA",
    "kappa_P_RTA",
    "kappa_C",
    "mode_kappa_P_RTA",
    "mode_kappa_C",
    "gamma_isotope"
]

CONDUCTIVITY_DEFAULT_KWARGS = {
    'is_isotope' : True,
    'conductivity_type' : 'wigner',
    'boundary_mfp' : 1e6
}



def calculate_conductivity_phono3py(
        ph3 : Phono3py,
        temperatures : ArrayLike = BENCHMARK_TEMPERATURES,
        q_mesh : ArrayLike | None = None,
        dict_output : Union[Literal["all","benchmark"],dict] = "all",
        log : str | Path | bool | None = None, 
        nac_method : Literal["Gonze","gonze","Wang","wang"] | None = None,
        **kwargs
        ) -> Phono3py:


    if log == False:
       ph3._log_level = 0
    elif log is not None:
        ph3._log_level = 1


    if nac_method in ["Wang","wang"] :
        if ph3.nac_params is None:
            raise ValueError("No NAC parameters in phono3py object, when setting nac_method Wang.")
        else: 
            ph3.nac_params["method"] = "wang"
    elif nac_method in ["Gonze","gonze"]:
        if ph3.nac_params is None:
            raise ValueError("No NAC parameters in phono3py object, when setting nac_method Wang.")
        else: 
            ph3.nac_params["method"] = 'gonze'
    


    if q_mesh is not None:
        ph3.mesh_numbers = q_mesh

    ph3.init_phph_interaction(
        symmetrize_fc3q=False,
    )

    _kwargs = kwargs

    if kwargs is None :
        _kwargs = CONDUCTIVITY_DEFAULT_KWARGS
    else:
        for key, value in CONDUCTIVITY_DEFAULT_KWARGS.items():
            if key not in _kwargs:
                _kwargs[key] = value
    
    
    ph3.run_thermal_conductivity(
        temperatures = temperatures,
        **_kwargs
    )

    cond = ph3.thermal_conductivity

    output_names = {}
    if dict_output == "benchmark" :
        output_names = {
            'frequencies' : True,
            'qpoints' : True,
            'temperatures' : True,
            'weights' : True,
            'kappa_TOT_RTA' : True,
            'kappa_P_RTA' : True,
            'kappa_C' : True,
            'mode_kappa_P_RTA' : True,
            'mode_kappa_C' : True,
            'heat_capacity' : True
        }
    elif dict_output == "all" :
        output_names = {
            'frequencies' : True,
            'qpoints' : True,
            'temperatures' : True,
            'weights' : True,
            'kappa_TOT_RTA' : True,
            'kappa_P_RTA' : True,
            'kappa_C' : True,
            'mode_kappa_P_RTA' : True,
            'mode_kappa_C' : True,
            'heat_capacity' : True,
            'gamma' : True,
            'gamma_isotope' : True,
            'group_velocities' : True,
            'gv_by_gv' : True,
            'velocity_operator' : True,
            'gv_by_gv_operator' : True
        }
    elif dict_output is dict:
        output_names = dict_output

    kappa_dict = {}
    for key, value in output_names.items():
        try: 

            if value and key in KAPPA_OUTPUT_NAME_MAP: 
                if KAPPA_OUTPUT_NAME_MAP[key] in CONDUCTIVITY_SIGMAS_LIST and getattr(cond,KAPPA_OUTPUT_NAME_MAP[key]).shape[0] == 1:
                    kappa_dict[key] = deepcopy(getattr(cond,KAPPA_OUTPUT_NAME_MAP[key])[0])
                else:
                    kappa_dict[key] = deepcopy(getattr(cond,KAPPA_OUTPUT_NAME_MAP[key]))
            elif value : 
                if key in CONDUCTIVITY_SIGMAS_LIST and getattr(cond,key).shape[0] == 1:
                    kappa_dict[key] = deepcopy(getattr(cond,key)[0])
                else:
                    kappa_dict[key] = deepcopy(getattr(cond,key))
        except AttributeError as exc:
            warnings.warn(f"Phono3py conductivity does not have attribute {key=}: {exc}")



    return ph3, kappa_dict

def calculate_conductivity_atoms(
    atoms : Atoms | PhonopyAtoms, 
    fc2_set : ArrayLike,
    fc3_set : ArrayLike,
    q_mesh : ArrayLike,
    temperatures : ArrayLike = BENCHMARK_TEMPERATURES,
    phono3py_kwargs : dict[str, Any] | None = None,
    fc2_supercell : ArrayLike | None = None,
    fc3_supercell : ArrayLike | None = None,
    primitive_matrix : ArrayLike | str | None = "auto",
    symmetrize_fc : bool = True,
    save_fc_filename : str | bool = False,
    disp_kwargs : dict[str, Any] | None = None,
    log : str | Path | bool = True, 
    dict_output : Union[Literal["all","benchmark"],dict] = "all",
    **kwargs
    ) -> Tuple[Phono3py, dict]:

    if isinstance(atoms,Atoms) : 
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
        
        if primitive_matrix is not None :
            _primitive_matrix = primitive_matrix
        elif "primitive_matrix" in atoms.info.keys() :
                _primitive_matrix = atoms.info["primitive_matrix"]
        else :
            _primitive_matrix = primitive_matrix
        
    elif isinstance(atoms,PhonopyAtoms) :
        if fc2_supercell is not None :
            _fc2_supercell = fc2_supercell
        else:
            raise ValueError(f'{atoms.get_chemical_formula(mode="metal")=} "fc2_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.')
        
        if fc3_supercell is not None :
            _fc3_supercell = fc3_supercell
        else:
            raise ValueError(f'{atoms.get_chemical_formula(mode="metal")=} "fc3_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.')
    
    if phono3py_kwargs is None:
        _phono3py_kwargs = {}
    else:
        _phono3py_kwargs = phono3py_kwargs
    
    ph3 = aseatoms2phono3py(atoms,
                  fc2_supercell=_fc2_supercell,
                  fc3_supercell=_fc3_supercell,
                  primitive_matrix=_primitive_matrix,
                  log_level = 1 if log != False else 0,
                  **_phono3py_kwargs)
    
    log_message("Generating displacement",output=log)
    if disp_kwargs is None : 
        disp_kwargs = {"distance" : 0.03,
                       "cutoff_pair_distance" : None}
    ph3.generate_displacements(**disp_kwargs)
    

    log_message("Setting and producing force constants",output=log)
    ph3.phonon_forces = fc2_set
    ph3.forces = fc3_set
    ph3.produce_fc2(symmetrize_fc2=symmetrize_fc)
    ph3.produce_fc3(symmetrize_fc3r=symmetrize_fc)

    if save_fc_filename != False :
        log_message("Saving HDF5 force constants.",output=log)
        if save_fc_filename == True:
            fc_filename_append = ""
        else:
            fc_filename_append = f".{save_fc_filename}"
        write_fc2_to_hdf5(
            ph3.fc2,
            p2s_map=ph3.phonon_primitive.p2s_map,
            physical_unit="eV/angstrom^2",
            filename = f"fc2{fc_filename_append}.hdf5"
        )
        write_fc3_to_hdf5(
            ph3.fc3,
            p2s_map=ph3.primitive.p2s_map,
            physical_unit="eV/angstrom^2",
            filename = f"fc3{fc_filename_append}.hdf5"
        )
    
    log_message("Starting thermal conductivity calculation",output=log)
    
    if kwargs is None:
        kwargs = {
            "is_isotope" : True,
            "conductivity_type" : 'wigner' 
        } 

    result = calculate_conductivity_phono3py(ph3,temperatures,q_mesh=q_mesh,dict_output = dict_output, **kwargs)

    return result




def calculate_conductivity_load(
    displacement_output : str ,
    q_mesh : ArrayLike,
    temperatures : ArrayLike = BENCHMARK_TEMPERATURES,
    fc2_set : ArrayLike | None = None,
    fc3_set : ArrayLike | None = None,
    produce_fc : bool= True,
    symmetrize_fc : bool = True,
    save_fc_filename : str | bool = False,
    log : str | Path | bool = True, 
    dict_output : Union[Literal["all","benchmark"],dict] = "all",
    **kwargs
    ) -> Tuple[Phono3py, dict]:

    ph3 = load(displacement_output,produce_fc=produce_fc,symmetrize_fc=symmetrize_fc)

    if fc2_set is not None:
        ph3.phonon_forces = fc2_set
        ph3.produce_fc2(symmetrize_fc2=symmetrize_fc)
    

    if fc3_set is not None: 
        ph3.forces = fc2_set
        ph3.produce_fc3(symmetrize_fc3r=symmetrize_fc)

    if save_fc_filename != False :
        log_message("Saving HDF5 force constants.",output=log)
        if save_fc_filename == True:
            fc_filename_append = ""
        else:
            fc_filename_append = f".{save_fc_filename}"
        write_fc2_to_hdf5(
            ph3.fc2,
            p2s_map=ph3.phonon_primitive.p2s_map,
            physical_unit="eV/angstrom^2",
            filename = f"fc2{fc_filename_append}.hdf5"
        )
        write_fc3_to_hdf5(
            ph3.fc3,
            p2s_map=ph3.primitive.p2s_map,
            physical_unit="eV/angstrom^2",
            filename = f"fc3{fc_filename_append}.hdf5"
        )
    
    log_message("Starting thermal conductivity calculation",output=log)


    
    
    if kwargs is None:
        kwargs = {
            "is_isotope" : True,
            "conductivity_type" : 'wigner' 
        } 

    result = calculate_conductivity_phono3py(ph3,temperatures,q_esh=q_mesh, dict_output = dict_output, **kwargs)

    return result


    

    









