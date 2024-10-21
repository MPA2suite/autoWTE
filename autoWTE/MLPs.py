from autoWTE.fc_interface import *
from autoWTE.load import *

def setup_CHGNet_calculator(device="cpu"):
    from chgnet.model.dynamics import CHGNetCalculator
    return CHGNetCalculator(use_device=device)

def relax_CHGNet_ASE(atoms):
    from chgnet.model import StructOptimizer
    from pymatgen.io.ase import AseAtomsAdaptor
    
    structure=AseAtomsAdaptor.get_structure(atoms)
    relaxer = StructOptimizer()
    result = relaxer.relax(structure)
    return AseAtomsAdaptor.get_atoms(structure)

def relax_M3GNet_ASE(atoms):
    from matgl.ext.ase import  Relaxer
    import matgl
    from pymatgen.io.ase import AseAtomsAdaptor
    
    structure=AseAtomsAdaptor.get_structure(atoms)
    pot=matgl.load_model('M3GNet-MP-2021.2.8-PES')
    relaxer = Relaxer(potential=pot)
    result = relaxer.relax(structure)
    return AseAtomsAdaptor.get_atoms(structure)


def setup_M3GNet_calculator(device):
    import matgl
    from matgl.ext.ase import M3GNetCalculator
    pot=matgl.load_model('M3GNet-MP-2021.2.8-PES')
    calc= M3GNetCalculator(potential=pot)
    return calc


def setup_MACE_lammps_calculator(ph3,path_MACE_potential='/mnt/scratch2/q13camb_scratch/POTENTIALS/foundational/MACE/m0-L2/2024-01-07-mace-128-L2_epoch-199.model-lammps.pt'):
    from ase.calculators.lammpslib import LAMMPSlib
    import numpy as np
    from ase.symbols import Symbols
    
    atoms=ph3.unitcell

    atom_type_unique = np.unique(atoms.get_atomic_numbers(), return_index=True)
    sort_index = np.argsort(atom_type_unique[1])
    elements_atomic_numbers = np.array(atom_type_unique[0])[sort_index]
    atoms_symbols_string=" ".join(list(Symbols(elements_atomic_numbers)))
    print("Calculating %s"%(atoms_symbols_string))
    atoms_dict=dict(zip(list(Symbols(elements_atomic_numbers)),range(1,len(elements_atomic_numbers)+1)))
    print(atoms_dict)

    

    #lammps commands 
    lammps_cmds = [    'pair_style    mace no_domain_decomposition',
                        'pair_coeff    * * %s %s'%(path_MACE_potential,atoms_symbols_string)
        ]

    #lammps header
    lammps_header=[ 'units metal',
                    'boundary p p p',
                    'box tilt large',
                    'atom_style atomic',
                    'atom_modify   map yes',
                    'newton        on'
        ]

    
    #lammps log file
    lammps_log_file='lammps_out.log'


    #lammps calculator
    calculator=LAMMPSlib(lmpcmds=lammps_cmds,
                        log_file=lammps_log_file,
                        lammps_header=lammps_header, 
                        atom_types=atoms_dict,
                        keep_alive=True)

    return calculator

def setup_MACE_ASE_calculator(device="cpu",path_MACE_potential='/mnt/scratch2/q13camb_scratch/POTENTIALS/foundational/MACE/m0-L2/2024-01-07-mace-128-L2_epoch-199.model'):
    from mace.calculators import mace_mp

    calculator=mace_mp(model=path_MACE_potential,device=device,default_dtype="float64")

    return calculator

def setup_SevenNet_calculator(device="cpu"):
    from sevenn.sevennet_calculator import SevenNetCalculator

    calculator= SevenNetCalculator(device=device)

    return calculator 

def setup_ORB_MPtraj_calculator(device="cpu"):
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    orbff = pretrained.orb_v1_mptraj_only() # or choose another model using ORB_PRETRAINED_MODELS[model_name]()
    calculator = ORBCalculator(orbff, device=device)

    return calculator