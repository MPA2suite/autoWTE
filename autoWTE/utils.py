from phono3py.cui.load import load
from ase.io import read, write
from ase.io.vasp import read_vasp
from ase import Atoms
from ase.spacegroup.symmetrize import check_symmetry
from phonopy.structure.atoms import PhonopyAtoms
from phono3py.api_phono3py import Phono3py
from phonopy.api_phonopy import Phonopy
from phonopy import file_IO
import h5py
import numpy as np
import datetime
import io
import sys



def aseatoms2phonoatoms(atoms):
    phonoatoms = PhonopyAtoms(atoms.symbols, cell=atoms.cell, positions=atoms.positions, pbc=True)
    return phonoatoms


def aseatoms2phono3py(atoms,fc2_supercell,fc3_supercell,primitive_matrix=None, **kwargs) -> Phono3py:
    unitcell=aseatoms2phonoatoms(atoms)
    return Phono3py(unitcell=unitcell,supercell_matrix=fc3_supercell,phonon_supercell_matrix=fc2_supercell,primitive_matrix=primitive_matrix,**kwargs)

def aseatoms2str(atoms, format='extxyz'):
    buffer = io.StringIO()
    write(buffer, Atoms(atoms.symbols,positions=atoms.positions,cell=atoms.cell,pbc=True), format=format)  # You can choose different formats (like 'cif', 'pdb', etc.)
    atoms_string = buffer.getvalue()
    return atoms_string

def str2aseatoms(atoms_string,format='extxyz'):
    buffer = io.StringIO(atoms_string)
    return read(buffer, format=format)

def log_message(*messages,output=True,sep=" ",**kwargs):

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    write_message = f"{timestamp} - {sep.join((str(m) for m in messages))}"

    if output == False:
        pass
    elif output == True:
        print(write_message,**kwargs)
        sys.stdout.flush()
    elif isinstance(output, str):
        # Write to the file specified by 'output'
        with open(output, 'a') as file:
            file.write(write_message + '\n',**kwargs)
    elif hasattr(output, 'write'):
        # Write to the file object
        output.write(write_message + '\n',**kwargs)
        output.flush()

def phono3py2aseatoms(ph3 : Phono3py) -> Atoms:
    phonopyatoms=ph3.unitcell
    atoms = Atoms(phonopyatoms.symbols, cell=phonopyatoms.cell, positions=phonopyatoms.positions, pbc=True)
    
    if ph3.supercell_matrix is not None:
        atoms.info["fc3_supercell"] = ph3.supercell_matrix

    if ph3.phonon_supercell_matrix is not None:
        atoms.info["fc2_supercell"] = ph3.phonon_supercell_matrix

    if ph3.primitive_matrix is not None:
        atoms.info["primitive_matrix"] = ph3.primitive_matrix

    if ph3.mesh_numbers is not None:
        atoms.info["q_mesh"] = ph3.mesh_numbers

    # TODO : Non-default values and BORN charges to be added

    return atoms

def get_chemical_formula(ph3 : Phono3py, mode='metal', **kwargs):
    unitcell=ph3.unitcell
    atoms = Atoms(unitcell.symbols, cell=unitcell.cell, positions=unitcell.positions, pbc=True)
    return atoms.get_chemical_formula(mode=mode,**kwargs)

class ImaginaryFrequencyError(Exception):
    def __init__(self, value):
        self.value = value
 
    def __str__(self):
        return(repr(self.value))