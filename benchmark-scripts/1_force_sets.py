import os
import datetime
import warnings
from typing import Literal, Any
from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from ase.constraints import FixSymmetry
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from ase.spacegroup import get_spacegroup
from ase import Atoms
from ase.io import read


from autoWTE import aseatoms2str, get_force_sets, mutlistage_relax, BENCHMARK_ID
from autoWTE.utils import ImaginaryFrequencyError
import autoWTE

from mace.calculators import mace_mp
import torch



# editable config
model_name = "MACE"
checkpoint = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model" #"https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "float64"
calc = mace_mp(model=checkpoint, device=device, default_dtype=dtype)

ase_optimizer : Literal["FIRE", "LBFGS"] = "LBFGS"
ase_filter: Literal["frechet", "exp"] = "frechet"
multistage_relaxation = True
max_steps = 300
force_max = 0.0001  # Run until the forces are smaller than this in eV/A
prog_bar = True


slurm_array_task_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", "debug")
slurm_array_task_min = int(os.getenv("SLURM_ARRAY_TASK_MIN", "0"))


task_type = "LTC" # lattice thermal conductivity
job_name = f"{model_name}-phononDB-{task_type}-{ase_optimizer}{'_MSR' if multistage_relaxation else ''}_{force_max}"
module_dir = os.path.dirname(__file__)
out_dir = os.getenv("SBATCH_OUTPUT", f"{module_dir}/{datetime.datetime.now().strftime('%Y-%m-%d')}-{job_name}")
out_path = f"{out_dir}/force_sets_{slurm_array_job_id}-{slurm_array_task_id:>03}.json.gz"
os.makedirs(os.path.dirname(out_path),exist_ok=True)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib") 

timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')



data_path = autoWTE.BENCHMARK_STRUCTURES


print(f"\nJob {job_name} started {timestamp}")
print(f"{data_path=}")


#check this 
print(f"Read data from {data_path}")
atoms_list: list[Atoms] = read(data_path,format="extxyz",index=":")

if slurm_array_job_id == "debug":
    atoms_list = atoms_list[:5]
    print("Running in DEBUG mode.")
elif slurm_array_task_count > 1:
    atoms_list = np.array_split(np.asarray(atoms_list,dtype = object), slurm_array_task_count)[
        slurm_array_task_id - slurm_array_task_min
    ]




filter_cls: Callable[[Atoms], Atoms] = {
    "frechet": FrechetCellFilter,
    "exp": ExpCellFilter,
}[ase_filter]
optim_cls: Callable[..., Optimizer] = {"FIRE": FIRE, "LBFGS": LBFGS}[ase_optimizer]

force_set_results : dict[str, dict[str, Any]] = {}

tqdm_bar = tqdm(atoms_list, desc="Relaxing and force sets calculation",disable=not prog_bar)
for atoms in tqdm_bar:
    mat_id = atoms.info[autoWTE.BENCHMARK_ID] 
    if mat_id in force_set_results:
        continue
    if "name" in atoms.info.keys():
        mat_name=atoms.info["name"]
    else:
        mat_name=f"{atoms.get_chemical_formula(mode='metal')}-{get_spacegroup(atoms).no}"
    
    tqdm_bar.set_postfix_str(mat_name, refresh=True)

    try:
        atoms.calc = calc
        if max_steps > 0:
            if not multistage_relaxation :
                atoms.set_constraint(FixSymmetry(atoms))
                filtered_atoms = filter_cls(atoms,scalar_pressure = 0 if "residual_pressure" not in atoms.info.keys() else atoms.info["residual_pressure"])
                optimizer = optim_cls(filtered_atoms,logfile='relax.log')
                optimizer.run(fmax=force_max, steps=max_steps)
                if optimizer.step == max_steps:
                    print(f"Material {mat_name=}, {mat_id=} reached max step {max_steps=} during relaxation.")
                
                atoms.constraints = None
                atoms.calc = None
            else:
                atoms = mutlistage_relax(
                    atoms,
                    fmax=force_max,
                    allow_tilt = False,
                    log="relax.log"
                )

    except Exception as exc:
        warnings.warn(f"Failed to relax {mat_name=}, {mat_id=}: {exc!r}")
        continue


    try:
        if True:
            force_set_results[mat_id]={
                "structure" : aseatoms2str(atoms),
                "primitive_matrix" : atoms.info["primitive_matrix"] if "primitive_matrix" in atoms.info.keys() else None,
                "fc2_supercell" : atoms.info["fc2_supercell"],
                "fc3_supercell" : atoms.info["fc3_supercell"],
                "q_mesh" : atoms.info["q_mesh"],
                "name" : mat_name,
            }

            _, fc2_set, fc3_set=get_force_sets(atoms,
                    calculator=calc,
                    cutoff_pair_distance=None,
                    log=False,
                    pbar_kwargs={"leave" : False},
                    check_frequencies=True)


            force_set_results[mat_id]["fc2_set"] = fc2_set
            force_set_results[mat_id]["fc3_set"] = fc3_set
            force_set_results[mat_id]["freqs_positive"] = True

    except ImaginaryFrequencyError as exc:
        warnings.warn(f"Obtained imaginary phonon frequencies in {mat_name},{mat_id}: {exc!r}")
        force_set_results[mat_id]["are_frequencies_positive"] = False
        continue
    except KeyError as exc:
        warnings.warn(f"Failed to calculate force sets {mat_id}: {exc!r}")
        continue


df_out = pd.DataFrame(force_set_results).T
df_out.index.name = BENCHMARK_ID
df_out.reset_index().to_json(out_path)
    
    

