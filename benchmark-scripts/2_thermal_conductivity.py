import autoWTE
import warnings
import os
from copy import deepcopy
from typing import Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from autoWTE.load import str2aseatoms
from autoWTE.data import glob2df
from autoWTE import  calculate_conductivity_atoms, BENCHMARK_TEMPERATURES, BENCHMARK_ID, BENCHMARK_DFT_NAC_REF
from autoWTE.benchmark import get_metrics, add_benchmark_descriptors

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib") 

module_dir = os.path.dirname(__file__)

model_name = "MACE"
checkpoint = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model"
in_path = f"{module_dir}/2024-10-21-MACE-phonondb3-LTC-LBGS_MSR_0.005/force_sets_interactive-000.json.gz"
DFT_RESULS_FILE = BENCHMARK_DFT_NAC_REF

prog_bar = True

df_force_sets = glob2df(in_path,pbar=False).set_index(BENCHMARK_ID)


slurm_array_task_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID", "debug")
slurm_array_task_min = int(os.getenv("SLURM_ARRAY_TASK_MIN", "0"))

out_path = f"{os.path.dirname(in_path)}/conductivity_{slurm_array_job_id}-{slurm_array_task_id:>03}.json.gz"

if slurm_array_job_id == "debug":
    index_list = df_force_sets.index.tolist()[:3]
elif slurm_array_task_count > 1:
    index_list = df_force_sets.index.tolist()[slurm_array_task_id - slurm_array_task_min::slurm_array_task_count]

df_force_sets = df_force_sets.loc[index_list]
tc_results: dict[str, dict[str, Any]] = {}

pbar = tqdm(df_force_sets.itertuples(),disable=not prog_bar)


for force_sets in pbar:
    #try: 
        mat_id = force_sets.index 

        pbar.set_postfix_str(mat_id)

        _, tc_results[mat_id] = calculate_conductivity_atoms(
            atoms=str2aseatoms(force_sets.structure),
            temperatures=BENCHMARK_TEMPERATURES,
            q_mesh=force_sets.q_mesh,
            fc2_set=force_sets.fc2_set,
            fc3_set=force_sets.fc3_set,
            primitive_matrix=force_sets.primitive_matrix,
            fc2_supercell=np.diagonal(force_sets.fc2_supercell),
            fc3_supercell=np.diagonal(force_sets.fc3_supercell),
            log=True,
            dict_output="benchmark"
        )

        tc_results[mat_id].update({
            "name" : force_sets.name,
            "fc2_supercell" : force_sets.fc2_supercell,
            'fc3_supercell' : force_sets.fc3_supercell,
            "q_mesh" : force_sets.q_mesh,
        })
    #except Exception as exc:
    #    print(f"Failed to conductivity calculation {mat_id}: {exc!r}")
    #    continue


df_mlp_results = pd.DataFrame(tc_results).T
df_mlp_results.index.name = BENCHMARK_ID 
df_mlp_results.reset_index().to_json(out_path)



