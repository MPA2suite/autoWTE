import warnings
import os, sys
from copy import deepcopy
from typing import Any
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

from autoWTE import  calculate_conductivity_atoms, BENCHMARK_TEMPERATURES
from autoWTE.benchmark import get_metrics, add_benchmark_descriptors

module_dir = os.path.dirname(__file__)
in_pattern = f"{module_dir}/2024-10-18-MACE-phonondb3-LTC-FIRE_0.005/conductivity_989612-*.json.gz"
out_path = f"{module_dir}/2024-10-18-MACE-phonondb3-LTC-FIRE_0.005/conductivity_989612.json.gz"


model_name = "MACE"

DFT_RESULS_FILE = "../data/kappas_phonondb3_Togo_PBE_NAC.json.gz"

DEBUG = False

in_files = sorted(glob(in_pattern))

print(*in_files,sep="\n")

# join NAC files
dfs = {}
for file_path in tqdm(in_files):
    if file_path in dfs:
        continue
    df_i = pd.read_json(file_path).set_index("index")
    dfs[file_path] = df_i

df_mlp_results = pd.concat(dfs.values())

#df_mlp_results= pd.read_json(in_path).set_index("index")#, default_handler=as_dict_handler)
df_mlp_results.to_json(out_path)

print(df_mlp_results)

# Calculating mean SRME and mean SRE


# Read DFT results
df_dft_results = pd.read_json(DFT_RESULS_FILE).set_index("index")
print(df_dft_results.keys())

if not DEBUG:
    df_mlp_filtered=df_mlp_results[df_mlp_results.index.isin(df_dft_results.index)]
    df_mlp_filtered = df_mlp_filtered.reindex(df_dft_results.index)
else:
    common_indexes = df_mlp_results.index.intersection(df_dft_results.index)
    df_mlp_filtered = df_mlp_results.reindex(common_indexes)
    df_dft_results = df_dft_results.reindex(common_indexes)

df_mlp_filtered = add_benchmark_descriptors(df_mlp_filtered,df_dft_results)

mSRE, mSRME, rmseSRE, rmseSRME = get_metrics(df_mlp_filtered)

df_mlp_filtered["DFT_kappa_TOT_ave"] = df_dft_results["kappa_TOT_ave"]

pd.set_option('display.max_rows', None)
print(df_mlp_filtered[["SRME","SRE","kappa_TOT_ave",'DFT_kappa_TOT_ave']].round(4))

print(f"MODEL: {model_name}")
print(f"\tmean SRME: {mSRME}")
print(f"\tmean SRE: {mSRE}")
