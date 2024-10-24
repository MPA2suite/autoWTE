import warnings
import os
import pandas as pd


from autoWTE import   glob2df, BENCHMARK_ID, BENCHMARK_DFT_NAC_REF
from autoWTE.benchmark import get_metrics, add_benchmark_descriptors


module_dir = os.path.dirname(__file__)
in_pattern = f"{module_dir}/2024-10-21-MACE-phononDB-LTC-LBFGS_MSR_0.005/conductivity_1058314-*.json.gz"
out_path = f"{module_dir}/2024-10-21-MACE-phononDB-LTC-LBFGS_MSR_0.005/conductivity_1058314.json.gz"


model_name = "MACE"

DFT_RESULTS_FILE = BENCHMARK_DFT_NAC_REF 

DEBUG = False

df_mlp_results = glob2df(in_pattern,max_files=None).set_index(BENCHMARK_ID)

df_mlp_results.to_json(out_path)



# Calculating mean SRME and mean SRE


# Read DFT results
df_dft_results = pd.read_json(DFT_RESULTS_FILE).set_index(BENCHMARK_ID)


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

# Print 
pd.set_option('display.max_rows', None)
df_mlp_final = df_mlp_filtered[["SRME","SRE","kappa_TOT_ave",'DFT_kappa_TOT_ave']].copy()
df_mlp_final["kappa_TOT_ave"] = df_mlp_final["kappa_TOT_ave"].apply(lambda x : x[0] if not pd.isna(x) else x)
df_mlp_final["DFT_kappa_TOT_ave"] = df_mlp_final["DFT_kappa_TOT_ave"].apply(lambda x : x[0] if not pd.isna(x) else x)
print(df_mlp_final.round(4))

print(f"MODEL: {model_name}")
print(f"\tmean SRME: {mSRME}")
print(f"\tmean SRE: {mSRE}")
