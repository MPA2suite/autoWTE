import numpy as np
import pandas as pd

from autoWTE import BENCHMARK_TEMPERATURES

import warnings

FREQUENCY_THRESHOLD = -1e-2
MODE_KAPPA_THRESHOLD = 1e-6

def fill_na_in_list(lst,y):
    return [y if pd.isna(x) else x for x in lst]

def add_benchmark_descriptors(
        df_mlp_filtered,
        df_dft_results,
    ):
    
    df_mlp_filtered = df_mlp_filtered.applymap(lambda x: np.array(x) if isinstance(x, list) else x)
    df_dft_results = df_dft_results.applymap(lambda x: np.array(x) if isinstance(x, list) else x)
    df_mlp_filtered["heat_capacity"] = df_mlp_filtered["heat_capacity"].apply(lambda x: np.array(x))
    
    #[print(df_mlp_filtered[s].apply(lambda x :x.shape),s) for s in ["kappa_TOT_RTA","heat_capacity","temperatures","weights",'mode_kappa_C','mode_kappa_P_RTA']]


    df_mlp_filtered["are_frequencies_positive"] = df_mlp_filtered["frequencies"].apply(are_frequencies_positive)
    
    df_mlp_filtered["kappa_TOT_ave"] = df_mlp_filtered['kappa_TOT_RTA'].apply(calculate_kappa_ave)
    df_dft_results["kappa_TOT_ave"] = df_dft_results['kappa_TOT_RTA'].apply(calculate_kappa_ave)

    df_mlp_filtered["SRD"] = 2*(df_mlp_filtered["kappa_TOT_ave"] - df_dft_results['kappa_TOT_ave'])/(df_mlp_filtered["kappa_TOT_ave"] + df_dft_results['kappa_TOT_ave'])
    df_mlp_filtered["SRD"] = df_mlp_filtered["SRD"].apply(fill_na_in_list,args=(-2,))


    df_mlp_filtered["SRE"] = df_mlp_filtered["SRD"].abs()

    df_mlp_filtered["mode_kappa_TOT"] = df_mlp_filtered.apply(calculate_mode_kappa_TOT,axis=1)

    df_mlp_filtered["SRME"] = calculate_SRME_dataframes(df_mlp_filtered,df_dft_results)

    return df_mlp_filtered



def get_metrics(df_mlp_filtered):

    mSRE = df_mlp_filtered["SRE"].mean()
    rmseSRE = ((df_mlp_filtered["SRE"]-mSRE)**2).mean() ** 0.5

    mSRME = df_mlp_filtered["SRME"].mean()
    rmseSRME = ((df_mlp_filtered["SRME"]-mSRME)**2).mean() ** 0.5

    return mSRE, mSRME, rmseSRE, rmseSRME


def are_frequencies_positive(frequencies):
    if np.all(pd.isna(frequencies)):
        return False
    
    if np.any(frequencies[0,3:] < 0):
        return False
    
    if np.any(frequencies[0,:3] < FREQUENCY_THRESHOLD):
        return False
    
    if np.any(frequencies[1:] < 0):
        return False

    return True


def calculate_kappa_ave(kappa):

    if np.any((pd.isna(kappa))):
        return np.nan
    else:
        _kappa = np.asarray(kappa)

    kappa_ave = _kappa[...,:3].mean(axis=-1)

    return kappa_ave



def calculate_SRME_dataframes(df_mlp,df_dft):

    srme_list = []
    for idx, row_mlp in df_mlp.iterrows():
        row_dft = df_dft.loc[idx]  
    
        result = calculate_SRME(row_mlp,row_dft)
        srme_list.append(result)

    return srme_list


def calculate_SRME(kappas_mlp,kappas_dft):

    if np.all(pd.isna(kappas_mlp["kappa_TOT_ave"])):
        return 2
    if np.any(pd.isna(kappas_mlp["kappa_TOT_RTA"])):
        return 2 #np.nan
    if np.any(pd.isna(kappas_mlp["weights"])):
        return 2 #np.nan
    if np.any(pd.isna(kappas_dft["kappa_TOT_RTA"])):
        return 2 #np.nan
    
    mlp_mode_kappa_TOT = calculate_mode_kappa_TOT(kappas_mlp)

    dft_mode_kappa_TOT = calculate_mode_kappa_TOT(kappas_dft)

    # calculating microscopic error for all temperatures
    microscopic_error = (np.abs(
        calculate_kappa_ave(mlp_mode_kappa_TOT-dft_mode_kappa_TOT) # reduce ndim by 1
        ).sum( axis=tuple(range(1,mlp_mode_kappa_TOT.ndim-1)) ) # summing axes
        / kappas_mlp["weights"].sum())
    
    
    SRME = 2 * microscopic_error / (kappas_mlp["kappa_TOT_ave"]+kappas_dft["kappa_TOT_ave"])
    
    return SRME




def calculate_mode_kappa_TOT(kappas_dict):
    if np.any(pd.isna(kappas_dict["mode_kappa_C"])):
        return np.nan
    if np.any(pd.isna(kappas_dict["heat_capacity"])):
        return np.nan
    if np.any(pd.isna(kappas_dict["mode_kappa_P_RTA"])):
        return np.nan
    
    mode_kappa_C = np.asarray(kappas_dict["mode_kappa_C"])
    heat_capacity = np.asarray(kappas_dict['heat_capacity'])
    mode_kappa_P_RTA = np.asarray(kappas_dict["mode_kappa_P_RTA"])
    kappa_P_RTA = np.asarray(kappas_dict["kappa_P_RTA"])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mode_kappa_C_per_mode = 2*((mode_kappa_C * heat_capacity[:,:,:,np.newaxis,np.newaxis])/(heat_capacity[:,:,:,np.newaxis,np.newaxis]+heat_capacity[:,:,np.newaxis,:,np.newaxis])).sum(axis=2)
    
    mode_kappa_C_per_mode[np.isnan(mode_kappa_C_per_mode)]=0

    mode_kappa_TOT = mode_kappa_C_per_mode + mode_kappa_P_RTA

    sum_mode_kappa_TOT = mode_kappa_TOT.sum(axis = tuple(range(1,mode_kappa_TOT.ndim-1)))/np.sum(kappas_dict["weights"])

    if np.all((sum_mode_kappa_TOT - kappa_P_RTA) <= MODE_KAPPA_THRESHOLD) :
        warnings.warn(f"Total mode kappa does not sum to total kappa. mode_kappa_TOT sum : {sum_mode_kappa_TOT}, kappa_TOT_RTA : {kappa_P_RTA}")

    return mode_kappa_TOT
    


