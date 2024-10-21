import os


#from autoWTE.data import *
from importlib.metadata import Distribution, version


PKG_NAME = "autoWTE"
__version__ = "0.0.0"

PKG_DIR = os.path.dirname(__file__)
# repo root directory if editable install, else the pkg directory
ROOT = os.path.dirname(PKG_DIR)
DATA_DIR = f"{ROOT}/data"  # directory to store raw data
BENCHMARK_DATA_DIR = f"{DATA_DIR}"
BENCHMARK_STRUCTURES_FILE = "phononDB-PBE-structures.extxyz"
BENCHMARK_STRUCTURES = f"{DATA_DIR}/{BENCHMARK_STRUCTURES_FILE}"

BENCHMARK_DFT_NAC_REF_FILE = "benchmark_kappas_phononDB_PBE_NAC.json.gz"
BENCHMARK_DFT_NONAC_REF_FILE = "benchmark_kappas_phononDB_PBE_NAC.json.gz"
BENCHMARK_DFT_NAC_REF = f"{DATA_DIR}/{BENCHMARK_DFT_NAC_REF_FILE}"
BENCHMARK_DFT_NONAC_REF = f"{DATA_DIR}/{BENCHMARK_DFT_NONAC_REF_FILE}"

###
pkg_is_editable = True


######
BENCHMARK_TEMPERATURES = [300]
BENCHMARK_ID = "mp_id"

from autoWTE.fc_interface import *
from autoWTE.tc_calc import *
from autoWTE.load import *
from autoWTE.MLPs import *
from autoWTE.relax import *
from autoWTE.data import glob2df
from autoWTE.benchmark import *