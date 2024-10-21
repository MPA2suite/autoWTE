import os


PKG_NAME = "autoWTE"
__version__ = "0.1.1"

PKG_DIR = os.path.dirname(__file__)
# repo root directory if editable install, TODO: else the package dir
ROOT = os.path.dirname(PKG_DIR)
DATA_DIR = f"{ROOT}/data"  # directory to store default data
BENCHMARK_DATA_DIR = f"{DATA_DIR}"
BENCHMARK_STRUCTURES_FILE = "phononDB-PBE-structures.extxyz"
BENCHMARK_STRUCTURES = f"{DATA_DIR}/{BENCHMARK_STRUCTURES_FILE}"

BENCHMARK_DFT_NAC_REF_FILE = "kappas_phononDB_PBE_NAC.json.gz"
BENCHMARK_DFT_NONAC_REF_FILE = "kappas_phononDB_PBE_NAC.json.gz"
BENCHMARK_DFT_NAC_REF = f"{DATA_DIR}/{BENCHMARK_DFT_NAC_REF_FILE}"
BENCHMARK_DFT_NONAC_REF = f"{DATA_DIR}/{BENCHMARK_DFT_NONAC_REF_FILE}"

###
pkg_is_editable = True


######
BENCHMARK_TEMPERATURES = [300]
BENCHMARK_ID = "mp_id"

from autoWTE.fc_calc import *
from autoWTE.tc_calc import *
from autoWTE.utils import *
from autoWTE.MLPs import *
from autoWTE.relax import *
from autoWTE.data import glob2df
from autoWTE.benchmark import *