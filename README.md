# autoWTE: Automatic heat-conductivity predictions from the Wigner Transport Equation 

autoWTE employs foundation Machine Learning Interatomic Potentials and phono3py to determine the Wigner Thermal conductivity in crystals with arbitrary composition and structure.

# Install 
Clone repository:
```
git clone https://github.com/MPA2suite/autoWTE.git
```
Then install in editable mode:
```
pip install -e .
```

 Pre-requisites (need to be installed seperately or added to PYTHONPATH)
- phono3py (see https://phonopy.github.io/phono3py/install.html for installation instructions)


Installed automatically during pip install:
- phonopy
- ase
- numpy
- matplotlib
- spglib
- tqdm
- h5py
- pandas




# Usage
The example scripts showcase a sample workflow for testing a MACE potential and comparing the thermal conductivity with DFT calculations for a collection of different materials. The scripts may be modified easily to use any foundation Machine Learning Interatomic Potentials. See autoWTE/MLPS.py for calculator setup utilities.

1. Modify and execute `1_force_sets.py` file in benchmark-scripts to obtain the force sets for second and third order force constants.
2. Modify and execute `2_thermal_conductivity.py` file in benchmark-scripts to obtain the thermal conductivity results needed for the bencmark evaluation.
3. Modify and execute `3_evaluate.py` file in benchmark-scripts to obtain the benchmark metrics (SRME) and results.

# How to cite

```
@misc{póta2024thermalconductivitypredictionsfoundation,
      title={Thermal Conductivity Predictions with Foundation Atomistic Models}, 
      author={Balázs Póta and Paramvir Ahlawat and Gábor Csányi and Michele Simoncelli},
      year={2024},
      eprint={2408.00755},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2408.00755}, 
}
```