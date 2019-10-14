# Dynamic Economics - Replication Codes
## What's in here?
This repo contains the code to replicate the content of my final essay for the course in Dynamic Economics. The course is taught by professors J. Adda and J.S. Goerlach to second-year PhD students in Economics and Finance at Bocconi University.

More precisely, the paper estimates a consumption-savings problem with **habits in consumption**. The code is not intended in *any* way for publication, and it may very well contain errors. Hope not, though.

:boom: Should you be inclined for whatever reason to use my code, please cite the source :boom:

## How to run
##### Required Packages
All packages are contained in a standard Anaconda distribution. In particular, I use `numpy`, `scipy`, `numba`, `matplotlib`, `pandas` and `time`. All codes are written in Python 3. Please put all files in the same directory before running the scripts.

##### Run the code
The file `simulate_data.py` solves the model *once*, generates exogenous processes for income, and populates a dataset over which the subsequent estimation is performed. Hence, it should be run first. Estimated runtime around 1'.

The file `main.py` performs estimation. The code supports with minor changes estimation of α alone, and α,σ jointly. The following instructions describe such minor changes:
* To estimate α **alone**:
  - In `main.py`, in the `PLOT RESULTS` section (*i.e.* lines 26-70) comment lines 29-50 and uncomment lines 52-69;
  - In `estimation.py`:
    1. In line 64, set `self.brute_run_estimation(double = False)` ;
    2. Comment line 164 and uncomment line 165.
* To estimate α and σ **jointly**:
  - In `main.py`, in the `PLOT RESULTS` section (*i.e.* lines 26-70) uncomment lines 29-50 and comment lines 52-69;
  - In `estimation.py`:
    1. In line 64, set `self.brute_run_estimation(double = True)` ;
    2. Uncomment line 164 and comment line 165.
   
Expected runtime for single-parameter estimation is around 5'. Double-parameter estimation takes about 1h.
    
