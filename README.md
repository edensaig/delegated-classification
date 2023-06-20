# Delegated Classification

Comments are welcome! 🙂

## Environment setup

Python:
```
conda create -n dclf anaconda
conda activate dclf
conda install -c conda-forge pyomo
```

Data:
* After setting up the Python environment, download the LCDB database from Github ([`database.tar.7z`](https://github.com/fmohr/lcdb/blob/main/python/lcdb/database.tar.7z)) and extract its contents to `~/Documents/data/delegated_classification/lcdb/database`.

## Repository structure

* The root directory contains analysis notebooks generating all the figures and numerical results presented in the paper. 
* The `dclf` library contains implementation of contract design optimizers, simulation and analysis code. Code is imported as a module by the analysis notebooks.

