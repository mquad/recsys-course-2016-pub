# recsys-course
This is the official repository for the 2016 Recommender Systems course at Polimi.

## Install

- Install Miniconda for Python3.5 [link](http://conda.pydata.org/miniconda.html)
- Create the virtual environment: `conda create -n recsys-env --file requirements.txt`
- Activate the virtual environment: `source activate recsys-env`
- Install recpy: `sh install.sh`
- Run one example: `sh run_example.sh`

NOTE: to deactivate the virtual environment run: `source deactivate recsys-env`

# Packages
- `recpy`: main package
- `recpy/recommenders`: recommendation algorithms
- `recpy/utils`: dataset management and split utils
- `recpy/cython`: Cython code



