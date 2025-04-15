# Pregnancy blood test dynamics resemble rejuvenation of some organs and aging of others

[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2025.02.24.639848-blue)](https://www.biorxiv.org/content/10.1101/2025.02.24.639848)

The repository contains the data and jupyter notebook for the analysis and graphics of the paper "<b>Pregnancy blood test dynamics resemble rejuvenation of some organs and aging of others</b>" by Moran et al. (2025).
For information about the data, please refer to [this page on Dryad 🔗](https://datadryad.org/dataset/doi:10.5061/dryad.1c59zw44t).

# Running the notebook

Jupyter notebook was used to run the analyses and to create the graphs. To re-run, [install anaconda or miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and download the repository.
Open the command prompt/terminal in the root directory of the repository and install the environment:
```
conda env create -f environment.yml
```
Start the conda environment and the jupyter server:
```
conda activate pregnancy
jupyter notebook
```
And run the notebooks from the new browser window. For support regarding running Jupyter notebooks, please refer to [Jupyter's support page](https://docs.jupyter.org/en/latest/running.html).

A more brief requirements file is provided in `requirements.txt` for installation of the packages necessary for the plots and analyses alone.<br>
Use the requirements file if you have a working python environment with Jupyter installed. With the virtual environment activated, run the following from the root directory of the repository in the command line:
```
python -m pip install -r requirements.txt
jupyter notebook
```


