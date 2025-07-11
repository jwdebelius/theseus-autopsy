# theseus-autopsy

Simulations of the effect of modeling on clustered data 

# Enviromental Requirements

This code has been evaluted on a Mac, but has not been executed in a windows system. 

The code is designed to run using python v 3.10 with numpy v 1.26, scipy v 1.14, pandas v. 2.2.3, statsmodels v., matplotlib v. 3.9.2 and seaborn v. 0.13.2. The enviroment can be found in the enviroment.txt file.

Additionally, the code requries LaTeX to be installed to render figures appropriately. A system installation with the multirow package installed is appropriate. Please refer to [Text rendering with LatTex](https://matplotlib.org/3.9.3/users/explain/text/usetex.html) in the Matplotlib documentation.

# Execution

The simulations are run throught the `Simulations.ipynb` notebook found in the `ipynb` folder. Install and activate the appropriate conda enviroment, and the notebook should run. Please note that due to the size of the simulations, this may be slow. Code specific to individaul steps (e.g. wrappers for modeling functions) can be found in the `ipynb/scripts` folder.

# Funding

Research reported in this publication was supported by the Environmental influences on Child Health Outcomes (ECHO) Program, Office of the Director, National Institutes of Health, under Award Number U24OD023382 (Data Analysis Center). All analyses and code are the responsibiliity of the study statisticians, independently of the sponsor. 