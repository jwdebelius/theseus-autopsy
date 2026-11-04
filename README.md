# theseus-autopsy

This is code to support Kress, Debelius, Bakre, et al (2026). "A Guide to 
Combining Individual Participant from Observational Studies in Consortia 
Using Data from the Environmental influences on Child Health Outcomes 
(ECHO) Cohort." Accepted in *AJE Advances*, 2026.

# Enviromental Requirements

This code has been evaluted on a Mac, but has not been executed in a 
windows system. 

This code has been tested with python 3.10 and 3.11. The enviroment file 
gives the full requirements. If the code is run in a Jupyer notebook, 
we recommend using the 
[Jupyter dask extension](https://github.com/dask/dask-labextension) 
to track processing.

Additionally, the code requries LaTeX to be installed to render figures 
appropriately. A system installation with the multirow package installed is 
appropriate. Please refer to 
[Text rendering with LatTex](https://matplotlib.org/3.9.3/users/explain/text/usetex.html) 
in the Matplotlib documentation.

From the current folder, you can build this enviroment using

```{bash}
conda env create --name thesus --file enviroment.txt
pip install pip install dask-labextension
pip install .
```


# Execution

The simulations are run throught the `Simulations.ipynb` notebook found in 
the `ipynb` folder. Install and activate the appropriate conda enviroment, 
and the notebook should run. Please note that due to the size of the 
simulations, this may be slow. Code specific to individaul steps (e.g. 
wrappers for modeling functions) can be found in the `thesus` folder.

# Funding

Research reported in this publication was supported by the Environmental 
influences on Child Health Outcomes (ECHO) Program, Office of the Director, 
National Institutes of Health, under Award Number U24OD023382 (Data Analysis C
enter). All analyses and code are the responsibiliity of the study 
statisticians, independently of the sponsor. 
