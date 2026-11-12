# Some General Links
- https://rapidapi.com/ for free API for data 

# Little about pythons 
💡To install anything with winget please use "--scope machine" with **_admin context_**. This will make the installtion machine wide and easy to manage from installation and uninstalltion perspective.
```powershell
winget install --id Python.Pyton.3.14 --scope machine
```
💡To upgrade python version from winget, please check which are the version(s) that are available
```powershell
# List all Python installations
winget list --name Python
```
and to upgrade existing installtion go ahead with specific id and version
```powershell
# Upgrade Python using winget
winget upgrade --id Python.Python.3
```
#### *Virtual Environments*
This required by doing this in mind of MLDLC or MLOPS. It provides neat segregation of your environemnt installtion needs.
>
- First run python -m venv PutYourEnvName
- then go inside PutYourEnvName\scripts\ and run activate.ps1

#### *How to run jupyter notebook from virtual environment in side VS Code*
Running a Jupyter Notebook from a virtual environment in VS Code is straightforward — you just need to ensure that:
>
- The virtual environment has Jupyter installed.(pip install jupyter --> after activating the virtual environment)
- VS Code is configured to use the correct Python interpreter (your venv). --> (Select kernel from jupyter notebook)
- The Jupyter extension for VS Code is installed. 