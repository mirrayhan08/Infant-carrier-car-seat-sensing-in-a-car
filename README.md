# Infant-carrier-car-seat-sensing-in-a-car
## Introduction
This project is to determine weither a vahicle has baby seat installed and baby is on it or not.
We have gathered 32,000 ultra-sound sensor signal to predict the outcome. Dataset has 16384 timesteps in each signal. Also added
distance of the baby and seat as `measured_distance` column, one meta-data value which differ in each case in column `metadata_value`
and a `label` column where values are 1 when installed and baby on seat properly, 0 otherwise.

### Setting up environment.
- Install python 3.9.18
- Create virtual environment. `python -m venv PATH-AND-NAME-OF-ENVIRONMENT`
- Install packages from requirements.txt. `pip install -r PATH-AND-NAME-OF-REQUIREMENTS.TXT`

### Train model
- Activate virtual environment. `PATH-AND-NAME-OF-ENVIRONMENT\scripts\activate`
- Edit preferred PREFERRED_model.py file for dataset filepath. `file ="dataset.csv file path here"`
- Run preferred model.py file. In terminal when virtual environment is active, `python PREFERRED_model.py`
- We can see the output predictions in the terminal.

