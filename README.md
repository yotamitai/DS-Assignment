# Nvidia Data Science Assignment

## Overview
This repository contains the code and documentation for the Nvidia Data Science Assignment. The objective of this assignment is to develop an end-to-end solution to predict the outcome of the costly test 'TLJYWBE' using historical testing data from a production line.


## Directory Structure

Here is the directory structure of the project:
### Data

The `data` directory contains the raw and processed datasets used in this assignment. It is structured as follows:

- `raw`: Contains the raw, unprocessed data.
- `processed`: Contains the processed data ready for model evaluation.

### Notebooks

The `notebooks` directory contains Jupyter notebooks used for the assignment.

- `01_EDA.ipynb`: Exploratory data analysis.
- `02_Data_Preperation.ipynb`: The process of data preparation & feature selection.
- `03_Model_Development.ipynb`: Model development, comparison, selection and training.
- `04_Final_Analysis.ipynb`: The final loading and testing of the model and it's results.


### Source Code

The `src` directory contains all the code for splitting and processing the raw data. It includes:

- `split_data.py`: splitis the original raw data into train and test files for easier loading and usage.
- `process.py`: processes the raw data outputting a dataset that is ready for training for production.

### model

The `modle` directory contains all the code for splitting and processing the raw data. It includes:

- `finalized_model.json`: the assignment final model. 



## Requirements

```bash
pip install -r requirements.txt
```


## Usage
- place the `home_assignment.feather` file in /data/raw 
- run the `split_data.py` to create the `train_data.feather` and `test_data.feather` files
- Explore the EDA and Data Preparation notebooks
- To process the data run `process.py` on the desired raw data to obtain the processed version of it
- Now you can utilize the processed version in the Model_Development and Final_Analysis notebooks


