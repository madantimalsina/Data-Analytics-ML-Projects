# Notebooks

Original exploratory Jupyter notebooks. Each one corresponds to a script of the
same purpose under [`../src`](../src/README.md).

| File | Description |
|---|---|
| [CA_housing.ipynb](CA_housing.ipynb) | Exploratory data analysis and a Keras regression model on California housing data (`../data/CA_housing.csv`). |
| [KerasTF_SimpleRegression.ipynb](KerasTF_SimpleRegression.ipynb) | EDA and Keras regression model predicting house sale price on King County housing data (`../data/kc_house_data.csv`). |
| [KerasTF_BreastCancer_Classification.ipynb](KerasTF_BreastCancer_Classification.ipynb) | Binary classification (benign vs. malignant) on the breast cancer dataset, including dropout and early stopping experiments (`../data/cancer_classification.csv`). |
| [KerasTF_LoadDefaultProject.ipynb](KerasTF_LoadDefaultProject.ipynb) | Full LendingClub loan-default pipeline: feature engineering, one-hot encoding, and a Keras classifier predicting loan repayment. |
| [Simple_KerasTF_project.ipynb](Simple_KerasTF_project.ipynb) | Introductory Keras regression exercise on synthetic data (`../data/fake_reg.csv`). |

To run these, activate the project virtualenv (see the [top-level README](../README.md)) and launch `jupyter notebook` from the repo root.
