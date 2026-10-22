# data

Datasets used by the notebooks and scripts.

| File | Description |
|---|---|
| [CA_housing.csv](CA_housing.csv) | California housing district data (location, income, rooms, median house value). Used by `CA_housing.ipynb`. |
| [kc_house_data.csv](kc_house_data.csv) | King County, WA house sale records with price and property features. Used by `KerasTF_SimpleRegression.ipynb` / `house_price_regression.py`. |
| [cancer_classification.csv](cancer_classification.csv) | Breast cancer diagnostic features with benign/malignant label. Used by `KerasTF_BreastCancer_Classification.ipynb` / `cancer_classification.py`. |
| [fake_reg.csv](fake_reg.csv) | Synthetic two-feature regression dataset. Used by `Simple_KerasTF_project.ipynb` / `simple_regression.py`. |
| [lending_club_info.csv](lending_club_info.csv) | Column glossary/description for the LendingClub loan dataset. |

**Not included:** `lending_club_loan_two.csv`, the raw LendingClub loan dataset used by
`KerasTF_LoadDefaultProject.ipynb` / `loan_default.py`, is too large to commit. Supply your own
copy and pass it via `--data` to run that script end-to-end.
