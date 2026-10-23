# src/keras

Keras/TensorFlow implementations, converted directly from the [notebooks](../../notebooks/README.md).

| File | Description |
|---|---|
| [db_utils.py](db_utils.py) | Shared helper: loads CSVs into an in-memory SQLite database and runs a `.sql` file against it. Used by the other scripts' `--sql` option. |
| [simple_regression.py](simple_regression.py) | Trains a small regression model on synthetic data (`../../data/fake_reg.csv`). |
| [house_price_regression.py](house_price_regression.py) | Trains a regression model to predict house price (works with `../../data/kc_house_data.csv` or `../../data/CA_housing.csv`). |
| [cancer_classification.py](cancer_classification.py) | Trains a binary classifier on the breast cancer dataset (`../../data/cancer_classification.csv`). |
| [loan_default.py](loan_default.py) | Full LendingClub feature-engineering + classification pipeline predicting loan repayment. Requires the raw `lending_club_loan_two.csv` (not included; see [data README](../../data/README.md)). |

See [../README.md](../README.md) for usage examples.
