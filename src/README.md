# src

Standalone Python scripts converted from the [notebooks](../notebooks/README.md), runnable from the
command line. Several accept an optional `--sql` flag pointing at a file in [`../sql`](../sql/README.md)
to preview a data selection (via an in-memory SQLite database) before training.

| File | Description |
|---|---|
| [db_utils.py](db_utils.py) | Shared helper: loads CSVs into an in-memory SQLite database and runs a `.sql` file against it. Used by the other scripts' `--sql` option. |
| [simple_regression.py](simple_regression.py) | Trains a small Keras regression model on synthetic data (`../data/fake_reg.csv`). |
| [house_price_regression.py](house_price_regression.py) | Trains a Keras regression model to predict house price (works with `../data/kc_house_data.csv` or `../data/CA_housing.csv`). |
| [cancer_classification.py](cancer_classification.py) | Trains a Keras binary classifier on the breast cancer dataset (`../data/cancer_classification.csv`). |
| [loan_default.py](loan_default.py) | Full LendingClub feature-engineering + Keras classification pipeline predicting loan repayment. Requires the raw `lending_club_loan_two.csv` (not included; see [data README](../data/README.md)). |

## Usage

From the repo root, with the virtualenv active:

```bash
python src/simple_regression.py
python src/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql
python src/cancer_classification.py --sql sql/cancer_class_balance.sql
python src/loan_default.py --data data/lending_club_loan_two.csv
```

Run `python src/<script>.py --help` for all options.
