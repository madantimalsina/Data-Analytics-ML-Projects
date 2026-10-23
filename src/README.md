# src

Standalone Python scripts converted from the [notebooks](../notebooks/README.md), runnable from the
command line. Each project is implemented twice, in separate folders, with matching CLIs and behavior:

- [keras/](keras/README.md) — original Keras/TensorFlow implementations.
- [pytorch/](pytorch/README.md) — equivalent PyTorch implementations.

Both accept an optional `--sql` flag pointing at a file in [`../sql`](../sql/README.md) to preview a
data selection (via an in-memory SQLite database) before training.

## Usage

From the repo root, with the virtualenv active:

```bash
python src/keras/simple_regression.py
python src/pytorch/simple_regression.py

python src/keras/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql
python src/pytorch/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql

python src/keras/cancer_classification.py --sql sql/cancer_class_balance.sql
python src/pytorch/cancer_classification.py --sql sql/cancer_class_balance.sql

python src/keras/loan_default.py --data data/lending_club_loan_two.csv
python src/pytorch/loan_default.py --data data/lending_club_loan_two.csv
```

Run `python src/<framework>/<script>.py --help` for all options.
