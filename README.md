# Keras/TensorFlow Project Collection

A collection of Keras/TensorFlow machine learning projects (classification and regression),
originally developed as Jupyter notebooks and since reorganized into a runnable Python project.

## Layout

| Folder | Contents |
|---|---|
| [notebooks/](notebooks/README.md) | Original exploratory Jupyter notebooks. |
| [src/](src/README.md) | Standalone Python scripts converted from the notebooks, runnable from the command line. |
| [sql/](sql/README.md) | SQL files used for data-selection/preview steps, run against an in-memory SQLite database built from the CSVs. |
| [data/](data/README.md) | CSV datasets used by the notebooks and scripts. |
| [models/](models/README.md) | Saved/trained model files. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running a notebook

```bash
source .venv/bin/activate
jupyter notebook
```

Then open any file in [notebooks/](notebooks/README.md).

## Running a script

```bash
source .venv/bin/activate
python src/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql
```

See [src/README.md](src/README.md) for all available scripts and options.
