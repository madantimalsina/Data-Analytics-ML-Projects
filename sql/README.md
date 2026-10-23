# sql

SQL files used for data selection/preview steps before model training. Each script
in [`../src`](../src/README.md) that supports `--sql` loads the relevant CSV(s) into an
in-memory SQLite database (table name = CSV filename without extension) and runs the query.

| File | Description |
|---|---|
| [ca_housing_top_price.sql](ca_housing_top_price.sql) | Top 20 districts by median house value in `CA_housing.csv`. |
| [kc_house_price_by_year.sql](kc_house_price_by_year.sql) | Average sale price and number of sales per year in `kc_house_data.csv`. |
| [cancer_class_balance.sql](cancer_class_balance.sql) | Class balance (benign vs. malignant) in `cancer_classification.csv`. |

## Usage

```bash
python src/keras/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql
python src/pytorch/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql
```
