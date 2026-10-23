"""Keras regression for house price prediction (kc_house_data.csv).

Note: data/CA_housing.csv uses a different schema (no "price"/"date"/"zipcode"
columns) and is not compatible with this script; see data/README.md.

Usage:
    python src/keras/house_price_regression.py --data data/kc_house_data.csv
    python src/keras/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from db_utils import query_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/kc_house_data.csv")
    parser.add_argument("--sql", help="optional .sql file to preview a data selection before training")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df = df.drop("date", axis=1)
    if "zipcode" in df.columns:
        df = df.drop("zipcode", axis=1)
    return df


def build_model(n_features: int) -> Sequential:
    model = Sequential()
    for _ in range(4):
        model.add(Dense(n_features, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def main() -> None:
    args = parse_args()

    if args.sql:
        preview = query_csv(args.sql, [args.data])
        print(f"--- preview from {args.sql} ---")
        print(preview)
        print()

    df = load_data(args.data)
    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1])
    model.fit(
        x=X_train,
        y=y_train.values,
        validation_data=(X_test, y_test.values),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=0,
    )

    predictions = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("Explained variance:", explained_variance_score(y_test, predictions))


if __name__ == "__main__":
    main()
