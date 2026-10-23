"""Simple Keras regression on synthetic data (fake_reg.csv).

Usage:
    python src/simple_regression.py
    python src/simple_regression.py --data data/fake_reg.csv --epochs 500
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/fake_reg.csv")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--model-out", default="models/my_model.keras")
    return parser.parse_args()


def build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(4, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop", loss="mse")
    return model


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)

    X = df[["feature1", "feature2"]].values
    y = df["price"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model()
    model.fit(X_train, y_train, epochs=args.epochs, verbose=0)

    predictions = model.predict(X_test).reshape(-1)
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

    model.save(args.model_out)
    reloaded = load_model(args.model_out)
    print("Reloaded model prediction sample:", reloaded.predict(X_test[:1]))


if __name__ == "__main__":
    main()
