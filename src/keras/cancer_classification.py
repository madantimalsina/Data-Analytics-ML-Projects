"""Keras binary classification on the breast cancer dataset (cancer_classification.csv).

Usage:
    python src/keras/cancer_classification.py
    python src/keras/cancer_classification.py --sql sql/cancer_class_balance.sql
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from db_utils import query_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cancer_classification.csv")
    parser.add_argument("--sql", help="optional .sql file to preview a data selection before training")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=25)
    return parser.parse_args()


def build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(units=30, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=15, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def main() -> None:
    args = parse_args()

    if args.sql:
        preview = query_csv(args.sql, [args.data])
        print(f"--- preview from {args.sql} ---")
        print(preview)
        print()

    df = pd.read_csv(args.data)
    X = df.drop("benign_0__mal_1", axis=1).values
    y = df["benign_0__mal_1"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model()
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=args.patience)
    model.fit(
        x=X_train,
        y=y_train,
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[early_stop],
    )

    predicted_probabilities = model.predict(X_test)
    predictions = (predicted_probabilities > 0.5).astype(int)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
