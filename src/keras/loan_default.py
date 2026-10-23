"""Keras binary classification predicting loan repayment (LendingClub data).

Note: the raw lending_club_loan_two.csv is not included in this repo (too
large / not committed). Point --data at your own copy to run this end-to-end.
lending_club_info.csv (the column glossary) is included under data/.

Usage:
    python src/keras/loan_default.py --data data/lending_club_loan_two.csv
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from db_utils import query_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/lending_club_loan_two.csv")
    parser.add_argument("--info", default="data/lending_club_info.csv")
    parser.add_argument("--sql", help="optional .sql file to preview a data selection before training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--model-out", default="models/full_data_project_model.keras")
    return parser.parse_args()


def feat_info(data_info: pd.DataFrame, col_name: str) -> None:
    print(data_info.loc[col_name]["Description"])


def fill_mort_acc(total_acc_avg: pd.Series, total_acc: float, mort_acc: float) -> float:
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    return mort_acc


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_repaid"] = df["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})

    df = df.drop("emp_title", axis=1)
    df = df.drop("emp_length", axis=1)
    df = df.drop("title", axis=1)

    total_acc_avg = df.groupby("total_acc").mean(numeric_only=True)["mort_acc"]
    df["mort_acc"] = df.apply(
        lambda row: fill_mort_acc(total_acc_avg, row["total_acc"], row["mort_acc"]), axis=1
    )
    df = df.dropna()

    df["term"] = df["term"].apply(lambda term: int(term[:3]))
    df = df.drop("grade", axis=1)

    dummies_subgrade = pd.get_dummies(df["sub_grade"], drop_first=True)
    df = pd.concat([df.drop("sub_grade", axis=1), dummies_subgrade], axis=1)

    dummies = pd.get_dummies(
        df[["verification_status", "application_type", "initial_list_status", "purpose"]],
        drop_first=True,
    )
    df = df.drop(["verification_status", "application_type", "initial_list_status", "purpose"], axis=1)
    df = pd.concat([df, dummies], axis=1)

    df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")
    dummies = pd.get_dummies(df["home_ownership"], drop_first=True)
    df = df.drop("home_ownership", axis=1)
    df = pd.concat([df, dummies], axis=1)

    df["zip_code"] = df["address"].apply(lambda address: address[-5:])
    dummies = pd.get_dummies(df["zip_code"], drop_first=True)
    df = df.drop(["zip_code", "address"], axis=1)
    df = pd.concat([df, dummies], axis=1)

    df = df.drop("issue_d", axis=1)

    df["earliest_cr_year"] = df["earliest_cr_line"].apply(lambda date: int(date[-4:]))
    df = df.drop("earliest_cr_line", axis=1)

    df = df.drop("loan_status", axis=1)
    return df


def build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(68, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(34, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(17, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def main() -> None:
    args = parse_args()

    data_info = pd.read_csv(args.info, index_col="LoanStatNew")

    if args.sql:
        preview = query_csv(args.sql, [args.data])
        print(f"--- preview from {args.sql} ---")
        print(preview)
        print()

    df = pd.read_csv(args.data)
    df = clean_data(df)

    X = df.drop("loan_repaid", axis=1).values
    y = df["loan_repaid"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model()
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=args.patience)
    model.fit(
        x=X_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[early_stop],
    )

    raw_predictions = model.predict(X_test)
    predictions = (raw_predictions > 0.5).astype(int)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    model.save(args.model_out)

    random.seed(100)
    random_ind = random.randint(0, len(df) - 1)
    customer = df.drop("loan_repaid", axis=1).iloc[random_ind]
    customer_scaled = scaler.transform(customer.values.reshape(1, -1))
    reloaded = load_model(args.model_out)
    prediction = (reloaded.predict(customer_scaled) > 0.5).astype(int)
    print("Sample customer prediction:", prediction, "actual:", df.iloc[random_ind]["loan_repaid"])


if __name__ == "__main__":
    main()
