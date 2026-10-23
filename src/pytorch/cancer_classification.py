"""PyTorch binary classification on the breast cancer dataset (cancer_classification.csv).
PyTorch port of src/keras/cancer_classification.py.

Usage:
    python src/pytorch/cancer_classification.py
    python src/pytorch/cancer_classification.py --sql sql/cancer_class_balance.sql
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from db_utils import query_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cancer_classification.csv")
    parser.add_argument("--sql", help="optional .sql file to preview a data selection before training")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=25)
    return parser.parse_args()


class CancerNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 30),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(15, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model = CancerNet(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_test_t), y_test_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    model.eval()
    with torch.no_grad():
        predicted_probabilities = model(X_test_t).numpy()
    predictions = (predicted_probabilities > 0.5).astype(int)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
