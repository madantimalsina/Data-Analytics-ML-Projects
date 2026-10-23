"""PyTorch regression for house price prediction (kc_house_data.csv / CA_housing.csv).
PyTorch port of src/keras/house_price_regression.py.

Usage:
    python src/pytorch/house_price_regression.py --data data/kc_house_data.csv
    python src/pytorch/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


class HousePriceNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, 1),
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

    df = load_data(args.data)
    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)

    model = HousePriceNet(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).numpy()

    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("Explained variance:", explained_variance_score(y_test, predictions))


if __name__ == "__main__":
    main()
