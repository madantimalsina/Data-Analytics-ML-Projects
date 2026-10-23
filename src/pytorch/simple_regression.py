"""Simple PyTorch regression on synthetic data (fake_reg.csv). PyTorch port of
src/keras/simple_regression.py.

Usage:
    python src/pytorch/simple_regression.py
    python src/pytorch/simple_regression.py --data data/fake_reg.csv --epochs 500
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/fake_reg.csv")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--model-out", default="models/my_model.pt")
    return parser.parse_args()


class RegressionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)

    X = df[["feature1", "feature2"]].values
    y = df["price"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = RegressionNet()
    optimizer = torch.optim.RMSprop(model.parameters())
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = loss_fn(predictions, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).numpy().reshape(-1)

    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

    torch.save(model.state_dict(), args.model_out)
    reloaded = RegressionNet()
    reloaded.load_state_dict(torch.load(args.model_out))
    reloaded.eval()
    with torch.no_grad():
        sample = reloaded(X_test_t[:1])
    print("Reloaded model prediction sample:", sample.numpy())


if __name__ == "__main__":
    main()
