"""Models."""

from typing import Self
import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike, NDArray
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, Dataset



class GradientBoostingQR:
    """Gradient Boosting Regressor."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the model with optional random seed."""
        self.seed = seed

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Self:
        """Train the model."""
        self.model = LGBMRegressor(
            boosting_type="gbdt",
            deterministic=True,
            force_row_wise=True,
            seed=self.seed,
        ).fit(X, y)

        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """Return predictions."""
        return self.model.predict(X)


# class KNNQR(QuantileRegressor):
#     """k-Nearest Neighbors Quantile Regressor."""

#     def fit(
#         self,
#         X: ArrayLike,
#         y: ArrayLike,
#     ) -> Self:
#         """Train quantile regression based on k-nearest neighbors."""
#         self.qr = KNeighborsQuantileRegressor(
#             q=(self.alpha / 2, 1 - self.alpha / 2),
#         ).fit(X, y)

#         return self

#     def predict(
#         self,
#         X: ArrayLike,
#     ) -> tuple[NDArray, NDArray]:
#         """Return lower and upper predictions."""
#         y_pred = self.qr.predict(X)

#         y_pred_lower = y_pred[0]
#         y_pred_upper = y_pred[1]
#         y_pred_lower, y_pred_upper = self._monotonize_curves(y_pred_lower, y_pred_upper)

#         return y_pred_lower, y_pred_upper


class LinearQR:
    """Linear Regressor."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the model with optional random seed."""
        self.seed = seed

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Self:
        """Train the model."""
        self.model = linear_model.LinearRegression().fit(X, y)

        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """Return predictions."""
        return self.model.predict(X)


class RandomForestQR:
    """Random Forest Regressor."""
    
    def __init__(
        self,
        seed: int | None = None,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
    ) -> None:
        """Initialize the model with optional random seed."""
        self.seed = seed
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Self:
        """Train the Regression Forest."""
        self.model = RandomForestRegressor(
            random_state=self.seed, 
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            ).fit(X, y)

        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """Return predictions."""
        return self.model.predict(X)


class mlpQR:
    """Neural Network: MLP."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        seed: int | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 64,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def _train_loop(
        self,
        dataloader: DataLoader,
        model: "mlp",
        loss_fn: nn.MSELoss,
        optimizer: torch.optim.AdamW,
    ) -> None:
        """Train a single epoch."""
        model.train()
        for X, y in dataloader:
            X = X.to(mlpQR.device)
            y = y.to(mlpQR.device)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    def fit(
        self,
        X: ArrayLike | torch.Tensor,
        y: ArrayLike | torch.Tensor,
    ) -> Self:
        """Train the MLP model."""
        self.model = self.mlp(
            input_size=X.shape[1],
        ).to(mlpQR.device)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = self.CustomDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for _ in range(self.epochs):
            self._train_loop(dataloader, self.model, loss_fn, optimizer)

        return self

    def predict(
        self,
        X: ArrayLike | torch.Tensor,
    ) -> NDArray:
        """Return predictions."""
        X = torch.as_tensor(X).float().to(mlpQR.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.cpu().numpy().squeeze()

    class mlp(nn.Module):
        """Standard neural network."""

        def __init__(self, input_size: int) -> None:
            """Initialize model with input and output sizes and architecture."""
            super().__init__()
            self.input_size = input_size

            self.mlp = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            x = self.mlp(x)
            return x

    class CustomDataset(Dataset):
        """Custom dataset class."""

        def __init__(
            self,
            values: ArrayLike | torch.Tensor,
            labels: ArrayLike | torch.Tensor,
        ):
            """Initialize class by converting data to appropriate tensors."""
            super().__init__()
            self.values = torch.as_tensor(values).float().to(mlpQR.device)
            self.labels = torch.as_tensor(labels.reshape(-1, 1)).float().to(mlpQR.device)

        def __len__(self) -> int:
            """Length."""
            return len(self.labels)

        def __getitem__(
            self,
            index: ArrayLike,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Get item."""
            return self.values[index], self.labels[index]


class lstmQR:
    """Neural Network: LSTM."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        seed: int | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def _train_loop(
        self,
        dataloader: DataLoader,
        model: "lstm",
        loss_fn: nn.MSELoss,
        optimizer: torch.optim.AdamW,
    ) -> None:
        """Train a single epoch."""
        model.train()
        for X, y in dataloader:
            X = X.to(lstmQR.device)
            y = y.to(lstmQR.device)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    def fit(
        self,
        X: ArrayLike | torch.Tensor,
        y: ArrayLike | torch.Tensor,
    ) -> Self:
        """Train the LSTM model."""
        self.model = self.lstm(
            input_size=X.shape[1],  # X.shape should be (batch, features, seq_len)
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(lstmQR.device)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = self.CustomDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for _ in range(self.epochs):
            self._train_loop(dataloader, self.model, loss_fn, optimizer)

        return self

    def predict(
        self,
        X: ArrayLike | torch.Tensor,
    ) -> NDArray:
        """Return predictions."""
        X = torch.as_tensor(X).float().to(lstmQR.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.cpu().numpy().squeeze()

    class lstm(nn.Module):
        """LSTM-based neural network."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 1) # change x.shape to (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out

    class CustomDataset(Dataset):
        """Custom dataset class for LSTM."""

        def __init__(
            self,
            values: ArrayLike | torch.Tensor,
            labels: ArrayLike | torch.Tensor,
        ):
            super().__init__()
            self.values = torch.as_tensor(values).float().to(lstmQR.device)
            self.labels = torch.as_tensor(labels.reshape(-1, 1)).float().to(lstmQR.device)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(
            self,
            index: ArrayLike,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.values[index], self.labels[index]

class TransformerQR:
    """Neural Network: Transformer."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        seed: int | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def _train_loop(
        self,
        dataloader: DataLoader,
        model: "transformer",
        loss_fn: nn.MSELoss,
        optimizer: torch.optim.AdamW,
    ) -> None:
        """Train a single epoch."""
        model.train()
        for X, y in dataloader:
            X = X.to(TransformerQR.device)
            y = y.to(TransformerQR.device)

            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    def fit(
        self,
        X: ArrayLike | torch.Tensor,
        y: ArrayLike | torch.Tensor,
    ) -> Self:
        """Train the Transformer model."""
        self.model = self.transformer(
            input_size=X.shape[1],  # (batch, features, seq_len)
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nhead=self.nhead,
            dropout=self.dropout,
        ).to(TransformerQR.device)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = self.CustomDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for _ in range(self.epochs):
            self._train_loop(dataloader, self.model, loss_fn, optimizer)

        return self

    def predict(
        self,
        X: ArrayLike | torch.Tensor,
    ) -> NDArray:
        """Return predictions."""
        X = torch.as_tensor(X).float().to(TransformerQR.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.cpu().numpy().squeeze()

    class transformer(nn.Module):
        """Transformer-based neural network."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            nhead: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.nhead = nhead
            self.dropout = dropout

            # Positional encoding
            self.pos_encoder = nn.Embedding(1024, hidden_size)  # Max seq_len=1024
            
            # Transformer layers
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dropout=dropout,
                dim_feedforward=hidden_size*4,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layers,
                num_layers=num_layers
            )
            
            # Input projection
            self.input_fc = nn.Linear(input_size, hidden_size)
            self.output_fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 1)  # (batch, seq_len, features)
            
            # Input projection
            x = self.input_fc(x)  # (batch, seq_len, hidden_size)
            
            # Generate positional indices
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = x + self.pos_encoder(positions)
            
            # Transformer processing
            x = self.dropout(x)
            x = self.transformer_encoder(x)  # (batch, seq_len, hidden_size)
            
            # Get last timestep output
            out = self.output_fc(x[:, -1, :])  # (batch, 1)
            return out

    class CustomDataset(Dataset):
        """Custom dataset class for Transformer."""

        def __init__(
            self,
            values: ArrayLike | torch.Tensor,
            labels: ArrayLike | torch.Tensor,
        ):
            super().__init__()
            self.values = torch.as_tensor(values).float().to(TransformerQR.device)
            self.labels = torch.as_tensor(labels.reshape(-1, 1)).float().to(TransformerQR.device)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(
            self,
            index: ArrayLike,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.values[index], self.labels[index]