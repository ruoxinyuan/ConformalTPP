"""Conformal Prediction."""

from abc import ABC, abstractmethod
from typing import Self, Type

import numpy as np
from numpy.typing import ArrayLike, NDArray


class QuantileRegressor(ABC):
    """Quantile Regressor Abstract Base Class."""

    def __init__(
        self,
        alpha: float,
        seed: int | None = None,
    ) -> None:
        """Initialize model with desired miscoverage level."""
        self.alpha = alpha
        self.seed = seed

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Self:
        """Train model to estimate quantiles."""

    @abstractmethod
    def predict(
        self,
        X: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """Return lower and upper predictions."""

    @staticmethod
    def _monotonize_curves(
        y_pred_lower: ArrayLike,
        y_pred_upper: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """Monotonize curves from arbitrary quantile regression model.

        Swap lower and upper predictions if the former is greater than the latter
        in order to build a proper interval.
        This can be seen as a particular case of the methodology described in
        'Quantile and Probability Curves without Crossing, 2010'.
        """
        if (y_pred_lower > y_pred_upper).any():
            y_preds = np.array([y_pred_lower, y_pred_upper]).T
            y_preds = np.sort(y_preds, axis=1)
            y_pred_lower, y_pred_upper = y_preds[:, 0], y_preds[:, 1]
        return y_pred_lower, y_pred_upper
    


class ConformalQR(QuantileRegressor):
    """Conformal Quantile Regressor."""

    def __init__(
        self,
        Model,
        alpha: float,
        seed: int | None = None,
        model_kwargs: dict | None = None,
    ) -> None:
        """Initialize CQR with desired base model and miscoverage level."""
        super().__init__(alpha, seed)
        self.Model = Model
        self.model_kwargs = model_kwargs or {}

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Self:
        """Fit base model."""
        self.model = self.Model(seed=self.seed, **self.model_kwargs).fit(X, y)
        return self

    def calibrate(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Self:
        """Calibrate via prediction error."""
        self.scores = np.abs(y - self.model.predict(X))
        self.q_hat = np.quantile(self.scores, 1 - self.alpha)
        return self

    def predict(
        self,
        X: ArrayLike,
        # y: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """Return lower and upper predictions."""

        y_pred = self.model.predict(X)

        # Conformalize prediction set
        y_pred_lower = y_pred - self.q_hat
        y_pred_upper = y_pred + self.q_hat
        y_pred_lower, y_pred_upper = self._monotonize_curves(y_pred_lower, y_pred_upper)
        
        # Optional: Ensure lower bound is non-negative (can be adjusted based on use case)
        y_pred_lower, y_pred_upper = np.maximum(0, y_pred_lower), np.maximum(0, y_pred_upper)

        # # Calculate prediction interval length
        # interval_lengths = y_pred_upper - y_pred_lower

        # # If true labels are provided, calculate coverage rate
        # if y is not None:
        #     coverage = np.mean((y >= y_pred_lower) & (y <= y_pred_upper))
        #     print(f"Coverage rate: {coverage}")

        # # Statistics for the prediction intervals
        # mean_length = np.mean(interval_lengths)
        # median_length = np.median(interval_lengths)

        # print(f"Mean interval length: {mean_length}")
        # print(f"Median interval length: {median_length}")
            
        return y_pred_lower, y_pred_upper
