import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    n_cross_validation: int,
    scoring: str,
) -> BaseEstimator:
    grid_search = GridSearchCV(model, param_grid, cv=n_cross_validation, scoring=scoring)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def save_output(output_data, path_to_output_dir, filename, logger=None) -> None:
    output_dir = os.path.join(path_to_output_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if isinstance(output_data, pd.DataFrame):
        output_data.to_csv(output_path, index=False)
    elif isinstance(output_data, BaseEstimator):
        joblib.dump(output_data, output_path)
    else:
        raise ValueError("Unsupported output_data type.")

    if logger:
        logger.info(f"Output saved at: {output_path}")
