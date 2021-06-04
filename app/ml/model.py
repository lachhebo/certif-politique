import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

from app import ROOT_PATH
from app.utils.logger import Logger

logger = Logger()


class Model:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=0, n_jobs=-1
        )
        self.metrics = {}

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """
        train a random forest regressor with
        X being the training columns and
        y the label to predict
        """
        self.model.fit(x, y)
        self.metrics = {}
        return self

    def predict(self, x: pd.DataFrame):
        return self.model.predict(x)

    def compute_metrics(self, x: pd.DataFrame, y) -> None:
        y_pred = self.predict(x)

        mean_absolute_err = metrics.mean_absolute_error(y, y_pred)
        mean_square_err = metrics.mean_squared_error(y, y_pred)
        root_mean_squared_err = np.sqrt(mean_square_err)
        r2_score = metrics.r2_score(y, y_pred)

        self.metrics = {
            "mean_absolute_err": mean_absolute_err,
            "mean_square_err": mean_square_err,
            "root_mean_squared_err": root_mean_squared_err,
            "r2_score": r2_score,
        }

        logger.info(f"Mean Absolute Error: {mean_absolute_err}")
        logger.info(f"Mean Squared Error: {mean_square_err}")
        logger.info(f"Root Mean Squared Error: {root_mean_squared_err}")
        logger.info(f"R2 compute_metrics: {r2_score}")

    def save_model(self, path=None) -> None:
        """
        Save to file in the current working directory
        """
        if path is None:
            path = (ROOT_PATH / "models" / "rf_model.pkl").resolve()
        with open(path, "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, path=None):
        """
        Load to file in the current working directory
        """
        if path is None:
            path = (ROOT_PATH / "models" / "rf_model.pkl").resolve()
        with open(path, "rb") as file:
            pickle_model = pickle.load(file)
            self.model = pickle_model  # TODO ?
        return self
