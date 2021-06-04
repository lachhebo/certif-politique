import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

from app.utils.logger import Logger


class Model:
    def __init__(self):
        self.rf_regressor = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=0, n_jobs=-1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        train a random forest regressor with
        X being the training columns and
        y the label to predict
        """
        self.rf_regressor.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.rf_regressor.predict(X)

    def eval_model(self, X: pd.DataFrame, y) -> None:
        y_pred = self.predict(X)
        logger = Logger().logger
        logger.info(f"Mean Absolute Error: {metrics.mean_absolute_error(y, y_pred)}")
        logger.info(f"Mean Squared Error: {metrics.mean_squared_error(y, y_pred)}")
        logger.info(
            f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y, y_pred))}"
        )
        logger.info(f"R2 eval_model: {metrics.r2_score(y, y_pred)}")

    def save_model(self, path=None) -> None:
        """
        Save to file in the current working directory
        """
        if path is None:
            path = "../models/rf_model.pkl"
        with open(path, "wb") as file:
            pickle.dump(self.rf_regressor, file)

    def load_model(self, path=None):
        """
        Load to file in the current working directory
        """
        if path is None:
            path = "../models/rf_model.pkl"
        with open(path, "rb") as file:
            pickle_model = pickle.load(file)
            self.rf_regressor = pickle_model
        return self