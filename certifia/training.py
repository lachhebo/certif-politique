from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle
import numpy as np

from certifia.utils.logger import Logger


class Training:
    def __init__(self):
        self.rf_regressor = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=0, n_jobs=-1)

    def fit(self, X, y):
        """
        train a random forest regressor with
        X being the training columns and
        y the label to predict
        """
        self.rf_regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.rf_regressor.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        logger = Logger().logger
        logger.info(f'Mean Absolute Error: {metrics.mean_absolute_error(y, y_pred)}')
        logger.info(f'Mean Squared Error: {metrics.mean_squared_error(y, y_pred)}')
        logger.info(f'Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y, y_pred))}')
        logger.info(f'R2 score: {metrics.r2_score(y, y_pred)}')

    def save_model(self, path=None):
        """
        Save to file in the current working directory
        """
        if path is None:
            path = "../models/rf_model.pkl"
        with open(path, 'wb') as file:
            pickle.dump(self.rf_regressor, file)
