import pandas as pd

from app import ROOT_PATH
from app.data_engineering.data_access import read_db
from app.data_engineering.feature_engineering import FeatureEngineering
from app.model import Model
from app.utils.logger import Logger

logger = Logger().logger


def main():
    logger.info("Reading database...")
    datasets = read_db()
    logger.info("Database read")

    vols_df = pd.concat([datasets["batch1"]["vols"], datasets["batch2"]["vols"]])

    logger.info("Feature engineering...")
    feature_engineering = FeatureEngineering(
        training_columns=[
            "AEROPORT DEPART",
            "AEROPORT ARRIVEE",
            "DATE",
            "MOIS",
            "SEMAINE",
            "IDENTIFIANT",
        ],
        columns_to_dummify=["AEROPORT DEPART", "AEROPORT ARRIVEE"],
    )
    X, y = feature_engineering.fit(vols_df)
    X = X.drop(columns=["DATE", "IDENTIFIANT"])

    logger.info("Training...")
    model = Model().fit(X, y)
    logger.info("model trained")
    # TODO: add date in model
    feature_engineering.save_feature_engineering(
        path=(ROOT_PATH / "data" / "output" / "feature_engineering.pkl").resolve()
    )
    logger.info("Feature Engineering saved.")
    model.save_model(path=(ROOT_PATH / "models" / "rf_model.pkl").resolve())
    logger.info("Model saved.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
