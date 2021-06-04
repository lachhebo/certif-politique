import pandas as pd

from app.data_engineering.data_access import read_db
from app.data_engineering.feature_engineering import FeatureEngineering
from app.model import Model
from app.utils.logger import Logger
from app.data_engineering.data_cleaning import DataCleaning

logger = Logger().logger


def main():
    logger = Logger().logger
    logger.info("Reading database...")
    datasets = read_db()
    logger.info("Db read.")

    vols_df = pd.concat([datasets['batch1']['vols'], datasets['batch2']['vols']]).head(10000)

    FEATURES = datasets['test']['vols'].columns.tolist()
    label = "RETARD A L'ARRIVEE"
    logger.info(f"Label to predict is {label}.")

    logger.info("Data cleaning...")
    cleaning = DataCleaning(features_columns=FEATURES, label=label)
    cleaned_vols_df = cleaning.cleaning(vols_df)
    logger.info("OK")

    logger.info("Split features from label.")
    X = cleaned_vols_df[FEATURES]
    y = cleaned_vols_df[label]  # .apply(lambda x: 1 if x>0 else 0)

    logger.info("Feature engineering...")
    feature_engineering = FeatureEngineering(
        training_columns=[
            'AEROPORT DEPART',
            'AEROPORT ARRIVEE',
            'TEMPS DE DEPLACEMENT A TERRE AU DECOLLAGE',
            "TEMPS DE DEPLACEMENT A TERRE A L'ATTERRISSAGE",
            'TEMPS PROGRAMME',
            'DISTANCE',
            'COMPAGNIE AERIENNE',
            'NOMBRE DE PASSAGERS',
            'MOIS',
            'SEMAINE',
            'HEURE DEPART PROGRAMME',
            'HEURE ARRIVEE PROGRAMMEE'
        ],
        columns_to_dummify=['AEROPORT DEPART', 'AEROPORT ARRIVEE', 'COMPAGNIE AERIENNE'],
    )
    X = feature_engineering.fit(X)
    logger.info("OK")

    logger.info("Training...")
    model = Model().fit(X, y)
    logger.info("Model trained.")

    # TODO: add date in model
    feature_engineering.save_feature_engineering(path="data/output/feature_engineering.pkl")
    logger.info("Model saved.")

    logger.info("Feature Engineering saved.")
    model.save_model(path="models/rf_model.pkl")
    logger.info("Model saved.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
