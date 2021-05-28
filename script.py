import pandas as pd
from certifia.data_engineering.data_access import read_db
from certifia.feature_engineering import FeatureEngineering
from certifia.training import Training
from certifia.utils.logger import Logger


def main():
    logger = Logger().logger
    logger.info("Reading database...")
    datasets = read_db()
    logger.info("Db read.")

    vols_df = pd.concat([datasets['batch1']['vols'], datasets['batch2']['vols']])

    logger.info("Feature engineering...")
    X, y = FeatureEngineering(
        training_columns=['AEROPORT DEPART',
                          'AEROPORT ARRIVEE',
                          'NOMBRE DE PASSAGERS'],
        columns_to_dummify=['AEROPORT DEPART', 'AEROPORT ARRIVEE']
    ).transform(vols_df)

    logger.info("Training...")
    model = Training().fit(X, y)
    logger.info("Model trained.")
    model.save_model(path="models/rf_model.pkl")
    logger.info("Model saved.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
