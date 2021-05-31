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

    vols_df = pd.concat([datasets['batch1']['vols'], datasets['batch2']['vols']]).head(500000)

    logger.info("Feature engineering...")
    feature_engineering = FeatureEngineering(
        training_columns=['AEROPORT DEPART',
                          'AEROPORT ARRIVEE',
                          'DATE',
                          'MOIS',
                          'SEMAINE',
                          'IDENTIFIANT'],
        columns_to_dummify=['AEROPORT DEPART', 'AEROPORT ARRIVEE']
    )
    X, y = feature_engineering.fit(vols_df)
    X = X.drop(columns=['DATE', 'IDENTIFIANT'])

    logger.info("Training...")
    model = Training().fit(X, y)
    logger.info("Model trained.")
    # TODO: add date in model
    feature_engineering.save_feature_engineering(path="data/output/feature_engineering.pkl")
    logger.info("Feature Engineering saved.")
    model.save_model(path="models/rf_model.pkl")
    logger.info("Model saved.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
