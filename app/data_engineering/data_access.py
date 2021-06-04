from typing import Dict

import pandas

from app.data_engineering.database import Batch1DB, Batch2DB, TestDB, DataBase


def read_db() -> Dict:
    datasets: Dict[str, Dict] = {}
    for base in [Batch1DB(), Batch2DB(), TestDB()]:
        _convert_database_to_pd(base, datasets)

    return datasets


def _convert_database_to_pd(base: DataBase, datasets: Dict):
    datasets[base.name] = {}
    for table in base.tables:
        with base:
            df = pandas.read_sql_query(f"SELECT * from {table}", base.conn)
            datasets[base.name][table] = df
