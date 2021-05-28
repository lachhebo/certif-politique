from typing import Dict

import pandas

from certifia.data_engineering.database import Batch1DB, Batch2DB, TestDB


def read_db() -> Dict:
    datasets: Dict[str, pandas.DataFrame] = {}
    for base in [Batch1DB(), Batch2DB(), TestDB()]:
        datasets[base.name] = {}
        for table in base.tables:
            with base:
                df = pandas.read_sql_query(f'SELECT * from {table}', base.conn)
                datasets[base.name][table] = df

    return datasets
