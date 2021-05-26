import sqlite3
from pathlib import Path
from typing import List

from certifia.utils.metaclass import Singleton


class DataBase(metaclass=Singleton):
    batch1_path: Path = Path(__file__).parent.parent.parent / 'data' / 'batch_1.db'
    batch2_path: Path = Path(__file__).parent.parent.parent / 'data' / 'batch_2.db'
    test_path: Path = Path(__file__).parent.parent.parent / 'data' / 'test.db'
    name: str = ''
    conn =  None
    tables: List[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Batch1DB(DataBase):

    def __init__(self):
        self.conn = None
        self.name = 'batch1'
        self.tables = ['vols', 'aeroports', 'compagnies', 'prix_fuel']


    def __enter__(self):
        self.conn = sqlite3.connect(str(self.batch1_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()


class Batch2DB(DataBase):

    def __init__(self):
        self.conn = None
        self.name = 'batch2'
        self.tables = ['vols', 'aeroports', 'compagnies', 'prix_fuel']

    def __enter__(self):
        self.conn = sqlite3.connect(str(self.batch2_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()


class TestDB(DataBase):

    def __init__(self):
        self.conn = None
        self.name = 'test'
        self.tables = ['vols']

    def __enter__(self):
        self.conn = sqlite3.connect(str(self.test_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
