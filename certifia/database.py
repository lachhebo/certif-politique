import os
import sqlite3
from pathlib import Path
from typing import Union

from certifia.utils.logger import Logger
from certifia.utils.metaclass import Singleton


class Session(metaclass=Singleton):

    def __init__(self):
        debug: bool = 'True' == os.getenv('DEBUG', 'False')
        directory = 'test' if debug else 'production'
        self.database_path: Path = Path(__file__).parent.parent / 'database' / directory / 'database.db'
        self.conn = None

    def get_user_password_by_email(self, email) -> Union[None, str]:
        with self as _:
            cursor = self.conn.execute("SELECT hashed_password from user WHERE email=:searched_email",
                                       {'searched_email': email})
            first_user = cursor.fetchall()
            if len(first_user) == 0:
                return None
            password_user = first_user[0]
            return password_user[0]

    def __enter__(self):
        print(str(self.database_path))
        self.conn = sqlite3.connect(str(self.database_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()


def get_db():
    db = Session()
    try:
        yield db
    finally:
        Logger().info('conn is closed')
