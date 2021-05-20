from pathlib import Path
from unittest.mock import patch

TESTED_MODULE = 'certifia.database'


@patch(f'{TESTED_MODULE}.Path', return_value=Path('/root/api/database.py'))
def test_session__is_connecting_to_the_good_database(path, monkeypatch):
    # given
    monkeypatch.setenv('DEBUG', 'False')
    from certifia.database import Session

    # when
    sess = Session()

    # then
    assert sess.database_path == Path('/root/database/production/database.db')


@patch(f'{TESTED_MODULE}.sqlite3')
def test_get_user_password_by_email__return_user_password_if_user_found(m_sqlite, monkeypatch):
    # given
    monkeypatch.setenv('DEBUG', 'False')
    from certifia.database import Session
    sess = Session()
    m_sqlite.connect().execute().fetchall.return_value = [('fake_password1',)]

    # when
    result = sess.get_user_password_by_email('dev@debug.com')
    # then

    assert result == 'fake_password1'


@patch(f'{TESTED_MODULE}.sqlite3')
def test_get_user_password_by_email__return_none_if_user_not_found(m_sqlite, monkeypatch):
    # given
    from certifia.database import Session
    sess = Session()
    m_sqlite.connect().execute().fetchall.return_value = []

    # when
    result = sess.get_user_password_by_email('fake-email@debug.com')
    # then

    assert result is None
