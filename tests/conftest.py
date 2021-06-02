import pytest

from certifia.main import app


@pytest.fixture
def client():
    app.config['TESTING'] = True

    with app.test_client() as client:
        with app.app_context():
            yield client
