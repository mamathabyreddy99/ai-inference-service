import os

import pytest
from fastapi.testclient import TestClient

from app.training import train_and_save


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    model_path = d / "model.joblib"
    train_and_save(str(model_path), verbose=False)
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["DB_PATH"] = str(d / "test.db")
    from app.main import app
    with TestClient(app) as c:
        yield c
