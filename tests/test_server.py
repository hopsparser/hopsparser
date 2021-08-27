import io
import json
import pathlib
from typing import Generator

import pytest
from pytest_assert_utils import util

from fastapi.testclient import TestClient

from hopsparser import deptree


@pytest.fixture
def api_client(
    model_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("models", json.dumps({"default": str(model_path)}))
    from hopsparser.server import app

    yield TestClient(app=app)


def test_model_availability(api_client: TestClient):
    response = api_client.get("/models")
    assert response.status_code == 200
    assert response.json() == {
        "models": {"default": ["tagger", "parser"]},
        "default_model": util.Any(),
    }


def test_server_processing(api_client: TestClient, raw_text: pathlib.Path):
    response = api_client.post(
        "/process", json={"data": raw_text.read_text(), "input": "horizontal"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "model": util.Any(),
        "acknowledgements": util.Any(),
        "data": util.Any(),
    }

    parsed_trees = list(
        deptree.DepGraph.read_conll(io.StringIO(response.json()["data"]))
    )
    assert len(parsed_trees) == len(raw_text.read_text().splitlines())
