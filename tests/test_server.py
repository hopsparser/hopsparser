import pathlib

from pytest_assert_utils import util

from fastapi.testclient import TestClient


def test_model_availability(api_client: TestClient):
    response = api_client.get("/models")
    assert response.status_code == 200
    assert response.json() == {
        "models": {"default": ["tagger", "parser"]},
        "default_model": util.Any(),
    }


def process(api_client: TestClient, raw_text: pathlib.Path):
    response = api_client.post("/process", data={"data": raw_text.read_text()})
    assert response.status_code == 200
    assert response.json() == {
        "model": util.Any(),
        "acknowledgements": util.Any(),
        "data": util.Any()
    }
