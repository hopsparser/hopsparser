import io
import pathlib

from pytest_assert_utils import util

from fastapi.testclient import TestClient

from hopsparser import deptree


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
