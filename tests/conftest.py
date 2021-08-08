import json
import pathlib
from typing import Generator

from fastapi.testclient import TestClient
import pytest

from hopsparser import parser, deptree


@pytest.fixture(scope="session")
def test_data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def treebank(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "truncated-sv_talbanken-ud-dev.conllu"


@pytest.fixture
def raw_text(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "raw.txt"


@pytest.fixture(scope="session")
def fasttext_model_path(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "fasttext_model.bin"


@pytest.fixture(
    params=[
        "toy_nobert",
        pytest.param("toy_flaubert", marks=pytest.mark.slow),
        pytest.param("toy_bert_fasttok", marks=pytest.mark.slow),
    ]
)
def train_config(test_data_dir: pathlib.Path, request) -> pathlib.Path:
    return test_data_dir / f"{request.param}.yaml"


@pytest.fixture
def model_path(
    tmp_path: pathlib.Path, train_config: pathlib.Path, treebank: pathlib.Path
) -> pathlib.Path:
    model_path = tmp_path / "model"
    with open(treebank) as in_stream:
        trees = list(deptree.DepGraph.read_conll(in_stream))
    _ = parser.BiAffineParser.initialize(
        config_path=train_config, model_path=model_path, treebank=trees
    )
    return model_path


@pytest.fixture
def api_client(
    model_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    monkeypatch.setenv("models", json.dumps({"default": str(model_path)}))
    from hopsparser.server import app

    yield TestClient(app=app)
