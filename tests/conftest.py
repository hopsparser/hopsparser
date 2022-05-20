import pathlib

import pytest

from hopsparser import deptree, parser


@pytest.fixture(scope="session")
def test_data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def scripts_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "scripts"


@pytest.fixture(
    params=[
        "truncated-sv_talbanken-ud-dev-partial.conllu",
    ],
    scope="session",
)
def treebank(test_data_dir: pathlib.Path, request) -> pathlib.Path:
    return test_data_dir / request.param


@pytest.fixture(
    params=[
        "truncated-sv_talbanken-ud-dev.conllu",
    ],
    scope="session",
)
def test_treebank(test_data_dir: pathlib.Path, request) -> pathlib.Path:
    return test_data_dir / request.param


@pytest.fixture(scope="session")
def raw_text(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "raw.txt"


@pytest.fixture(
    params=[
        "toy_bert_fasttok",
        "toy_everylexer",
        "toy_flaubert",
        "toy_onlychars",
        "toy_onlyfasttext",
        "toy_onlywords",
    ],
    scope="session",
)
def train_config(test_data_dir: pathlib.Path, request) -> pathlib.Path:
    return test_data_dir / f"{request.param}.yaml"


@pytest.fixture
def model_path(
    tmp_path: pathlib.Path,
    train_config: pathlib.Path,
    treebank: pathlib.Path,
) -> pathlib.Path:
    model_path = tmp_path / "model"
    with open(treebank) as in_stream:
        trees = list(deptree.DepGraph.read_conll(in_stream))
    model = parser.BiAffineParser.initialize(config_path=train_config, treebank=trees)
    model.save(model_path)
    return model_path
