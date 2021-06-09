import pathlib
import pytest


@pytest.fixture
def test_data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def treebank(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "truncated-sv_talbanken-ud-dev.conllu"


@pytest.fixture
def raw_text(test_data_dir: pathlib.Path) -> pathlib.Path:
    return test_data_dir / "raw.txt"


@pytest.fixture(
    params=["toy_nobert", pytest.param("toy_flaubert", marks=pytest.mark.slow)]
)
def train_config(test_data_dir: pathlib.Path, request) -> pathlib.Path:
    return test_data_dir / f"{request.param}.yaml"
