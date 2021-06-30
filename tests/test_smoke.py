import filecmp
import pathlib

import torch.cuda

import pytest
import pytest_console_scripts

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")


@pytest.mark.parametrize("device", devices)
def test_train_parse(
    device: str,
    raw_text: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
    train_config: pathlib.Path,
    treebank: pathlib.Path,
):
    ret = script_runner.run(
        "hopsparser",
        "train",
        "--device",
        device,
        str(train_config),
        str(treebank),
        str(tmp_path),
        "--dev-file",
        str(treebank),
        "--test-file",
        str(treebank),
    )
    assert ret.success
    ret = script_runner.run(
        "hopsparser",
        "parse",
        "--device",
        device,
        str(tmp_path / "model"),
        str(treebank),
        str(tmp_path / f"{treebank.stem}.parsed2.conllu"),
    )
    assert ret.success
    assert filecmp.cmp(
        tmp_path / f"{treebank.stem}.parsed.conllu",
        tmp_path / f"{treebank.stem}.parsed2.conllu",
        shallow=False,
    )
    ret = script_runner.run(
        "hopsparser",
        "parse",
        "--device",
        device,
        "--raw",
        str(tmp_path / "model"),
        str(raw_text),
        str(tmp_path / f"{raw_text.stem}.parsed.conllu"),
    )
    assert ret.success
    ret = script_runner.run(
        "hopsparser",
        "parse",
        "--device",
        device,
        str(tmp_path / "model"),
        str(tmp_path / f"{raw_text.stem}.parsed.conllu"),
        str(tmp_path / f"{raw_text.stem}.reparsed.conllu"),
    )
    assert ret.success
    assert filecmp.cmp(
        str(tmp_path / f"{raw_text.stem}.parsed.conllu"),
        str(tmp_path / f"{raw_text.stem}.reparsed.conllu"),
        shallow=False,
    )


def test_gold_evaluation(
    script_runner: pytest_console_scripts.ScriptRunner, treebank: pathlib.Path
):
    ret = script_runner.run(
        "eval_parse",
        "-v",
        str(treebank),
        str(treebank),
    )
    assert ret.success
