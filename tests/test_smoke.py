import filecmp
import pathlib

import pytest
import pytest_console_scripts
import torch.cuda

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
    test_treebank: pathlib.Path,
):
    ret = script_runner.run(
        [
            "hopsparser",
            "train",
            "--device",
            device,
            str(train_config),
            str(treebank),
            str(tmp_path),
            "--dev-file",
            str(test_treebank),
            "--test-file",
            str(test_treebank),
        ]
    )
    assert ret.success
    ret = script_runner.run(
        [
            "eval_parse",
            "-v",
            str(tmp_path / f"{test_treebank.stem}.parsed.conllu"),
            str(test_treebank),
        ]
    )
    assert ret.success
    ret = script_runner.run(
        [
            "hopsparser",
            "parse",
            "--device",
            device,
            str(tmp_path / "model"),
            str(treebank),
            str(tmp_path / f"{test_treebank.stem}.parsed2.conllu"),
        ]
    )
    assert ret.success
    assert filecmp.cmp(
        tmp_path / f"{test_treebank.stem}.parsed.conllu",
        tmp_path / f"{test_treebank.stem}.parsed2.conllu",
        shallow=False,
    )
    ret = script_runner.run(
        [
            "hopsparser",
            "parse",
            "--device",
            device,
            "--raw",
            str(tmp_path / "model"),
            str(raw_text),
            str(tmp_path / f"{raw_text.stem}.parsed.conllu"),
        ]
    )
    assert ret.success
    ret = script_runner.run(
        [
            "hopsparser",
            "parse",
            "--device",
            device,
            str(tmp_path / "model"),
            str(tmp_path / f"{raw_text.stem}.parsed.conllu"),
            str(tmp_path / f"{raw_text.stem}.reparsed.conllu"),
        ]
    )
    assert ret.success
    assert filecmp.cmp(
        str(tmp_path / f"{raw_text.stem}.parsed.conllu"),
        str(tmp_path / f"{raw_text.stem}.reparsed.conllu"),
        shallow=False,
    )


@pytest.mark.parametrize("device", devices)
def test_train_multi_parse(
    device: str,
    raw_text: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    tmp_path: pathlib.Path,
    train_config: pathlib.Path,
    treebank: pathlib.Path,
    test_treebank: pathlib.Path,
):
    ret = script_runner.run(
        [
            "hopsparser",
            "train-multi",
            "--device",
            device,
            str(train_config),
            f"one:{treebank}",
            f"two:{treebank}",
            str(tmp_path),
            "--dev-file",
            f"one:{test_treebank}",
            "--dev-file",
            f"two:{test_treebank}",
            "--dev-file",
            f"three:{test_treebank}",
            "--test-file",
            f"one:{test_treebank}",
            "--test-file",
            f"three:{test_treebank}",
            "--test-file",
            f"four:{test_treebank}",
        ]
    )
    assert ret.success
    ret = script_runner.run(
        [
            "eval_parse",
            "-v",
            str(tmp_path / f"one-{test_treebank.stem}.parsed.conllu"),
            str(test_treebank),
        ]
    )
    assert ret.success
    ret = script_runner.run(
        [
            "eval_parse",
            "-v",
            str(tmp_path / f"three-{test_treebank.stem}.parsed.conllu"),
            str(test_treebank),
        ]
    )
    assert ret.success
    ret = script_runner.run(
        [
            "eval_parse",
            "-v",
            str(tmp_path / f"four-{test_treebank.stem}.parsed.conllu"),
            str(test_treebank),
        ]
    )
    assert ret.success
    ret = script_runner.run(
        [
            "hopsparser",
            "parse",
            "--device",
            device,
            str(tmp_path / "model"),
            str(treebank),
            str(tmp_path / f"{test_treebank.stem}.parsed2.conllu"),
        ]
    )
    assert ret.success
    ret = script_runner.run(
        [
            "hopsparser",
            "parse",
            "--device",
            device,
            "--raw",
            str(tmp_path / "model"),
            str(raw_text),
            str(tmp_path / f"{raw_text.stem}.parsed.conllu"),
        ]
    )
    assert ret.success
    ret = script_runner.run(
        [
            "hopsparser",
            "parse",
            "--device",
            device,
            str(tmp_path / "model"),
            str(tmp_path / f"{raw_text.stem}.parsed.conllu"),
            str(tmp_path / f"{raw_text.stem}.reparsed.conllu"),
        ]
    )
    assert ret.success
    assert filecmp.cmp(
        str(tmp_path / f"{raw_text.stem}.parsed.conllu"),
        str(tmp_path / f"{raw_text.stem}.reparsed.conllu"),
        shallow=False,
    )


def test_gold_evaluation(
    script_runner: pytest_console_scripts.ScriptRunner, test_treebank: pathlib.Path
):
    ret = script_runner.run(
        [
            "eval_parse",
            "-v",
            str(test_treebank),
            str(test_treebank),
        ]
    )
    assert ret.success


@pytest.mark.parametrize("device", devices)
@pytest.mark.script_launch_mode("subprocess")
def test_train_script(
    device: str,
    scripts_dir: pathlib.Path,
    script_runner: pytest_console_scripts.ScriptRunner,
    test_data_dir: pathlib.Path,
    tmp_path: pathlib.Path,
):
    ret = script_runner.run(
        [
            "python",
            str(scripts_dir / "train_models.py"),
            str(test_data_dir / "train_script" / "train_config.yaml"),
            str(test_data_dir / "train_script" / "treebanks"),
            "--devices",
            f"{device},{device}",
            "--rand-seeds",
            "0,1",
            "--out-dir",
            str(tmp_path / "train_script_output"),
        ]
    )
    assert ret.success
    # TODO: check that rerunning doesn't retrain and gives the same results

