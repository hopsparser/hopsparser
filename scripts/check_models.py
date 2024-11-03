import json
import pathlib
import tempfile
from math import isclose
from tarfile import TarFile
from typing import Dict, TextIO

import click
import pooch
import rich

from hopsparser import conll2018_eval as evaluator
from hopsparser import parser


def check_model(
    device: str,
    intermediary_dir: pathlib.Path,
    model_url: str,
    scores: Dict[str, Dict[str, float]],
    treebank: Dict[str, pathlib.Path],
) -> bool:
    with tempfile.TemporaryDirectory(prefix=f"{intermediary_dir}/") as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        def extract_dir(fname, action, pooch):
            extract_dir = temp_path / "model"
            with TarFile.open(fname, "r") as tar_file:
                tar_file.extractall(path=extract_dir)
            extracted_content = list(extract_dir.iterdir())
            if len(extracted_content) != 1:
                raise ValueError(f"Invalid model archive. Content: {extracted_content}")
            return extracted_content[0]

        model_path = pooch.retrieve(
            model_url,
            known_hash=None,
            path=temp_path,
            processor=extract_dir,
        )

        for split_name, split_scores in scores.items():
            split_file = treebank[split_name]
            output_file = temp_path / f"{split_file.stem}.parsed.conllu"
            parser.parse(model_path, split_file, output_file, device=device)
            gold_set = evaluator.load_conllu_file(str(split_file))
            syst_set = evaluator.load_conllu_file(str(output_file))
            metrics = evaluator.evaluate(gold_set, syst_set)

            for score_name, score_value in split_scores.items():
                actual = metrics[score_name].f1
                if not isclose(actual, score_value, abs_tol=1e-4):
                    print(
                        f"Inconsistency for {score_name} on {split_name}: got {actual}, expected {score_value}"
                    )
                    return False
    return True


@click.command(help="Check the performance of multiple models")
@click.argument("models_list", type=click.File(mode="r"))
@click.option(
    "--device",
    default="cpu",
    help="The device to use for the parsing model. (cpu, gpu:0, …).",
    show_default=True,
)
def main(device: str, models_list: TextIO):
    logger = pooch.get_logger()
    logger.setLevel("WARNING")

    reference = json.load(models_list)
    with rich.progress.Progress() as progress:
        for dataset_name, dataset in progress.track(
            reference.items(), description="Checking models"
        ):
            if not dataset["treebank"]:
                print(f"No available data for {dataset_name}")
                continue
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = pathlib.Path(temp_dir)
                treebank = {
                    split_name: pathlib.Path(
                        pooch.retrieve(split_url, known_hash=None, path=temp_path)
                    )
                    for split_name, split_url in dataset["treebank"].items()
                }
                for model_name, model in progress.track(
                    dataset["models"].items(), description=f"Checking {dataset_name}"
                ):
                    ok = check_model(
                        device=device,
                        intermediary_dir=temp_path,
                        model_url=model["url"],
                        scores=model["scores"],
                        treebank=treebank,
                    )
                    if not ok:
                        print(f"Inconsistency with model {model_name} for treebank {dataset_name}.")


if __name__ == "__main__":
    main()
