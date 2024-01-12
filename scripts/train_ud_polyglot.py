import pathlib
import re
import shutil
from typing import Optional, Sequence

import click
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Column, Table

from hopsparser import conll2018_eval as evaluator
from hopsparser import deptree, parser
from hopsparser.utils import setup_logging


@click.command(help="Train a polyglot parser on UD treebanks")
@click.argument(
    "config_file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "ud_dir",
    type=click.Path(readable=True, exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "output_dir",
    type=click.Path(resolve_path=True, file_okay=False, writable=True, path_type=pathlib.Path),
)
@click.option(
    "--device",
    default="cpu",
    help="The device to use for the parsing model. (cpu, gpu:0, …).",
    show_default=True,
)
@click.option(
    "--lang",
    help=(
        "Use all the treebanks for this language. Can be given multiple times."
        "Not providing any lang means training on every treebank."
    ),
    multiple=True,
)
@click.option(
    "--max-tree-length",
    type=int,
    help="The maximum length for trees to be taken into account in the training dataset.",
)
@click.option(
    "--origin-label-name",
    default="original_treebank",
    help=(
        "The label name to use for marking the treebank of origin in the MISC column of the input"
        " and output CoNLL-U files. If origin prediction is desired, this label should be present"
        " in the `extra_annotations` field of the parser config."
    ),
    show_default=True,
)
@click.option(
    "--overwrite",
    is_flag=True,
    help=(
        "If a model already in the output directory, restart training from scratch instead of"
        "continuing.",
    ),
)
@click.option(
    "--rand-seed",
    type=int,
    help=(
        "Force the random seed fo Python and Pytorch (see"
        " <https://pytorch.org/docs/stable/notes/randomness.html> for notes on reproducibility)"
    ),
)
@click.option(
    "--skip-unencodable",
    is_flag=True,
    help="Skip unencodable trees in the training set",
)
@click.option(
    "--train-with-lang-labels",
    is_flag=True,
    help="Whether to train the lang prediction head",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="How much info should we dump to the console",
)
def main(
    config_file: pathlib.Path,
    output_dir: pathlib.Path,
    device: str,
    lang: Optional[Sequence[str]],
    max_tree_length: Optional[int],
    origin_label_name: str,
    overwrite: bool,
    rand_seed: int,
    skip_unencodable: bool,
    train_with_lang_labels: bool,
    ud_dir: pathlib.Path,
    verbose: bool,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(verbose=verbose, logfile=output_dir / "train.log")
    model_path = output_dir / "model"
    shutil.copy(config_file, output_dir / config_file.name)

    if lang is not None:
        lang = [lg.lower() for lg in lang]

    train_files = []
    dev_files = []
    test_files = []

    for treebank_dir in ud_dir.glob("UD_*"):
        if (m := re.match(r"UD_(?P<lang>.*?)-(?P<name>.*)", treebank_dir.name)) is None:
            continue
        treebank_lang = m.group("lang").lower()
        if lang is not None and treebank_lang not in lang:
            continue
        if (train := next(treebank_dir.glob("*-train.conllu"), None)) is not None:
            train_files.append((treebank_lang, train))
            logger.debug(f"train {treebank_lang} treebank: {train}")
        if (dev := next(treebank_dir.glob("*-dev.conllu"), None)) is not None:
            dev_files.append((treebank_lang, dev))
            logger.debug(f"dev {treebank_lang} treebank: {dev}")
        if (test := next(treebank_dir.glob("*-test.conllu"), None)) is not None:
            test_files.append((treebank_lang, test))
            logger.debug(f"test {treebank_lang} treebank: {test}")

    # This changes nothing to training since we shuffle at sample level
    train_files.sort()
    dev_files.sort()
    test_files.sort()

    concat_train_file = output_dir / "train.conllu"
    with concat_train_file.open("w") as out_stream:
        for label, path in train_files:
            with path.open() as in_stream:
                for tree in deptree.DepGraph.read_conll(in_stream, max_tree_length=max_tree_length):
                    if train_with_lang_labels:
                        labelled_tree = tree.replace(
                            misc={
                                node.identifier: {origin_label_name: label} for node in tree.nodes
                            }
                        )
                    else:
                        labelled_tree = tree
                    out_stream.write(labelled_tree.to_conllu())
                    out_stream.write("\n\n")

    if dev_files:
        concat_dev_file = output_dir / "dev.conllu"
        with concat_dev_file.open("w") as out_stream:
            for label, path in dev_files:
                with path.open() as in_stream:
                    for tree in deptree.DepGraph.read_conll(
                        in_stream, max_tree_length=max_tree_length
                    ):
                        if train_with_lang_labels:
                            labelled_tree = tree.replace(
                                misc={
                                    node.identifier: {origin_label_name: label}
                                    for node in tree.nodes
                                }
                            )
                        else:
                            labelled_tree = tree
                        out_stream.write(labelled_tree.to_conllu())
                        out_stream.write("\n\n")
    else:
        concat_dev_file = None

    parser.train(
        config_file=config_file,
        dev_file=concat_dev_file,
        device=device,
        train_file=concat_train_file,
        max_tree_length=max_tree_length,
        model_path=model_path,
        overwrite=overwrite,
        rand_seed=rand_seed,
        skip_unencodable=skip_unencodable,
    )

    console = Console()
    metric_names = ("UPOS", "UAS", "LAS", "CLAS")
    if dev_files is not None:
        dev_metrics_table = Table(
            "Treebank",
            *(Column(header=m, justify="center") for m in metric_names),
            box=box.HORIZONTALS,
            title="Dev metrics",
        )
        for label, path in dev_files:
            parsed_devset_path = output_dir / f"{label}-{path.stem}.parsed.conllu"
            parser.parse(model_path, path, parsed_devset_path, device=device)
            gold_devset = evaluator.load_conllu_file(path)
            syst_devset = evaluator.load_conllu_file(parsed_devset_path)
            metrics = evaluator.evaluate(gold_devset, syst_devset)
            dev_metrics_table.add_row(
                f"{label}-{path.stem}", *(f"{100*metrics[m].f1:.2f}" for m in metric_names)
            )
        console.print(dev_metrics_table)

    if test_files is not None:
        test_metrics_table = Table(
            "Treebank",
            *(Column(header=m, justify="center") for m in metric_names),
            box=box.HORIZONTALS,
            title="Test metrics",
        )
        for label, path in test_files:
            parsed_testset_path = output_dir / f"{label}-{path.stem}.parsed.conllu"
            parser.parse(model_path, path, parsed_testset_path, device=device)
            gold_testset = evaluator.load_conllu_file(path)
            syst_testset = evaluator.load_conllu_file(parsed_testset_path)
            metrics = evaluator.evaluate(gold_testset, syst_testset)
            test_metrics_table.add_row(
                f"{label}-{path.stem}", *(f"{100*metrics[m].f1:.2f}" for m in metric_names)
            )
        console.print(test_metrics_table)


if __name__ == "__main__":
    main()
