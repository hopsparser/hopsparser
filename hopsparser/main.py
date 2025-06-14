import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Literal, Sequence, TextIO

import click
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Column, Table

from hopsparser import evaluator
from hopsparser import deptree, parser
from hopsparser.utils import (
    SeparatedTuple,
    dir_manager,
    make_markdown_metrics_table,
    setup_logging,
)

device_opt = click.option(
    "--device",
    default="cpu",
    help="The device to use for the parsing model. (cpu, cuda:0, …).",
    show_default=True,
)

verbose_opt = click.option(
    "--verbose",
    is_flag=True,
    help="How much info should we dump to the console",
)


@click.group(help="A graph dependency parser")
def cli():
    pass


@cli.command(help="Parse a raw or tokenized file")
@click.argument(
    "model_path",
    type=click.Path(resolve_path=True, exists=True, path_type=pathlib.Path),
)
@click.argument(
    "input_path",
    type=click.File("r"),
)
@click.argument(
    "output_path",
    type=click.File("w"),
    default="-",
)
@device_opt
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
)
@click.option(
    "--ignore-unencodable",
    is_flag=True,
    help="In raw mode, silently ignore sentences that can't be encoded (for instance too long sentences when using a transformer model).",
)
@click.option(
    "--raw",
    is_flag=True,
    help="Instead of a CoNLL-U file, take as input a document with one sentence per line, with tokens separated by spaces.",
)
def parse(
    batch_size: int | None,
    device: str,
    ignore_unencodable: bool,
    input_path: str,
    output_path: str,
    model_path: pathlib.Path,
    raw: bool,
):
    setup_logging()
    if ignore_unencodable and not raw:
        logger.warning("--ignore-unencodable is only meaningful in raw mode")

    parser.parse(
        batch_size=batch_size,
        device=device,
        in_file=input_path,
        model_path=model_path,
        out_file=output_path,
        raw=raw,
        strict=not ignore_unencodable,
    )


@cli.command(help="Train a parsing model")
@click.argument(
    "config_file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "train_file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "output_dir",
    type=click.Path(resolve_path=True, file_okay=False, writable=True, path_type=pathlib.Path),
)
@click.option(
    "--dev-file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
    help="A CoNLL-U treebank to use as a development dataset.",
)
@click.option(
    "--max-tree-length",
    type=int,
    help="The maximum length for trees to be taken into account in the training dataset.",
)
@click.option(
    "--rand-seed",
    default=0,
    help="Force the random seed fo Python and Pytorch (see <https://pytorch.org/docs/stable/notes/randomness.html> for notes on reproducibility)",
    show_default=True,
    type=int,
)
@click.option(
    "--test-file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
    help="A CoNLL-U treebank to use as a test dataset.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="If a model already in the output directory, restart training from scratch instead of continuing.",
)
@device_opt
@verbose_opt
def train(
    config_file: pathlib.Path,
    dev_file: pathlib.Path | None,
    device: str,
    max_tree_length: int | None,
    output_dir: pathlib.Path,
    overwrite: bool,
    rand_seed: int,
    test_file: pathlib.Path | None,
    train_file: pathlib.Path,
    verbose: bool,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(verbose=verbose, log_file=output_dir / "train.log")
    model_path = output_dir / "model"
    shutil.copy(config_file, output_dir / config_file.name)
    parser.train(
        config_file=config_file,
        dev_file=dev_file,
        device=device,
        train_file=train_file,
        max_tree_length=max_tree_length,
        model_path=model_path,
        overwrite=overwrite,
        rand_seed=rand_seed,
    )
    metrics = ("UPOS", "UAS", "LAS")
    metrics_table = Table(
        "Split",
        *(Column(header=m, justify="center") for m in metrics),
        box=box.HORIZONTALS,
        title="Evaluation metrics",
    )

    if dev_file is not None:
        parsed_devset_path = output_dir / f"{dev_file.stem}.parsed.conllu"
        parser.parse(model_path, dev_file, parsed_devset_path, device=device)
        gold_devset = evaluator.load_conllu_file(dev_file)
        syst_devset = evaluator.load_conllu_file(parsed_devset_path)
        dev_metrics = evaluator.evaluate(gold_devset, syst_devset)
        metrics_table.add_row("Dev", *(f"{100 * dev_metrics[m].f1:.2f}" for m in metrics))

    if test_file is not None:
        parsed_testset_path = output_dir / f"{test_file.stem}.parsed.conllu"
        parser.parse(model_path, test_file, parsed_testset_path, device=device)
        gold_testset = evaluator.load_conllu_file(test_file)
        syst_testset = evaluator.load_conllu_file(parsed_testset_path)
        test_metrics = evaluator.evaluate(gold_testset, syst_testset)
        metrics_table.add_row("Test", *(f"{100 * test_metrics[m].f1:.2f}" for m in metrics))

    if metrics_table.rows:
        console = Console()
        console.print(metrics_table)


@cli.command(
    help="Train a polyglot/cross-domain model. Treebank files should be given in a <label>:<path> form."
)
@click.argument(
    "config_file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "train_file",
    type=SeparatedTuple(":", (str, click.File(lazy=True))),
    nargs=-1,
)
@click.argument(
    "output_dir",
    type=click.Path(resolve_path=True, file_okay=False, writable=True, path_type=pathlib.Path),
)
@click.option(
    "--dev-file",
    type=SeparatedTuple(
        ":",
        (str, click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path)),
    ),
    multiple=True,
    help="CoNLL-U treebanks to use as a development dataset. Should be given in a <label>:<path> form. Can be given multiple times.",
)
@click.option(
    "--origin-label-name",
    default="original_treebank",
    help=(
        "The label name to use for marking the treebank of origin in the MISC column of the input and output CoNLL-U files."
        " If origin prediction is desired, this label should be present in the `extra_annotations` field of the parser config."
    ),
    show_default=True,
)
@click.option(
    "--max-tree-length",
    type=int,
    help="The maximum length for trees to be taken into account in the training dataset.",
)
@click.option(
    "--rand-seed",
    default=0,
    help="Force the random seed fo Python and Pytorch (see <https://pytorch.org/docs/stable/notes/randomness.html> for notes on reproducibility)",
    show_default=True,
    type=int,
)
@click.option(
    "--test-file",
    type=SeparatedTuple(
        ":",
        (str, click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path)),
    ),
    multiple=True,
    help="CoNLL-U treebanks to use as a development dataset. Should be given in a <label>:<path> form. Can be given multiple times.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="If a model already in the output directory, restart training from scratch instead of continuing.",
)
@device_opt
@verbose_opt
def train_multi(
    config_file: pathlib.Path,
    dev_file: Sequence[tuple[str, pathlib.Path]],
    device: str,
    max_tree_length: int | None,
    origin_label_name: str,
    output_dir: pathlib.Path,
    overwrite: bool,
    rand_seed: int,
    test_file: Sequence[tuple[str, pathlib.Path]],
    train_file: Sequence[tuple[str, TextIO]],
    verbose: bool,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(verbose=verbose, log_file=output_dir / "train.log")
    model_path = output_dir / "model"
    shutil.copy(config_file, output_dir / config_file.name)

    concat_train_file = output_dir / "train.conllu"
    with open(concat_train_file, "w") as out_stream:
        for label, f in train_file:
            for tree in deptree.DepGraph.read_conll(f, max_tree_length=max_tree_length):
                labelled_tree = tree.replace(
                    misc={node.identifier: {origin_label_name: label} for node in tree.nodes}
                )
                out_stream.write(labelled_tree.to_conllu())
                out_stream.write("\n\n")

    if dev_file:
        concat_dev_file = output_dir / "dev.conllu"
        with open(concat_dev_file, "w") as out_stream:
            for label, path in dev_file:
                with open(path) as in_stream:
                    for tree in deptree.DepGraph.read_conll(
                        in_stream, max_tree_length=max_tree_length
                    ):
                        labelled_tree = tree.replace(
                            misc={
                                node.identifier: {origin_label_name: label} for node in tree.nodes
                            }
                        )
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
    )

    console = Console()
    if dev_file is not None:
        dev_metrics = ("UPOS", "UAS", "LAS")
        dev_metrics_table = Table(
            "Treebank",
            *(Column(header=m, justify="center") for m in dev_metrics),
            box=box.HORIZONTALS,
            title="Dev metrics",
        )
        for label, path in dev_file:
            parsed_devset_path = output_dir / f"{label}-{path.stem}.parsed.conllu"
            parser.parse(model_path, path, parsed_devset_path, device=device)
            gold_devset = evaluator.load_conllu_file(path)
            syst_devset = evaluator.load_conllu_file(parsed_devset_path)
            metrics = evaluator.evaluate(gold_devset, syst_devset)
            dev_metrics_table.add_row(
                f"{label}-{path.stem}", *(f"{100 * metrics[m].f1:.2f}" for m in dev_metrics)
            )
        console.print(dev_metrics_table)

    if test_file is not None:
        test_metrics = ("UPOS", "UAS", "LAS")
        test_metrics_table = Table(
            "Treebank",
            *(Column(header=m, justify="center") for m in test_metrics),
            box=box.HORIZONTALS,
            title="Test metrics",
        )
        for label, path in test_file:
            parsed_testset_path = output_dir / f"{label}-{path.stem}.parsed.conllu"
            parser.parse(model_path, path, parsed_testset_path, device=device)
            gold_testset = evaluator.load_conllu_file(path)
            syst_testset = evaluator.load_conllu_file(parsed_testset_path)
            metrics = evaluator.evaluate(gold_testset, syst_testset)
            test_metrics_table.add_row(
                f"{label}-{path.stem}", *(f"{100 * metrics[m].f1:.2f}" for m in test_metrics)
            )
        console.print(test_metrics_table)


@cli.command(help="Evaluate a trained model")
@click.argument(
    "model_path",
    type=click.Path(resolve_path=True, exists=True, path_type=pathlib.Path),
)
@click.argument(
    "treebank_path",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, allow_dash=True),
)
@click.argument(
    "output",
    type=click.File(mode="w"),
    default="-",
)
@device_opt
@click.option(
    "--intermediary-dir",
    type=click.Path(resolve_path=True, file_okay=False, writable=True, path_type=pathlib.Path),
    help="A directory where the parsed data will be stored, defaults to a temp dir",
)
@click.option(
    "--to",
    "out_format",
    type=click.Choice(("json", "md", "terminal")),
    default="terminal",
    help="The output format for the scores",
    show_default=True,
)
def evaluate(
    device: str,
    intermediary_dir: str,
    model_path: pathlib.Path,
    out_format: Literal["json", "md", "terminal"],
    output: TextIO,
    treebank_path: str,
):
    input_file: pathlib.Path
    with dir_manager(intermediary_dir) as intermediary_path:
        if treebank_path == "-":
            input_file = intermediary_path / "input.conllu"
            input_file.write_text(sys.stdin.read())
        else:
            input_file = pathlib.Path(treebank_path)

        output_file = intermediary_path / "parsed.conllu"
        parser.parse(model_path, input_file, output_file, device=device)
        gold_set = evaluator.load_conllu_file(str(input_file))
        syst_set = evaluator.load_conllu_file(str(output_file))
    metrics = evaluator.evaluate(gold_set, syst_set)
    # TODO: harmonize this with what we have in the train script
    metrics_names = ("UPOS", "UAS", "LAS")
    if out_format == "md":
        output_metrics = {n: metrics[n].f1 for n in metrics_names}
        click.echo(make_markdown_metrics_table(output_metrics), file=output)
    elif out_format == "terminal":
        metrics_table = Table(box=box.HORIZONTALS)
        for m in metrics_names:
            metrics_table.add_column(m, justify="center")
        metrics_table.add_row(*(f"{100 * metrics[m].f1:.2f}" for m in metrics_names))
        console = Console(file=output)
        console.print(metrics_table)
    elif out_format == "json":
        json.dump({m: metrics[m].f1 for m in metrics_names}, output)
    else:
        raise ValueError(f"Unkown format {out_format!r}.")


@cli.command(help="Start a parsing server")
@click.argument(
    "model_path",
    type=click.Path(resolve_path=True, exists=True, path_type=pathlib.Path),
)
@click.option(
    "--device",
    default="cpu",
    help="The device to use for parsing. (cpu, gpu:0, …).",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="The port to use for the API endpoint.",
    show_default=True,
)
def serve(
    model_path: pathlib.Path,
    device: str,
    port: int,
):
    subprocess.run(
        ["uvicorn", "hopsparser.server:app", "--port", str(port)],
        env={
            "models": json.dumps({"default": str(model_path)}),
            "device": device,
            **os.environ,
        },
    )


if __name__ == "__main__":
    cli()
