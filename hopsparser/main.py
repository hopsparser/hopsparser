import json
import os
import pathlib
import shutil
import subprocess
import sys
import warnings
from typing import Literal, Optional

import click
import click_pathlib
from rich.console import Console
from rich.table import Table
from rich import box

from hopsparser import conll2018_eval as evaluator
from hopsparser import parser
from hopsparser.utils import dir_manager, make_markdown_metrics_table, setup_logging

device_opt = click.option(
    "--device",
    default="cpu",
    help="The device to use for the parsing model. (cpu, gpu:0, …).",
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
    type=click_pathlib.Path(resolve_path=True, exists=True),
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
    help="In raw mode, silently ignore sentences that can't be encoded (for instance too long sentences when using a transformer model).",
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
    batch_size: Optional[int],
    device: str,
    ignore_unencodable: bool,
    input_path: str,
    output_path: str,
    model_path: pathlib.Path,
    raw: bool,
):
    if ignore_unencodable and not raw:
        warnings.warn("--ignore-unencodable is only meaningful in raw mode")

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
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.argument(
    "train_file",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.argument(
    "output_dir",
    type=click_pathlib.Path(resolve_path=True, file_okay=False, writable=True),
)
@click.option(
    "--dev-file",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
    help="A CoNLL-U treebank to use as a development dataset.",
)
@click.option(
    "--max-tree-length",
    type=int,
    help="The maximum length for trees to be taken into account in the training dataset.",
)
@click.option(
    "--rand-seed",
    type=int,
    help="Force the random seed fo Python and Pytorch (see <https://pytorch.org/docs/stable/notes/randomness.html> for notes on reproducibility)",
)
@click.option(
    "--test-file",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
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
    dev_file: Optional[pathlib.Path],
    device: str,
    max_tree_length: Optional[int],
    output_dir: pathlib.Path,
    overwrite: bool,
    rand_seed: int,
    test_file: Optional[pathlib.Path],
    train_file: pathlib.Path,
    verbose: bool,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(verbose=verbose, logfile=output_dir / "train.log")
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
    metrics_table = Table(box=box.HORIZONTALS)
    metrics_table.add_column("Split")
    metrics = ("UPOS", "UAS", "LAS")
    for m in metrics:
        metrics_table.add_column(m, justify="center")
    if dev_file is not None:
        parsed_devset_path = output_dir / f"{dev_file.stem}.parsed.conllu"
        parser.parse(model_path, dev_file, parsed_devset_path, device=device)
        gold_devset = evaluator.load_conllu_file(dev_file)
        syst_devset = evaluator.load_conllu_file(parsed_devset_path)
        dev_metrics = evaluator.evaluate(gold_devset, syst_devset)
        metrics_table.add_row("Dev", *(f"{100*dev_metrics[m].f1:.2f}" for m in metrics))

    if test_file is not None:
        parsed_testset_path = output_dir / f"{test_file.stem}.parsed.conllu"
        parser.parse(model_path, test_file, parsed_testset_path, device=device)
        gold_testset = evaluator.load_conllu_file(test_file)
        syst_testset = evaluator.load_conllu_file(parsed_testset_path)
        test_metrics = evaluator.evaluate(gold_testset, syst_testset)
        metrics_table.add_row(
            "Test", *(f"{100*test_metrics[m].f1:.2f}" for m in metrics)
        )

    if metrics_table.rows:
        console = Console()
        console.print(metrics_table)


@cli.command(help="Evaluate a trained model")
@click.argument(
    "model_path",
    type=click_pathlib.Path(resolve_path=True, exists=True),
)
@click.argument(
    "treebank_path",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, allow_dash=True),
)
@device_opt
@click.option(
    "--intermediary-dir",
    type=click_pathlib.Path(resolve_path=True, file_okay=False, writable=True),
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
    metrics_names = ("UPOS", "UAS", "LAS")
    if out_format == "md":
        output_metrics = {n: metrics[n].f1 for n in metrics_names}
        click.echo(make_markdown_metrics_table(output_metrics))
    elif out_format == "terminal":
        metrics_table = Table(box=box.HORIZONTALS)
        for m in metrics_names:
            metrics_table.add_column(m, justify="center")
        metrics_table.add_row(*(f"{100*metrics[m].f1:.2f}" for m in metrics_names))
        console = Console()
        console.print(metrics_table)
    elif out_format == "json":
        json.dump({m: metrics[m].f1 for m in metrics_names}, sys.stdout)
    else:
        raise ValueError(f"Unkown format {out_format!r}.")


@cli.command(help="Start a parsing server")
@click.argument(
    "model_path",
    type=click_pathlib.Path(resolve_path=True, exists=True),
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
