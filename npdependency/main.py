import contextlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Dict, Generator, Optional, Union
import warnings

import click
import click_pathlib

from npdependency import graph_parser
from npdependency import conll2018_eval as evaluator

# Python 3.7 shim
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore[misc]


device_opt = click.option(
    "--device",
    default="cpu",
    help="The device to use for the parsing model. (cpu, gpu:0, …).",
    show_default=True,
)


def make_metrics_table(metrics: Dict[str, float]) -> str:
    column_width = max(7, *(len(k) for k in metrics.keys()))
    keys, values = zip(*metrics.items())
    headers = "|".join(k.center(column_width) for k in keys)
    midrule = "|".join([f":{'-'*(column_width-2)}:"] * len(keys))
    row = "|".join(f"{100*v:05.2f}".center(column_width) for v in values)
    return "\n".join(f"|{r}|" for r in (headers, midrule, row))


@contextlib.contextmanager
def dir_manager(
    path: Optional[Union[pathlib.Path, str]] = None
) -> Generator[pathlib.Path, None, None]:
    """A context manager to deal with a directory, default to a self-destruct temp one."""
    if path is None:
        with tempfile.TemporaryDirectory() as tempdir:
            d_path = pathlib.Path(tempdir)
            yield d_path
    else:
        d_path = pathlib.Path(path).resolve()
        d_path.mkdir(parents=True, exist_ok=True)
        yield d_path


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
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, allow_dash=True),
)
@click.argument(
    "output_path",
    type=click.Path(resolve_path=True, dir_okay=False, writable=True, allow_dash=True),
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

    graph_parser.parse(
        batch_size=batch_size,
        in_file=input_path,
        model_path=model_path,
        out_file=output_path,
        overrides={"device": device},
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
    "--fasttext",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
    help="The path to either an existing FastText model or a raw text file to train one. If this option is absent, a model will be trained from the parsing train set.",
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
def train(
    config_file: pathlib.Path,
    dev_file: Optional[pathlib.Path],
    device: str,
    fasttext: Optional[pathlib.Path],
    max_tree_length: Optional[int],
    output_dir: pathlib.Path,
    overwrite: bool,
    rand_seed: int,
    test_file: Optional[pathlib.Path],
    train_file: pathlib.Path,
):
    model_path = output_dir / "model"
    graph_parser.train(
        config_file=config_file,
        dev_file=dev_file,
        train_file=train_file,
        fasttext=fasttext,
        max_tree_length=max_tree_length,
        model_path=model_path,
        overrides={"device": device},
        overwrite=overwrite,
        rand_seed=rand_seed,
    )
    output_metrics = dict()
    if dev_file is not None:
        parsed_devset_path = output_dir / f"{dev_file.stem}.parsed.conllu"
        graph_parser.parse(
            model_path, dev_file, parsed_devset_path, overrides={"device": device}
        )
        gold_devset = evaluator.load_conllu_file(dev_file)
        syst_devset = evaluator.load_conllu_file(parsed_devset_path)
        dev_metrics = evaluator.evaluate(gold_devset, syst_devset)
        for m in ("UPOS", "LAS"):
            output_metrics[f"{m} (dev)"] = dev_metrics[m].f1

    if test_file is not None:
        parsed_testset_path = output_dir / f"{test_file.stem}.parsed.conllu"
        graph_parser.parse(
            model_path, test_file, parsed_testset_path, overrides={"device": device}
        )
        gold_testset = evaluator.load_conllu_file(test_file)
        syst_testset = evaluator.load_conllu_file(parsed_testset_path)
        test_metrics = evaluator.evaluate(gold_testset, syst_testset)
        for m in ("UPOS", "LAS"):
            output_metrics[f"{m} (test)"] = test_metrics[m].f1

    if output_metrics:
        click.echo(make_metrics_table(output_metrics))


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
    type=click.Choice(("md", "json")),
    default="md",
    help="The output format for the scores",
    show_default=True,
)
def evaluate(
    device: str,
    intermediary_dir: str,
    model_path: pathlib.Path,
    out_format: Literal["md", "json"],
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
        graph_parser.parse(
            model_path, input_file, output_file, overrides={"device": device}
        )
        gold_set = evaluator.load_conllu_file(str(input_file))
        syst_set = evaluator.load_conllu_file(str(output_file))
    metrics = evaluator.evaluate(gold_set, syst_set)
    if out_format == "md":
        output_metrics = {n: metrics[n].f1 for n in ("UPOS", "UAS", "LAS")}
        click.echo(
            make_metrics_table(output_metrics)
        )
    elif out_format == "json":
        json.dump({m: metrics[m].f1 for m in ("UPOS", "UAS", "LAS")}, sys.stdout)
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
        ["uvicorn", "npdependency.server:app", "--port", str(port)],
        env={
            "models": json.dumps({"default": str(model_path)}),
            "device": device,
            **os.environ,
        },
    )


if __name__ == "__main__":
    cli()
