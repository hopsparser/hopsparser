import contextlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Generator, IO, Optional, TextIO, Union

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


@click.group()
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
    "--raw",
    is_flag=True,
    help="Instead of a CoNLL-U file, take as input a document with one sentence per line, with tokens separated by spaces",
)
def parse(
    model_path: pathlib.Path,
    input_path: str,
    output_path: str,
    device: str,
    raw: bool,
):
    input_file: Union[IO[str], str]
    if input_path == "-":
        input_file = sys.stdin
    else:
        input_file = input_path

    output_file: Union[TextIO, str]
    if output_path == "-":
        output_file = sys.stdout
    else:
        output_file = output_path

    graph_parser.parse(
        model_path, input_file, output_file, overrides={"device": device}, raw=raw
    )


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
    type=click_pathlib.Path(resolve_path=True, file_okay=False),
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
        click.echo(
            "| UPOS  |  UAS  |  LAS  |\n"
            "|-------|-------|-------|\n"
            f"| {100*metrics['UPOS'].f1:.2f} | {100*metrics['UAS'].f1:.2f} | {100*metrics['LAS'].f1:.2f} |"
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
