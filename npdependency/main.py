import os
import pathlib
import subprocess
import sys
import tempfile
from typing import IO, TextIO, Union

import click
import click_pathlib

from npdependency import graph_parser
from npdependency.utils import smart_open


@click.group()
def cli():
    pass


@cli.command(help="Parse a raw or tokenized file")
@click.argument(
    "config_path",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
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
@click.option(
    "--device", default="cpu", help="The device to use for parsing. (cpu, gpu:0, …).", show_default=True
)
@click.option(
    "--raw",
    is_flag=True,
    help="Instead of a CoNLL-U file, take as input a document with one sentence per line, with tokens separated by spaces",
)
def parse(
    config_path: pathlib.Path,
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

    if raw:
        intermediary_file = tempfile.TemporaryFile(mode="w+")
        with smart_open(input_file, "r") as in_stream:
            for line in in_stream:
                if not line or line.isspace():
                    continue
                for i, w in enumerate(line.strip().split(), start=1):
                    intermediary_file.write(f"{i}\t{w}\n")
                intermediary_file.write("\n")
        intermediary_file.seek(0)
        input_file = intermediary_file

    output_file: Union[TextIO, str]
    if output_path == "-":
        output_file = sys.stdout
    else:
        output_file = output_path

    graph_parser.parse(
        config_path, input_file, output_file, overrides={"device": device}
    )


@cli.command(help="Start a parsing server")
@click.argument(
    "config_path",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--device", default="cpu", help="The device to use for parsing. (cpu, gpu:0, …).", show_default=True
)
@click.option(
    "--port", type=int, default=8000, help="The port to use for the API endpoint.", show_default=True,
)
def serve(
    config_path: pathlib.Path,
    device: str,
):
    subprocess.run(
        ["uvicorn", "npdependency.server:app"],
        env={
            "models": f'{{"default":"{config_path}"}}',
            "device": device,
            **os.environ,
        },
    )


if __name__ == "__main__":
    cli()
