import pathlib
import sys
from typing import TextIO, Union

import click
import click_pathlib

from npdependency import graph_parser


@click.group()
def cli():
    pass


@cli.command()
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
    "--device", default="cpu", help="The device to use for parsing. (cpu, gpu:0, …)."
)
@click.option(
    "--tokenize", is_flag=True, help="Pass this if the input has to be tokenized."
)
def parse(
    config_path: pathlib.Path,
    input_path: str,
    output_path: str,
    device: str,
    tokenize: bool,
):
    input_file: Union[TextIO, str]
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
        config_path, input_file, output_file, overrides={"device": device}
    )


if __name__ == "__main__":
    cli()
