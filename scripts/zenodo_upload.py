import pathlib
from typing import Optional, Sequence
import urllib.parse

import click
import httpx
import rich.progress
import yaml


@click.command(help="Upload files to a Zenodo deposit.")
@click.argument("deposit_id")
@click.argument(
    "files",
    type=click.Path(readable=True, path_type=pathlib.Path),
    nargs=-1,
)
@click.option(
    "--access-token",
    metavar="TOKEN",
    help="A Zenodo API token with upload rights for the targeted deposit",
)
@click.option(
    "--config",
    "config_path",
    help="A YAML file with an upload config",
    type=click.Path(readable=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--sandbox",
    is_flag=True,
    help="Whether to use sandbox.zenodo.org instead of the real Zenodo",
)
def upload(
    access_token: Optional[str],
    config_path: Optional[pathlib.Path],
    deposit_id: str,
    files: Sequence[pathlib.Path],
    sandbox: bool,
):
    if config_path is not None:
        with open(config_path) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.SafeLoader)
    else:
        config = None
    if access_token is None:
        if config is None:
            raise ValueError("You must either provide a config file or an access token")
        else:
            access_token = config["access_token"]

    if sandbox:
        base_url = httpx.URL("https://sandbox.zenodo.org/api/")
    else:
        base_url = httpx.URL("https://zenodo.org/api/")

    with httpx.Client(
        http2=True,
        params={"access_token": access_token},
        timeout=None,
    ) as client:
        deposit_url = base_url.join("deposit/depositions/").join(
            urllib.parse.quote(deposit_id, safe="")
        )
        deposit_info = client.get(deposit_url)
        deposit_info.raise_for_status()
        bucket_url = httpx.URL(deposit_info.json()["links"]["bucket"] + "/")
        deposit_metadata = deposit_info.json()["metadata"]
        click.echo(
            f"Uploading {len(files)} files to Zenodo deposit {deposit_id}:"
            f" “{deposit_metadata['title']}” v{deposit_metadata.get('version', '??')}"
        )
        with rich.progress.Progress(
            *rich.progress.Progress.get_default_columns(),
            rich.progress.DownloadColumn(),
            rich.progress.TransferSpeedColumn(),
        ) as progress:
            for f in files:
                with open(f, "rb") as in_stream:
                    with progress.wrap_file(
                        in_stream,
                        total=f.stat().st_size,
                        description=f"Uploading {f.name}",
                    ) as wrapped_stream:
                        r = client.put(
                            bucket_url.join(urllib.parse.quote(f.name, safe="")),
                            content=wrapped_stream,
                        )
                        try:
                            r.raise_for_status()
                        except httpx.HTTPStatusError as e:
                            click.echo(f"Error with upload of {f.name}")
                            click.echo(r.json())
                            raise e


if __name__ == "__main__":
    upload()
