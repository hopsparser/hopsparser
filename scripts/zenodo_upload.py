import pathlib
import urllib.parse
from collections.abc import Sequence

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
    access_token: str | None,
    config_path: pathlib.Path | None,
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
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=None,  # noqa: S113
    ) as client:
        deposit_url = base_url.join("deposit/depositions/").join(
            urllib.parse.quote(deposit_id, safe="")
        )
        deposit_info = client.get(deposit_url)
        deposit_info.raise_for_status()
        bucket_url = httpx.URL(deposit_info.json()["links"]["bucket"] + "/")
        deposit_metadata = deposit_info.json()["metadata"]
        existing_files = deposit_info.json()["files"]
        existing_file_names = {f["filename"] for f in existing_files}
        files_to_upload = [f for f in files if f.name not in existing_file_names]
        click.echo(
            f"Uploading {len(files_to_upload)} files to Zenodo deposit {deposit_id}:"
            f" “{deposit_metadata['title']}” v{deposit_metadata.get('version', '??')}"
        )
        if (n := len(files) - len(files_to_upload)) > 0:
            # TODO: check md5
            click.echo(f"{n} files are already present in the deposit")
        # TODO: is this multiplexing?
        # rich.progress, I hate you
        with rich.progress.Progress(
            *rich.progress.Progress.get_default_columns(),
            rich.progress.MofNCompleteColumn(),
        ) as progress:
            for f in progress.track(
                files_to_upload,
                description="Uploading…",
            ):
                with rich.progress.Progress(
                    *rich.progress.Progress.get_default_columns(),
                    rich.progress.DownloadColumn(),
                    rich.progress.TransferSpeedColumn(),
                    transient=True,
                ) as f_progress:
                    with open(f, "rb") as in_stream:
                        with f_progress.wrap_file(
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
