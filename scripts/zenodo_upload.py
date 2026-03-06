import hashlib
import pathlib
import sys
import urllib.parse
from collections.abc import Sequence

import click
import httpx
import rich.progress
import yaml


def upload_file(f: pathlib.Path, client: httpx.Client, target: httpx.URL):
    with rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(),
        rich.progress.DownloadColumn(),
        rich.progress.TransferSpeedColumn(),
        refresh_per_second=1,
        transient=True,
    ) as progress:
        with f.open("rb") as in_stream:
            response = client.put(
                target,
                content=progress.wrap_file(
                    in_stream,
                    description=f"Uploading {f.name}",
                    total=f.stat().st_size,
                ),
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                click.echo(f"Error with upload of {f.name}", file=sys.stderr)
                click.echo(response.json(), file=sys.stderr)
                raise e


def _upload(client: httpx.Client, files: Sequence[pathlib.Path], bucket: httpx.URL):
    # rich.progress, I hate you
    with rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeElapsedColumn(),
    ) as progress:
        for f in progress.track(
            files,
            description="Uploading…",
        ):
            upload_file(
                f,
                client=client,
                target=bucket.join(urllib.parse.quote(f.name, safe="")),
            )


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
        headers={"Authorization": f"Bearer {access_token}"},
        http2=True,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=16),
        timeout=10,  # noqa: S113
    ) as client:
        deposit_url = base_url.join("deposit/depositions/").join(
            urllib.parse.quote(deposit_id, safe="")
        )
        deposit_info = client.get(deposit_url)
        deposit_info.raise_for_status()
        deposit_metadata = deposit_info.json()["metadata"]
        existing_files = {f["filename"]: f["checksum"] for f in deposit_info.json()["files"]}
        files_to_upload = []
        for f in rich.progress.track(files, description="Checking local files…"):
            if (c := existing_files.get(f.name)) is not None:
                with f.open("rb") as in_stream:
                    local_checksum = hashlib.file_digest(in_stream, "md5").hexdigest()
                if local_checksum != c:
                    click.echo(
                        (
                            f"WARNING: local file {f} has a checksum ({local_checksum})"
                            f"that differs from its remote counterpart's ({c})"
                        ),
                        file=sys.stderr,
                    )
            else:
                files_to_upload.append(f)
        click.echo(
            (
                f"Uploading {len(files_to_upload)} files to Zenodo deposit {deposit_id}:"
                f" “{deposit_metadata['title']}” v{deposit_metadata.get('version', '??')}"
            ),
            file=sys.stderr,
        )
        if (n := len(files) - len(files_to_upload)) > 0:
            click.echo(f"{n} files are already present in the deposit")

        _upload(
            bucket=httpx.URL(f"{deposit_info.json()['links']['bucket']}/"),
            client=client,
            files=files_to_upload,
        )


if __name__ == "__main__":
    upload()
