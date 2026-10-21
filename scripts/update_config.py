from typing import Any
import pathlib

import click
import yaml


TARGET_VERSION = 2


def v1(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "biaffine_dim": config["mlp_input"],
        "deprels_hidden_dim": config["mlp_lab_hidden"],
        "dropout": config["mlp_dropout"],
        "heads_hidden_dim": config["mlp_arc_hidden"],
        "pos_hidden_dim": config["mlp_tag_hidden"],
        "encoder": {
            "dropout": config["encoder_dropout"],
            "rnn_dim": config["mlp_input"] // 2,
            "lexers": config["lexers"],
        },
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "lr": config["lr"],
        "version": 2,
    }


UPDATERS = {1: v1}


@click.command()
@click.argument(
    "source",
    type=click.Path(
        exists=True,
        readable=True,
        dir_okay=False,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    "dest",
    default=None,
    type=click.Path(
        writable=True,
        dir_okay=False,
        path_type=pathlib.Path,
    ),
)
def main(source: pathlib.Path, dest: pathlib.Path | None):
    if dest is None:
        dest = source

    with open(source) as in_stream:
        old_config = yaml.load(in_stream, Loader=yaml.SafeLoader)

    source_version = old_config.get("version", 1)

    if not isinstance(source_version, int) or not (1 <= source_version <= TARGET_VERSION):
        raise ValueError(f"Unsupported config version: {source_version}")
    elif source_version == TARGET_VERSION:
        click.echo("Config already at the current version: nothing to do")
        return

    new_config = old_config
    for v in range(source_version, TARGET_VERSION):
        click.echo(f"v{v} → v{v + 1}")
        new_config = UPDATERS[v](new_config)

    with open(dest, "w") as out_stream:
        yaml.dump(new_config, out_stream)


if __name__ == "__main__":
    main()
