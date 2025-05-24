import contextlib
import datetime
import logging
import math
import os
import pathlib
import sys
import tempfile
import warnings
from types import FrameType
from typing import IO, Any, Callable, Generator, Sequence, TextIO, Type, Union, cast

import click
import loguru
import rich.progress
import rich.text
from loguru import logger


@contextlib.contextmanager
def smart_open(
    f: str | pathlib.Path | IO, mode: str = "r", *args, **kwargs
) -> Generator[IO, None, None]:
    """Open files, paths and i/o streams transparently."""
    fh: IO
    if f == "-":
        if "r" in mode:
            stream = sys.stdin
        else:
            stream = sys.stdout
        if "b" in mode:
            fh = stream.buffer
        else:
            fh = stream
        close = False
    elif hasattr(f, "write") or hasattr(f, "read"):
        fh = cast(IO, f)
        close = False
    else:
        fh = open(cast(pathlib.Path | str, f), mode, *args, **kwargs)
        close = True

    try:
        yield fh
    finally:
        if close:
            try:
                fh.close()
            except AttributeError:
                pass


@contextlib.contextmanager
def dir_manager(
    path: pathlib.Path | str | None = None,
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


# TODO: use rich table markdown style for this instead
def make_markdown_metrics_table(metrics: dict[str, float]) -> str:
    column_width = max(7, *(len(k) for k in metrics.keys()))
    keys, values = zip(*metrics.items(), strict=True)
    headers = "|".join(k.center(column_width) for k in keys)
    midrule = "|".join([f":{'-' * (column_width - 2)}:"] * len(keys))
    row = "|".join(f"{100 * v:05.2f}".center(column_width) for v in values)
    return "\n".join(f"|{r}|" for r in (headers, midrule, row))


class InterceptHandler(logging.Handler):
    def __init__(self, wrapped_name: str | None = None, *args, **kwargs):
        self.wrapped_name = wrapped_name
        super().__init__(*args, **kwargs)

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        if frame is None:
            warnings.warn(
                "Catching calls to logging is impossible in stackless environment,"
                " logging from external libraries might be lost.",
                stacklevel=2,
            )
        else:
            if self.wrapped_name is not None:
                logger.opt(depth=depth, exception=record.exc_info).log(
                    level, f"({self.wrapped_name}) {record.getMessage()}"
                )
            else:
                logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    appname: str = "hops",
    console: rich.console.Console | None = None,
    log_file: pathlib.Path | None = None,
    replace_warnings: bool = True,
    sink: Union[
        str,
        "loguru.PathLikeStr",
        TextIO,
        "loguru.Writable",
        Callable[["loguru.Message"], None],
        logging.Handler,
        None,
    ] = None,
    verbose: bool = False,
) -> list[int]:
    res: list[int] = []
    if console is None:
        console = rich.get_console()
    if sink is None:
        sink = lambda m: console.print(m, end="")  # noqa: E731
    logger.remove()  # Remove the default logger
    if "SLURM_JOB_ID" in os.environ:
        local_id = os.environ.get("SLURM_LOCALID", "someproc")
        node_name = os.environ.get("SLURMD_NODENAME", "somenode")
        appname = (
            f"{appname} ({os.environ.get('SLURM_PROCID', 'somerank')} [{local_id}@{node_name}])"
        )

    if verbose:
        log_level = "DEBUG"
        log_fmt = (
            f"\\[{appname}]"
            " [green]{time:YYYY-MM-DD HH:mm:ss.SSS}[/green] | [blue]{level: <8}[/blue] |"
            " {message}"
        )
    else:
        logging.getLogger(None).setLevel(logging.CRITICAL)
        log_level = "INFO"
        log_fmt = (
            f"\\[{appname}]"
            " [green]{time:YYYY-MM-DD}T{time:HH:mm:ss}[/green] {level} "
            " {message}"
        )

    res.append(
        logger.add(
            sink,
            colorize=True,
            enqueue=True,
            format=log_fmt,
            level=log_level,
        )
    )

    if log_file:
        res.append(
            logger.add(
                log_file,
                colorize=False,
                enqueue=True,
                format=(
                    f"[{appname}] {{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | {{message}}"
                ),
                level="DEBUG",
            )
        )

    # Deal with stdlib.logging
    # Yes, listing them all is annoying
    for libname in (
        "datasets",
        "huggingface/tokenizers",
        "huggingface_hub",
        "lightning",
        "lightning.pytorch",
        "lightning_fabric",
        "pytorch_lightning",
        "torch",
        "torchmetrics",
        "transformers",
    ):
        lib_logger = logging.getLogger(libname)
        # FIXME: ugly, but is there a better way? What if they rely on having other handlers?
        if lib_logger.handlers:
            lib_logger.handlers = []
        lib_logger.addHandler(InterceptHandler(libname))
        logger.debug(f"Intercepting logging from {libname}")

    # Deal with stdlib.warnings
    def showwarning(message, category, filename, lineno, file=None, line=None):
        logger.warning(warnings.formatwarning(message, category, filename, lineno, None).strip())

    if replace_warnings:
        warnings.showwarning = showwarning

    return res


def log_epoch(epoch_name: str, metrics: dict[str, Any]):
    metrics_table = "\t".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(f"Epoch {epoch_name}: {metrics_table}")


class SpeedColumn(rich.progress.ProgressColumn):
    def render(self, task: rich.progress.Task) -> rich.text.Text:
        if not task.speed:
            return rich.text.Text("-:--:--")
        if task.speed >= 1:
            return rich.text.Text(f"{task.speed:.2f} it/s")
        else:
            return rich.text.Text(f"{datetime.timedelta(seconds=math.ceil(1 / task.speed))} /it")


# NOTE: if the need arise, using a separator regex instead of string would not be very hard but for
# now we don't need the extra complexity.
class SeparatedTuple(click.ParamType):
    """A click parameters type that accept tuples formatted as strings spearated by an arbitrary
    separator.

    This is particularly useful to make variadic composite parameters. For instance, this is how we
    specify tagged paths for the `train_multi` command:

    ```python
    @click.argument(
        "train_files", type=SeparatedTuple(
            ":",
            (
                str,
                click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path)
            ),
        ),
        nargs=-1,
    )
    ```

    This parses argumenst of the form `somelabel:/path/to/something` and returns a couple `(label:
    str, path: pathlib.Path)`.
    """

    def __init__(self, separator: str, types: Sequence[Type | click.ParamType]):
        self.separator = separator
        self.types = [click.types.convert_type(ty) for ty in types]

    def to_info_dict(self) -> dict[str, Any]:
        info_dict = super().to_info_dict()
        info_dict["types"] = [t.to_info_dict() for t in self.types]
        return info_dict

    @property  # type: ignore[no-redef]
    def name(self) -> str:  # type: ignore[override]  # noqa: F811
        return f"<{' '.join(ty.name for ty in self.types)}>"

    # NOTE: the way this is written forbids using the separator character in the values at all.
    # If the needs arises, we could allow escaping it.
    def convert(self, value: Any, param: click.Parameter | None, ctx: click.Context | None) -> Any:
        if isinstance(value, str):
            value = value.split(self.separator)

        len_type = len(self.types)
        len_value = len(value)

        if len_value != len_type:
            self.fail(
                f"{len_type} values are required, but {len_value} was given.",
                param=param,
                ctx=ctx,
            )

        return tuple(ty(x, param, ctx) for ty, x in zip(self.types, value, strict=True))
