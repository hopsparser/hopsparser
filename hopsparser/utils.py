import contextlib
import logging
import pathlib
import sys
import tempfile
from typing import IO, Dict, Generator, Optional, Union, cast
from loguru import logger


@contextlib.contextmanager
def smart_open(
    f: Union[pathlib.Path, str, IO], mode: str = "r", *args, **kwargs
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
        fh = open(cast(Union[pathlib.Path, str], f), mode, *args, **kwargs)
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


def make_markdown_metrics_table(metrics: Dict[str, float]) -> str:
    column_width = max(7, *(len(k) for k in metrics.keys()))
    keys, values = zip(*metrics.items())
    headers = "|".join(k.center(column_width) for k in keys)
    midrule = "|".join([f":{'-'*(column_width-2)}:"] * len(keys))
    row = "|".join(f"{100*v:05.2f}".center(column_width) for v in values)
    return "\n".join(f"|{r}|" for r in (headers, midrule, row))


def setup_logging(verbose: bool, logfile: Optional[pathlib.Path] = None):
    logger.remove(0)  # Remove the default logger
    appname = "hops"

    if verbose:
        log_level = "DEBUG"
        log_fmt = (
            f"[{appname}]"
            " <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> |"
            " <level>{message}</level>"
        )
    else:
        logging.getLogger(None).setLevel(logging.CRITICAL)
        log_level = "INFO"
        log_fmt = (
            f"[{appname}]"
            " <green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            " <level>{message}</level>"
        )

    logger.add(
        sys.stderr,
        level=log_level,
        format=log_fmt,
        colorize=True,
    )

    if logfile:
        logger.add(
            logfile,
            level="DEBUG",
            format=(
                f"[{appname}]"
                " {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} |"
                " {message}"
            ),
            colorize=False,
        )
