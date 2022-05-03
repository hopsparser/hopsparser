import contextlib
import datetime
import logging
import math
import pathlib
import sys
import tempfile
from typing import IO, Dict, Generator, Optional, Union, cast

from loguru import logger
import rich.text
import rich.progress
import transformers


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
    
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    transformers.utils.logging.disable_default_handler
    transformers.utils.logging.add_handler(InterceptHandler())


def log_epoch(epoch_name: str, metrics: Dict[str, str]):
    metrics_table = "\t".join(f"{k} {v}" for k, v in metrics.items())
    logger.info(f"Epoch {epoch_name}: {metrics_table}")


class SpeedColumn(rich.progress.ProgressColumn):
    def render(self, task: rich.progress.Task) -> rich.text.Text:
        if not task.speed:
            return rich.text.Text("-:--:--")
        if task.speed >= 1:
            return rich.text.Text(f"{task.speed:.2f} it/s")
        else:
            return rich.text.Text(
                f"{datetime.timedelta(seconds=math.ceil(1/task.speed))} /it"
            )
