import contextlib
import pathlib
import sys
from typing import Generator, cast, IO, Union


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
