import contextlib
import pathlib
import sys
import tempfile
from typing import IO, Generator, Optional, Union, cast
from typing import IO, Generator, Union, cast


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
