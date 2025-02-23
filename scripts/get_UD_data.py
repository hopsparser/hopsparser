import pathlib
import re
import shutil
from collections import defaultdict
from typing import Callable, Iterator

import click
import pooch
from rich import progress

UD_URLS = {
    "2.15": {
        "url": "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5787/ud-treebanks-v2.15.tgz",
        "hash": "md5:1ebca6a1cf594ea689c1687a56fbb9d4",
    }
}

UD_RELEASES = list(UD_URLS.keys())


def count_trees(treebank_path: pathlib.Path) -> int:
    # cheesy stuff to make it faster (no joke shaves off a whole minute)
    def _make_gen(reader: Callable[[int], bytes]) -> Iterator[bytes]:
        while True:
            b = reader(2**16)
            if not b:
                break
            yield b

    with open(treebank_path, "rb") as in_stream:
        count = sum(buf.count(b"# sent_id") for buf in _make_gen(in_stream.raw.read))
    return count


@click.group()
def cli():
    pass


@cli.command(help="List available UD releases.")
def avail():
    for release in UD_RELEASES:
        click.echo(release)


@cli.command(help="Download a UD release.")
@click.argument(
    "target_dir", type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path)
)
@click.option("--train-threshold", default=2048, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--release", default=UD_RELEASES[-1], type=click.Choice(UD_RELEASES), show_default=True
)
def download(target_dir: pathlib.Path, release: str, train_threshold: int):
    ud_url = UD_URLS[release]
    downloaded_path = pooch.retrieve(
        known_hash=ud_url["hash"],
        path=target_dir,
        progressbar=True,
        url=ud_url["url"],
    )
    extract_path = target_dir / f"ud-treebanks-v{release}"
    all_corpora_dir = extract_path / "all_corpora"
    if not all_corpora_dir.exists():
        shutil.unpack_archive(downloaded_path, extract_dir=extract_path, filter="data")
        (extract_path / f"ud-treebanks-v{release}").rename(all_corpora_dir)

    train_corpora_path = extract_path / "train_corpora"
    train_corpora_path.mkdir(exist_ok=True)

    test_corpora_dirs: defaultdict[str, list[pathlib.Path]] = defaultdict(list)
    trainable_corpora_dirs: defaultdict[str, list[pathlib.Path]] = defaultdict(list)
    n_direct_train: defaultdict[str, int] = defaultdict(int)
    for p in progress.track(list(all_corpora_dir.glob("*/"))):
        if (m := re.match(r"UD_(?P<lang>.*?)-.*", p.stem)) is None:
            raise ValueError(f"Unsupported dir {p}")

        # These treebanks have no words, let's skip them
        if b"Includes text: no" in next(p.glob("README.*")).read_bytes():
            shutil.rmtree(p)
            continue

        lang = m.group("lang")
        test_corpora_dirs[lang].append(p)
        if (train_file := next(p.glob("*-train.conllu"), None)) is not None:
            trainable_corpora_dirs[lang].append(p)
            if count_trees(train_file) > train_threshold:
                n_direct_train[lang] += 1
                shutil.copytree(p, train_corpora_path / p.stem)

    for lang, corpora in progress.track(trainable_corpora_dirs.items()):
        # If there's only one corpus and its already in train_corpora, skip, this would just be a
        # duplicate.
        if len(corpora) == 1 and n_direct_train[lang] == 1 and len(test_corpora_dirs[lang]) == 1:
            continue
        lang_all_path = train_corpora_path / f"UD_{lang}-ALL"
        lang_all_path.mkdir()
        with open(lang_all_path / f"{lang}_all-ud-train.conllu", "wb") as out_stream:
            for corpus in corpora:
                corpus_train = next(corpus.glob("*-train.conllu"))
                if (corpus_dev := next(corpus.glob("*-dev.conllu"), None)) is not None:
                    shutil.copyfile(corpus_dev, lang_all_path / corpus_dev.name)
                with open(corpus_train, "rb") as in_stream:
                    shutil.copyfileobj(in_stream, out_stream)

        for corpus in test_corpora_dirs[lang]:
            corpus_test = next(corpus.glob("*-test.conllu"))
            shutil.copyfile(corpus_test, lang_all_path / corpus_test.name)


if __name__ == "__main__":
    cli()
