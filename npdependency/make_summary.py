import pathlib
from typing import Iterable, List, TextIO

import click
import click_pathlib

from npdependency import conll2018_eval as evaluator

CONLL_METRICS = [
    "Tokens",
    "Sentences",
    "Words",
    "UPOS",
    "XPOS",
    "UFeats",
    "AllTags",
    "Lemmas",
    "UAS",
    "LAS",
    "CLAS",
    "MLAS",
    "BLEX",
]


@click.command()
@click.argument(
    "gold_file",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.argument(
    "syst_files",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
    nargs=-1,
)
@click.option(
    "--out_file",
    type=click.File("w"),
    default="-",
)
@click.option("--onlyf", is_flag=True)
@click.option("--metric", "metrics", multiple=True, default=CONLL_METRICS)
def make_csv_summary(
    syst_files: Iterable[pathlib.Path],
    gold_file: pathlib.Path,
    out_file: TextIO,
    onlyf: bool,
    metrics: List[str],
):
    gold_conllu = evaluator.load_conllu_file(gold_file)

    if onlyf:
        header = ["name", *metrics]
    else:
        header = ["name", *(f"{m}_{p}" for m in CONLL_METRICS for p in ("P", "R", "F"))]
    print(",".join(header), file=out_file)
    for syst_file in syst_files:
        syst_conllu = evaluator.load_conllu_file(syst_file)
        eval_metrics = evaluator.evaluate(gold_conllu, syst_conllu)
        row: List[str] = [syst_file.stem]
        for m in metrics:
            mres = eval_metrics[m]
            if onlyf:
                row.append(str(mres.f1))
            else:
                row.extend((str(mres.precision), str(mres.recall), str(mres.f1)))
        print(",".join(row), file=out_file)


if __name__ == "__main__":
    make_csv_summary()
