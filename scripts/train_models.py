import itertools
import json
import multiprocessing
import os.path
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import click
import pandas as pd

from loguru import logger

from hopsparser import conll2018_eval as evaluator


class TrainResults(NamedTuple):
    dev_upos: float
    dev_las: float
    test_upos: float
    test_las: float


def train_single_model(
    train_file: pathlib.Path,
    dev_file: pathlib.Path,
    test_file: pathlib.Path,
    out_dir: pathlib.Path,
    config_path: pathlib.Path,
    device: str,
    additional_args: Dict[str, str],
) -> TrainResults:
    subprocess.run(
        [
            "hopsparser",
            "train",
            str(config_path),
            str(train_file),
            str(out_dir),
            "--dev-file",
            str(dev_file),
            "--test-file",
            str(test_file),
            "--device",
            device,
            *(
                a
                for key, value in additional_args.items()
                if value
                for a in (f"--{key}", value)
            ),
        ],
        check=True,
    )

    gold_devset = evaluator.load_conllu_file(dev_file)
    syst_devset = evaluator.load_conllu_file(out_dir / f"{dev_file.stem}.parsed.conllu")
    dev_metrics = evaluator.evaluate(gold_devset, syst_devset)

    gold_testset = evaluator.load_conllu_file(test_file)
    syst_testset = evaluator.load_conllu_file(
        out_dir / f"{test_file.stem}.parsed.conllu"
    )
    test_metrics = evaluator.evaluate(gold_testset, syst_testset)

    return TrainResults(
        dev_upos=dev_metrics["UPOS"].f1,
        dev_las=dev_metrics["LAS"].f1,
        test_upos=test_metrics["UPOS"].f1,
        test_las=test_metrics["LAS"].f1,
    )


# It would be nice to be able to have this as a closure, but unfortunately it doesn't work since
# closures are not picklable and multiprocessing can only deal with picklable workers
def worker(device_queue, name, kwargs) -> Tuple[str, TrainResults]:
    # We use no more workers than devices so the queue should never be empty when launching the
    # worker fun so we want to fail early here if the Queue is empty. It does not feel right but it
    # works.
    device = device_queue.get(block=False)
    kwargs["device"] = device
    logger.info(f"Start training {name} on {device}", file=sys.stderr)
    res = train_single_model(**kwargs)
    device_queue.put(device)
    logger.info(f"Run {name} finished with results {res}", file=sys.stderr)
    return (name, res)


def run_multi(
    runs: Iterable[Tuple[str, Dict[str, Any]]], devices: List[str]
) -> List[Tuple[str, TrainResults]]:
    with multiprocessing.Manager() as manager:
        device_queue = manager.Queue()
        for d in devices:
            device_queue.put(d)

        with multiprocessing.Pool(len(devices)) as pool:
            res_future = pool.starmap_async(
                worker,
                ((device_queue, *r) for r in runs),
            )
            res = res_future.get()
    return res


def parse_args_callback(
    _ctx: click.Context,
    _opt: Union[click.Parameter, click.Option],
    val: Optional[List[str]],
) -> Optional[List[Tuple[str, List[str]]]]:
    if val is None:
        return None
    res: List[Tuple[str, List[str]]] = []
    for v in val:
        name, values = v.split("=", maxsplit=1)
        res.append((name, values.split(",")))
    return res


def setup_logging():
    logger.remove(0)  # Remove the default logger
    appname = "hops_trainer"

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


@click.command()
@click.argument(
    "configs_dir",
    type=click.Path(
        resolve_path=True, exists=True, file_okay=False, path_type=pathlib.Path
    ),
)
@click.argument(
    "treebanks_dir",
    type=click.Path(
        resolve_path=True, exists=True, file_okay=False, path_type=pathlib.Path
    ),
)
@click.option(
    "--args",
    multiple=True,
    callback=parse_args_callback,
    help=(
        "An additional list of values for an argument, given as `name=value,value2,…`."
        " Leave a trailing comma to also run the default value of the argument"
        " Can be provided several times for different arguments."
        " Path options should have different file names."
    ),
)
@click.option(
    "--devices",
    default="cpu",
    callback=(lambda _ctx, _opt, val: val.split(",")),
    help="A comma-separated list of devices to run on.",
)
@click.option(
    "--out-dir",
    default=".",
    type=click.Path(
        resolve_path=True, exists=False, file_okay=False, path_type=pathlib.Path
    ),
)
@click.option("--prefix", default="", help="A custom prefix to prepend to run names.")
@click.option(
    "--rand-seeds",
    callback=(
        lambda _ctx, _opt, val: None
        if val is None
        else [int(v) for v in val.split(",") if v]
    ),
    help=(
        "A comma-separated list of random seeds to try and run stats on."
        " Only the seed with the best result will be kept for every running config."
    ),
)
def main(
    args: Optional[List[Tuple[str, List[str]]]],
    configs_dir: pathlib.Path,
    devices: List[str],
    out_dir: pathlib.Path,
    prefix: str,
    rand_seeds: Optional[List[int]],
    treebanks_dir: pathlib.Path,
):
    setup_logging()
    out_dir.mkdir(parents=True, exist_ok=True)
    treebanks = [train.parent for train in treebanks_dir.glob("**/*train.conllu")]
    logger.info(f"Training on {len(treebanks)} treebanks.")
    configs = list(configs_dir.glob("*.yaml"))
    logger.info(f"Training using {len(configs)} configs.")
    if rand_seeds is not None:
        args = [
            ("rand-seed", [str(s) for s in rand_seeds]),
            *(args if args is not None else []),
        ]
        logger.info(f"Training with {len(rand_seeds)} rand seeds.")
    additional_args_combinations: List[Dict[str, str]]
    if args:
        args_names, all_args_values = map(list, zip(*args))
        additional_args_combinations = [
            dict(zip(args_names, args_values))
            for args_values in itertools.product(*all_args_values)
        ]
    else:
        args_names = []
        additional_args_combinations = [{}]
    runs: List[Tuple[str, Dict[str, Any]]] = []
    runs_dict: Dict[str, Dict] = dict()
    skipped_res: List[Tuple[str, TrainResults]] = []
    for t in treebanks:
        for c in configs:
            train_file = next(t.glob("*train.conllu"))
            dev_file = next(t.glob("*dev.conllu"))
            test_file = next(t.glob("*test.conllu"))
            common_params = {
                "train_file": train_file,
                "dev_file": dev_file,
                "test_file": test_file,
                "config_path": c,
            }
            run_base_name = f"{prefix}{t.name}-{c.stem}"
            run_out_root_dir = out_dir / run_base_name
            for additional_args in additional_args_combinations:
                if not additional_args:
                    run_out_dir = run_out_root_dir
                    run_name = run_base_name
                else:
                    args_combination_str = "+".join(
                        f"{n}={os.path.basename(v)}" if v else f"no{n}"
                        for n, v in additional_args.items()
                    )
                    run_out_dir = run_out_root_dir / args_combination_str
                    run_name = f"{run_base_name}+{args_combination_str}"
                run_args = {
                    **common_params,
                    "additional_args": additional_args,
                    "out_dir": run_out_dir,
                }
                runs_dict[run_name] = run_args
                if run_out_dir.exists():
                    parsed_dev = run_out_dir / f"{dev_file.stem}.parsed.conllu"
                    parsed_test = run_out_dir / f"{test_file.stem}.parsed.conllu"

                    if parsed_dev.exists() and parsed_test.exists():
                        gold_devset = evaluator.load_conllu_file(dev_file)
                        syst_devset = evaluator.load_conllu_file(parsed_test)
                        dev_metrics = evaluator.evaluate(gold_devset, syst_devset)

                        gold_testset = evaluator.load_conllu_file(test_file)
                        syst_testset = evaluator.load_conllu_file(parsed_test)
                        test_metrics = evaluator.evaluate(gold_testset, syst_testset)

                        skip_res = TrainResults(
                            dev_upos=dev_metrics["UPOS"].f1,
                            dev_las=dev_metrics["LAS"].f1,
                            test_upos=test_metrics["UPOS"].f1,
                            test_las=test_metrics["LAS"].f1,
                        )
                        skipped_res.append((run_name, skip_res))
                        logger.info(
                            f"{run_out_dir} already exists, skipping run {run_name}. Results were {skip_res}"
                        )
                        continue
                    else:
                        logger.warning(
                            f"Incomplete run in {run_out_dir}, skipping it. You will probably want to delete it and rerun."
                        )
                        continue

                runs.append((run_name, run_args))

    logger.info(f"Starting {len(runs)} runs.")
    res = run_multi(runs, devices)
    res.extend(skipped_res)

    report_file = out_dir / "full_report.json"
    if report_file.exists():
        with open(report_file) as in_stream:
            report_dict = json.load(in_stream)
    else:
        report_dict = dict()
    for name, scores in res:
        run = runs_dict[name]
        report_dict[name] = {
            "additional_args": run["additional_args"],
            "config": str(run["config_path"]),
            "out_dir": str(run["out_dir"]),
            "results": scores._asdict(),
            "treebank": run["train_file"].parent.name,
        }
    with open(report_file, "w") as out_stream:
        json.dump(report_dict, out_stream)

    summary_file = out_dir / "summary.tsv"
    if rand_seeds is None:
        with open(summary_file, "w") as out_stream:
            summary_file.write_text("run\tdev UPOS\tdev LAS\ttest UPOS\ttest LAS\n")
            for name, report in report_dict.items():
                out_stream.write(name)
                for s in ("dev_upos", "dev_las", "test_upos", "test_las"):
                    out_stream.write(f"\t{100*report['results'][s]:.2f}")
                out_stream.write("\n")
    else:
        df_dict = {
            run_name: {
                **{
                    k: v
                    for k, v in run_report.items()
                    if k not in ("additional_args", "results")
                },
                **run_report["additional_args"],
                **run_report["results"],
            }
            for run_name, run_report in report_dict.items()
        }
        df = pd.DataFrame.from_dict(df_dict, orient="index")
        df.to_csv(out_dir / "full_report.csv")
        grouped = df.groupby(
            ["config", "treebank", *(a for a in args_names if a != "rand-seed")],
        )
        grouped[["dev_upos", "dev_las", "test_upos", "test_las"]].describe().to_csv(
            summary_file
        )
        best_dir = out_dir / "best"
        best_dir.mkdir(exist_ok=True, parents=True)
        with open(best_dir / "models.md", "w") as out_stream:
            out_stream.write(
                "| Model name | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download |\n"
                "|:-----------|:----------:|:---------:|:-----------:|:----------:|:--------:|\n"
            )
            for run_name, report in sorted(
                df.loc[grouped["dev_las"].idxmax()].iterrows()
            ):
                shutil.copytree(
                    report["out_dir"], best_dir / run_name, dirs_exist_ok=True
                )
                model_name = run_name.split("+", maxsplit=1)[0]
                out_stream.write("| ")
                out_stream.write(
                    " | ".join(
                        [
                            model_name,
                            *(
                                f"{100*report[v]:.2f}"
                                for v in [
                                    "dev_upos",
                                    "dev_las",
                                    "test_upos",
                                    "test_las",
                                ]
                            ),
                        ]
                    )
                )
                out_stream.write(f" | [link][{model_name}] |\n")


if __name__ == "__main__":
    main()
