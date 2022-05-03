from ast import literal_eval
import enum
from io import StringIO
import itertools
import json
import multiprocessing
import os.path
import pathlib
import shutil
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import click
import pandas as pd

from loguru import logger
from rich import box
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from hopsparser import parser
from hopsparser import conll2018_eval as evaluator


class Messages(enum.Enum):
    CLOSE = enum.auto()
    LOG = enum.auto()
    RUN_DONE = enum.auto()


class TrainResults(NamedTuple):
    dev_upos: float
    dev_las: float
    test_upos: float
    test_las: float


def train_single_model(
    train_file: pathlib.Path,
    dev_file: pathlib.Path,
    test_file: pathlib.Path,
    output_dir: pathlib.Path,
    config_file: pathlib.Path,
    device: str,
    additional_args: Dict[str, str],
) -> TrainResults:
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.add(
        output_dir / "train.log",
        level="DEBUG",
        format=(
            "[hops]"
            " {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} |"
            " {message}"
        ),
        colorize=False,
    )
    model_path = output_dir / "model"
    shutil.copy(config_file, output_dir / config_file.name)
    parser.train(
        config_file=config_file,
        dev_file=dev_file,
        device=device,
        train_file=train_file,
        model_path=model_path,
        **{k: literal_eval(v) for k, v in additional_args.items()},
    )
    metrics_table = Table(box=box.HORIZONTALS)
    metrics_table.add_column("Split")
    metrics = ("UPOS", "UAS", "LAS")
    for m in metrics:
        metrics_table.add_column(m, justify="center")
    if dev_file is not None:
        parsed_devset_path = output_dir / f"{dev_file.stem}.parsed.conllu"
        parser.parse(model_path, dev_file, parsed_devset_path, device=device)
        gold_devset = evaluator.load_conllu_file(dev_file)
        syst_devset = evaluator.load_conllu_file(parsed_devset_path)
        dev_metrics = evaluator.evaluate(gold_devset, syst_devset)
        metrics_table.add_row("Dev", *(f"{100*dev_metrics[m].f1:.2f}" for m in metrics))

    if test_file is not None:
        parsed_testset_path = output_dir / f"{test_file.stem}.parsed.conllu"
        parser.parse(model_path, test_file, parsed_testset_path, device=device)
        gold_testset = evaluator.load_conllu_file(test_file)
        syst_testset = evaluator.load_conllu_file(parsed_testset_path)
        test_metrics = evaluator.evaluate(gold_testset, syst_testset)
        metrics_table.add_row(
            "Test", *(f"{100*test_metrics[m].f1:.2f}" for m in metrics)
        )

    if metrics_table.rows:
        out = Console(file=StringIO())
        out.print(metrics_table)
        logger.info(f"\n{out.file.getvalue()}")

    return TrainResults(
        dev_upos=dev_metrics["UPOS"].f1,
        dev_las=dev_metrics["LAS"].f1,
        test_upos=test_metrics["UPOS"].f1,
        test_las=test_metrics["LAS"].f1,
    )


# It would be nice to be able to have this as a closure, but unfortunately it doesn't work since
# closures are not picklable and multiprocessing can only deal with picklable workers
def worker(device_queue, monitor_queue, name, kwargs) -> Tuple[str, TrainResults]:
    # We use no more workers than devices so the queue should never be empty when launching the
    # worker fun so we want to fail early here if the Queue is empty. It does not feel right but it
    # works.
    device = device_queue.get(block=False)
    setup_logging(lambda m: monitor_queue.put((Messages.LOG, m)), rich_fmt=True)
    kwargs["device"] = device
    logger.info(f"Start training {name} on {device}")
    res = train_single_model(**kwargs)
    device_queue.put(device)
    # logger.info(f"Run {name} finished with results {res}")
    monitor_queue.put((Messages.RUN_DONE, None))
    return (name, res)


def run_multi(
    runs: Sequence[Tuple[str, Dict[str, Any]]],
    devices: List[str],
) -> List[Tuple[str, TrainResults]]:
    with multiprocessing.Manager() as manager:
        device_queue = manager.Queue()
        for d in devices:
            device_queue.put(d)
        monitor_queue = manager.Queue()
        monitor = multiprocessing.Process(
            target=monitor_process,
            kwargs={
                "num_runs": len(runs),
                "queue": monitor_queue,
            },
        )
        monitor.start()

        with multiprocessing.Pool(len(devices)) as pool:
            res_future = pool.starmap_async(
                worker,
                ((device_queue, monitor_queue, *r) for r in runs),
            )
            res = res_future.get()
        monitor_queue.put((Messages.CLOSE, None))
        monitor.join()
        monitor.close()
    return res


def monitor_process(num_runs: int, queue: multiprocessing.Queue):
    with Progress() as progress:
        setup_logging(lambda m: progress.console.print(m, end=""), rich_fmt=True)
        train_task = progress.add_task("Training", total=num_runs)
        while True:
            try:
                msg_type, msg = queue.get()
            except EOFError:
                break
            if msg_type is Messages.CLOSE:
                break
            elif msg_type is Messages.LOG:
                logger.log(msg.record["level"].name, msg.record["message"])
            elif msg_type is Messages.RUN_DONE:
                progress.advance(train_task)
            else:
                raise ValueError("Unknown message")
        logger.complete()


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


def setup_logging(sink=sys.stderr, rich_fmt: bool = False):
    appname = "\\[hops_trainer]" if rich_fmt else "[hops_trainer]"

    log_level = "INFO"
    log_fmt = (
        f"{appname}"
        " <green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
        " <level>{message}</level>"
    )
    # FIXME: I hate this but it's the easiest way
    if rich_fmt:
        log_fmt = log_fmt.replace("<", "[").replace(">", "]")

    return logger.add(
        sink,
        colorize=True,
        enqueue=True,
        format=log_fmt,
        level=log_level,
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
    logger.remove(0)
    logging_handler = setup_logging()
    out_dir.mkdir(parents=True, exist_ok=True)
    treebanks = [train.parent for train in treebanks_dir.glob("**/*train.conllu")]
    logger.info(f"Training on {len(treebanks)} treebanks.")
    configs = list(configs_dir.glob("*.yaml"))
    logger.info(f"Training using {len(configs)} configs.")
    if rand_seeds is not None:
        args = [
            ("rand_seed", [str(s) for s in rand_seeds]),
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
                "config_file": c,
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
                    "output_dir": run_out_dir,
                }
                runs_dict[run_name] = run_args
                if run_out_dir.exists():
                    parsed_dev = run_out_dir / f"{dev_file.stem}.parsed.conllu"
                    parsed_test = run_out_dir / f"{test_file.stem}.parsed.conllu"

                    if parsed_dev.exists() and parsed_test.exists():
                        try:
                            gold_devset = evaluator.load_conllu_file(dev_file)
                            syst_devset = evaluator.load_conllu_file(parsed_dev)
                            dev_metrics = evaluator.evaluate(gold_devset, syst_devset)
                        except evaluator.UDError as e:
                            raise ValueError(
                                f"Corrupted parsed dev file for {run_out_dir}"
                            ) from e

                        try:
                            gold_testset = evaluator.load_conllu_file(test_file)
                            syst_testset = evaluator.load_conllu_file(parsed_test)
                            test_metrics = evaluator.evaluate(
                                gold_testset, syst_testset
                            )
                        except evaluator.UDError as e:
                            raise ValueError(
                                f"Corrupted parsed test file for {run_out_dir}"
                            ) from e

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
    logger.remove(logging_handler)
    res = run_multi(runs, devices)
    setup_logging()
    logger.info("Done with training")
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
            "config": str(run["config_file"]),
            "output_dir": str(run["output_dir"]),
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
            ["config", "treebank", *(a for a in args_names if a != "rand_seed")],
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
                    report["output_dir"], best_dir / run_name, dirs_exist_ok=True
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
