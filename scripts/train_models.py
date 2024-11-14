import enum
import itertools
import json
import multiprocessing
import multiprocessing.pool
import os.path
import pathlib
import shutil
from ast import literal_eval
from io import StringIO
from queue import Queue
from typing import (
    Any,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import click
import pytorch_lightning as pl
import polars as pol
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from pytorch_lightning import callbacks as pl_callbacks
from rich import box
from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress, TaskID, TimeElapsedColumn
from rich.table import Table

import hopsparser.traintools.trainer as trainer
from hopsparser import conll2018_eval as evaluator
from hopsparser import parser, utils


class Messages(enum.Enum):
    CLOSE = enum.auto()
    LOG = enum.auto()
    RUN_DONE = enum.auto()
    RUN_START = enum.auto()
    PROGRESS = enum.auto()


class EpochFeedbackCallback(pl_callbacks.Callback):
    def __init__(self, message_queue: Queue, run_name: str):
        self.message_queue = message_queue
        self.run_name = run_name

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.message_queue.put(
            (Messages.RUN_START, (self.run_name, trainer.estimated_stepping_batches))
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        self.message_queue.put(
            (
                Messages.PROGRESS,
                (
                    self.run_name,
                    (None, trainer.global_step),
                ),
            )
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.message_queue.put(
            (
                Messages.PROGRESS,
                (
                    self.run_name,
                    (
                        f"{self.run_name}: train {trainer.current_epoch+1}/{trainer.max_epochs}",
                        None,
                    ),
                ),
            )
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.sanity_checking:
            utils.log_epoch(
                epoch_name=str(trainer.current_epoch),
                metrics={
                    k: (f"{v:.08f}" if "loss" in k else f"{v:06.2%}")
                    for k, v in trainer.logged_metrics.items()
                },
            )
            # TODO: a validation progress bar would be nice
            self.message_queue.put(
                (
                    Messages.PROGRESS,
                    (
                        self.run_name,
                        (
                            f"{self.run_name}: eval {trainer.current_epoch+1}/{trainer.max_epochs}",
                            None,
                        ),
                    ),
                )
            )


class TrainResults(NamedTuple):
    dev_upos: float
    dev_las: float
    test_upos: float
    test_las: float


def train_single_model(
    additional_args: dict[str, str],
    config_file: pathlib.Path,
    device: str,
    dev_file: pathlib.Path,
    message_queue: Queue,
    output_dir: pathlib.Path,
    run_name: str,
    test_file: pathlib.Path,
    train_file: pathlib.Path,
) -> TrainResults:
    output_dir.mkdir(exist_ok=True, parents=True)
    log_handler = logger.add(
        output_dir / "train.log",
        level="DEBUG",
        format="[hops] {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        colorize=False,
    )
    model_path = output_dir / "model"
    shutil.copy(config_file, output_dir / config_file.name)
    # TODO: allow multi-device training? seems overkill for now but
    device_info = torch.device(device)
    accelerator = device_info.type
    devices = 1 if accelerator == "cpu" else [cast(int, device_info.index)]
    trainer.train(
        accelerator=accelerator,
        callbacks=[EpochFeedbackCallback(message_queue=message_queue, run_name=run_name)],
        config_file=config_file,
        dev_file=dev_file,
        devices=devices,
        output_dir=output_dir,
        run_name=run_name,
        train_file=train_file,
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
        metrics_table.add_row("Test", *(f"{100*test_metrics[m].f1:.2f}" for m in metrics))

    if metrics_table.rows:
        out = Console(file=StringIO())
        out.print(metrics_table)
        logger.info(f"Metrics for {run_name}\n{cast(StringIO, out.file).getvalue()}")

    logger.remove(log_handler)

    return TrainResults(
        dev_upos=dev_metrics["UPOS"].f1,
        dev_las=dev_metrics["LAS"].f1,
        test_upos=test_metrics["UPOS"].f1,
        test_las=test_metrics["LAS"].f1,
    )


def worker(
    device_queue: Queue,
    monitor_queue: Queue,
    run_name: str,
    train_kwargs: dict[str, Any],
) -> Tuple[str, TrainResults]:
    # We use no more workers than devices so the queue should never be empty when launching the
    # worker fun so we want to fail early here if the Queue is empty. It does not feel right but it
    # works.
    device = device_queue.get(block=False)
    # TODO: figure out a way to make the run name bubble up here
    log_handle = utils.setup_logging(
        appname=f"hops_trainer({run_name})",
        sink=lambda m: monitor_queue.put((Messages.LOG, m)),
    )[0]
    train_kwargs["device"] = device
    logger.info(f"Start training {run_name} on {device}")
    res = train_single_model(**train_kwargs, message_queue=monitor_queue, run_name=run_name)
    device_queue.put(device)
    # logger.info(f"Run {name} finished with results {res}")
    monitor_queue.put((Messages.RUN_DONE, run_name))
    logger.remove(log_handle)
    return (run_name, res)


# Currently the easiest way to have a pool of nondemonic process (which we need since they
# themselves will fork their dataloading workers)
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)  # type: ignore


class NoDaemonPool(multiprocessing.pool.Pool):
    @staticmethod
    def Process(ctx, *args, **kwargs):  # noqa: N802
        return NoDaemonProcess(*args, **kwargs)


def run_multi(
    runs: Sequence[Tuple[str, dict[str, Any]]],
    devices: list[str],
) -> list[Tuple[str, TrainResults]]:
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

        with NoDaemonPool(len(devices)) as pool:
            res_future = pool.starmap_async(
                worker,
                ((device_queue, monitor_queue, *r) for r in runs),
            )
            res = res_future.get()
        monitor_queue.put((Messages.CLOSE, None))
        monitor.join()
        monitor.close()
    return res


# TODO: use a dict for queue content
def monitor_process(num_runs: int, queue: multiprocessing.Queue):
    with Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        utils.SpeedColumn(),
        refresh_per_second=1.0,
        speed_estimate_period=1800,
    ) as progress:
        train_task = progress.add_task("Training", total=num_runs)
        ongoing: dict[str, TaskID] = dict()
        while True:
            try:
                msg_type, msg = queue.get()
            except EOFError:
                break

            if msg_type is Messages.CLOSE:
                progress.remove_task(train_task)
                break
            elif msg_type is Messages.LOG:
                progress.console.print(msg, end="")
            elif msg_type is Messages.RUN_DONE:
                progress.advance(train_task)
                progress.remove_task(ongoing[msg])
                ongoing.pop(msg)
            elif msg_type is Messages.RUN_START:
                ongoing[msg[0]] = progress.add_task(msg[0], total=msg[1])
            elif msg_type is Messages.PROGRESS:
                progress.update(ongoing[msg[0]], description=msg[1][0], completed=msg[1][1])
            else:
                raise ValueError("Unknown message")
        logger.complete()


def parse_args_callback(
    _ctx: click.Context,
    _opt: Union[click.Parameter, click.Option],
    val: Optional[list[str]],
) -> Optional[list[Tuple[str, list[str]]]]:
    if val is None:
        return None
    res: list[Tuple[str, list[str]]] = []
    for v in val:
        name, values = v.split("=", maxsplit=1)
        res.append((name, values.split(",")))
    return res


def to_describe(col: str, prefix=""):
    prefix = prefix or f"{col}_"
    return [
        pol.col(col).count().alias(f"{prefix}count"),
        pol.col(col).is_null().sum().alias(f"{prefix}null_count"),
        pol.col(col).mean().alias(f"{prefix}mean"),
        pol.col(col).std().alias(f"{prefix}std"),
        pol.col(col).min().alias(f"{prefix}min"),
        pol.col(col).quantile(0.25).alias(f"{prefix}25%"),
        pol.col(col).quantile(0.5).alias(f"{prefix}50%"),
        pol.col(col).quantile(0.75).alias(f"{prefix}75%"),
        pol.col(col).max().alias(f"{prefix}max"),
    ]


@click.command()
@click.argument(
    "configs_dir",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "treebanks_dir",
    type=click.Path(resolve_path=True, exists=True, file_okay=False, path_type=pathlib.Path),
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
    default=pathlib.Path.cwd(),
    type=click.Path(resolve_path=True, exists=False, file_okay=False, path_type=pathlib.Path),
)
@click.option("--prefix", default="", help="A custom prefix to prepend to run names.")
# FIXME: we should have to manually set the default like this, see how to uncomment
@click.option(
    "--rand-seeds",
    callback=(
        lambda _ctx, _opt, val: [0] if val is None else [int(v) for v in val.split(",") if v]
    ),
    # default=[0],
    help=(
        "A comma-separated list of random seeds to try and run stats on."
        " Only the seed with the best result will be kept for every running config."
        "[default: 0]"
    ),
    metavar="INT,...",
    # show_default=True,
)
def main(
    args: Optional[list[Tuple[str, list[str]]]],
    configs_dir: pathlib.Path,
    devices: list[str],
    out_dir: pathlib.Path,
    prefix: str,
    rand_seeds: Optional[list[int]],
    treebanks_dir: pathlib.Path,
):
    logger.remove()
    logging_handlers = utils.setup_logging(appname="hops_trainer")
    out_dir.mkdir(parents=True, exist_ok=True)
    treebanks = [train.parent for train in treebanks_dir.glob("**/*train.conllu")]
    logger.info(f"Training on {len(treebanks)} treebanks.")
    configs = list(configs_dir.glob("**/*.yaml"))
    logger.info(f"Training using {len(configs)} configs.")
    if rand_seeds is not None:
        args = [
            ("rand_seed", [str(s) for s in rand_seeds]),
            *(args if args is not None else []),
        ]
        logger.info(f"Training with {len(rand_seeds)} rand seeds.")
    additional_args_combinations: list[dict[str, str]]
    if args:
        args_names, all_args_values = map(list, zip(*args))
        additional_args_combinations = [
            dict(zip(args_names, args_values))
            for args_values in itertools.product(*all_args_values)
        ]
    else:
        args_names = []
        additional_args_combinations = [dict()]
    runs: list[Tuple[str, dict[str, Any]]] = []
    runs_dict: dict[str, dict] = dict()
    skipped_res: list[Tuple[str, TrainResults]] = []
    for t in treebanks:
        for c in configs:
            train_file = next(t.glob("*train.conllu"))
            dev_file = next(t.glob("*dev.conllu"))
            test_file = next(t.glob("*test.conllu"))
            # TODO: make this cleaner
            # Skip configs that are not for this lang
            if (
                c.parent != configs_dir
                and c.parent.name != "*"
                and not train_file.stem.startswith(c.parent.name)
            ):
                continue
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
                            raise ValueError(f"Corrupted parsed dev file for {run_out_dir}") from e

                        try:
                            gold_testset = evaluator.load_conllu_file(test_file)
                            syst_testset = evaluator.load_conllu_file(parsed_test)
                            test_metrics = evaluator.evaluate(gold_testset, syst_testset)
                        except evaluator.UDError as e:
                            raise ValueError(f"Corrupted parsed test file for {run_out_dir}") from e

                        skip_res = TrainResults(
                            dev_upos=dev_metrics["UPOS"].f1,
                            dev_las=dev_metrics["LAS"].f1,
                            test_upos=test_metrics["UPOS"].f1,
                            test_las=test_metrics["LAS"].f1,
                        )
                        skipped_res.append((run_name, skip_res))
                        logger.info(
                            f"{run_out_dir} already exists, skipping run {run_name}."
                            f" Results were {skip_res}"
                        )
                        continue
                    else:
                        logger.warning(
                            f"Incomplete run in {run_out_dir}, skipping it."
                            " You will probably want to delete it and rerun."
                        )
                        continue

                runs.append((run_name, run_args))

    logger.info(f"Starting {len(runs)} runs.")
    for h in logging_handlers:
        logger.remove(h)
    res = run_multi(runs, devices)
    utils.setup_logging(appname="hops_trainer")
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
        df_dicts = [
            {
                "run_name": run_name,
                **{k: v for k, v in run_report.items() if k not in ("additional_args", "results")},
                **run_report["additional_args"],
                **run_report["results"],
            }
            for run_name, run_report in report_dict.items()
        ]
        df = pol.from_records(df_dicts)
        df.write_csv(out_dir / "full_report.csv")
        summary_columns = [
            "dev_upos",
            "dev_las",
            "test_upos",
            "test_las",
        ]
        df.group_by("treebank").agg(
            *(c for col in summary_columns for c in to_describe(col))
        ).write_csv(summary_file)
        best_dir = out_dir / "best"
        best_dir.mkdir(exist_ok=True, parents=True)
        with open(best_dir / "models.md", "w") as out_stream:
            out_stream.write(
                "| Model name | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download |\n"
                "|:-----------|:----------:|:---------:|:-----------:|:----------:|:--------:|\n"
            )
            for report in (
                df.group_by("treebank")
                .agg(pol.all().top_k_by("dev_las", 1))
                .explode(pol.all().exclude("treebank"))
                .sort(pol.col("run_name"))
                .iter_rows(named=True)
            ):
                shutil.copytree(
                    report["output_dir"], best_dir / report["run_name"], dirs_exist_ok=True
                )
                model_name = report["run_name"].split("+", maxsplit=1)[0]
                out_stream.write("| ")

                out_stream.write(
                    " | ".join(
                        [
                            model_name,
                            *(f"{100*report[v]:.2f}" for v in summary_columns),
                        ]
                    )
                )
                out_stream.write(f" | [link][{model_name}] |\n")


if __name__ == "__main__":
    main()
