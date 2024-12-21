import enum
import multiprocessing
import multiprocessing.pool
import pathlib
import shutil
from queue import Queue
from typing import (
    Any,
    Optional,
    cast,
)

import click
import polars as pol
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from pytorch_lightning import callbacks as pl_callbacks
from rich.progress import MofNCompleteColumn, Progress, TaskID, TimeElapsedColumn
from tabulate2 import tabulate

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
    def __init__(self, message_queue: "Queue[tuple[Messages, Any]]", run_name: str):
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
                    k: (f"{v:.08f}" if "loss" in k else f"{v:06.2%}"[:-1])
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


def evaluate_model(
    model_path: pathlib.Path,
    parsed_dir: pathlib.Path,
    treebanks: dict[str, pathlib.Path],
    device: str = "cpu",
    metrics: Optional[list[str]] = None,
    reparse: bool = False,
) -> dict[str, dict[str, float]]:
    logger.debug(f"Evaluating {model_path} on {treebanks}.")
    # Avoid loading the model right now in case we don't need to reparse anything. This removes a
    # security (ensuring the model actually loads) but eh.
    model = None
    res: dict[str, dict[str, float]] = dict()
    for treebank_name, treebank_path in treebanks.items():
        parsed_path = parsed_dir / f"{treebank_path.stem}.parsed.conllu"
        if not parsed_path.exists() or reparse:
            logger.debug(f"Parsing {treebank_path}.")
            if model is None:
                logger.debug(f"Loading {model_path} for evaluation.")
                model = parser.BiAffineParser.load(model_path).to(device)
                model.eval()
            with treebank_path.open() as in_stream, parsed_path.open("w") as out_stream:
                with torch.inference_mode():
                    for tree in model.parse(inpt=in_stream, batch_size=None):
                        out_stream.write(tree.to_conllu())
                        out_stream.write("\n\n")
        gold_set = evaluator.load_conllu_file(treebank_path)
        syst_set = evaluator.load_conllu_file(parsed_path)
        eval_res = evaluator.evaluate(gold_set, syst_set)
        if metrics is None:
            res_metrics = list(eval_res.keys())
        else:
            res_metrics = metrics
        res[treebank_name] = {m: eval_res[m].f1 for m in res_metrics}
    return res


def train_single_model(
    additional_args: dict[str, Any],
    config_file: pathlib.Path,
    device: str,
    dev_file: pathlib.Path,
    message_queue: "Queue[tuple[Messages, Any]]",
    metrics: list[str],
    output_dir: pathlib.Path,
    run_name: str,
    test_file: pathlib.Path,
    train_file: pathlib.Path,
) -> dict[str, dict[str, float]]:
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
        **{k: v for k, v in additional_args.items()},
    )

    logger.info(f"Training done for {run_name}, running a full evaluation on the dev-best model.")

    eval_res = evaluate_model(
        device=device,
        metrics=metrics,
        model_path=model_path,
        parsed_dir=output_dir,
        treebanks={"Dev": dev_file, "Test": test_file},
    )

    table = tabulate(
        [{"Split": k, **{m: f"{v[m]:06.2%}"[:-1] for m in metrics}} for k, v in eval_res.items()],
        floatfmt="05.2f",
        headers="keys",
        headersglobalalign="center",
        tablefmt="pipe",
    )
    logger.info(f"Metrics for {run_name}:\n{table}")

    checkpoints_dir = output_dir / "lightning_checkpoints"
    logger.debug(f"Deleting lightning checkpoint at {checkpoints_dir} to save disk space.")
    shutil.rmtree(checkpoints_dir)

    logger.remove(log_handler)

    return {t: {m: r[m] for m in metrics} for t, r in eval_res.items()}


def worker(
    device_queue: "Queue[str]",
    monitor_queue: "Queue[tuple[Messages, Any]]",
    run_name: str,
    train_kwargs: dict[str, Any],
) -> tuple[str, dict[str, dict[str, float]]]:
    # We use no more workers than devices so the queue should never be empty when launching the
    # worker fun so we want to fail early here if the Queue is empty. It does not feel right but it
    # works.
    device = device_queue.get(block=False)
    log_handle = utils.setup_logging(
        appname=f"hops_trainer({run_name})",
        sink=lambda m: monitor_queue.put((Messages.LOG, m)),
    )[0]
    train_kwargs["device"] = device
    logger.info(f"Start training {run_name} on {device}")
    res = train_single_model(**train_kwargs, message_queue=monitor_queue, run_name=run_name)
    logger.info(f"Run {run_name} finished.")
    monitor_queue.put((Messages.RUN_DONE, run_name))
    device_queue.put(device)
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


# TODO: allow simulating multiprocessing for debugging purposes?
def run_multi(
    runs: dict[str, dict[str, Any]],
    devices: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
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
                (
                    (device_queue, monitor_queue, run_name, run_args)
                    for run_name, run_args in runs.items()
                ),
            )
            res = res_future.get()
        monitor_queue.put((Messages.CLOSE, None))
        monitor.join()
        monitor.close()
    return {run_name: run_res for run_name, run_res in res}


# TODO: use a dict for queue content
def monitor_process(num_runs: int, queue: "Queue[tuple[Messages, Any]]"):
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
    "--devices",
    default="cpu",
    callback=(lambda _ctx, _opt, val: val.split(",")),
    help="A comma-separated list of devices to run on.",
    metavar="STR,...",
    show_default=True,
)
@click.option(
    "--metrics",
    default="UPOS,UAS,LAS",
    callback=(lambda _ctx, _opt, val: val.split(",")),
    help=(
        "A comma-separated list of metrics to use for evaluation."
        " The last one will be use to select dev-best models."
    ),
    metavar="STR,...",
    show_default=True,
)
@click.option(
    "--out-dir",
    default=pathlib.Path.cwd(),
    type=click.Path(resolve_path=True, exists=False, file_okay=False, path_type=pathlib.Path),
)
@click.option("--prefix", default="", help="A custom prefix to prepend to run names.")
@click.option(
    "--rand-seeds",
    callback=(lambda _ctx, _opt, val: [int(v) for v in val.split(",")]),
    default="0",
    help=(
        "A comma-separated list of random seeds to try and run stats on."
        " Only the seed with the best result will be kept for every running config."
    ),
    metavar="INT,...",
    show_default=True,
)
def main(
    configs_dir: pathlib.Path,
    devices: list[str],
    metrics: list[str],
    out_dir: pathlib.Path,
    prefix: str,
    rand_seeds: list[int],
    treebanks_dir: pathlib.Path,
):
    logger.remove()
    logging_handlers = utils.setup_logging(appname="hops_trainer")
    out_dir.mkdir(parents=True, exist_ok=True)
    treebanks = [train.parent for train in treebanks_dir.glob("**/*train.conllu")]
    logger.info(f"Training on {len(treebanks)} treebanks.")
    configs = list(configs_dir.glob("**/*.yaml"))
    logger.info(f"Training using {len(configs)} configs.")
    logger.info(f"Training with rand seeds: {','.join(str(s) for s in rand_seeds)}.")

    runs: dict[str, dict] = dict()
    for t in treebanks:
        for c in configs:
            # TODO: logic for treebank dirs that don't have a test etc. Doesn't happen in UD (train
            # implies dev and test but could in other contexts)
            train_file = next(t.glob("*train.conllu"))
            dev_file = next(t.glob("*dev.conllu"))
            test_file = next(t.glob("*test.conllu"))
            # TODO: make this cleaner
            # Skip configs that are not for this lang
            # UD treebank files start with the iso langcode (like en_ewt, etc.)
            if c.parent != configs_dir and not train_file.stem.startswith(c.parent.name):
                continue
            common_params = {
                "train_file": train_file,
                "dev_file": dev_file,
                "test_file": test_file,
                "config_file": c,
                "metrics": metrics,
            }
            run_base_name = f"{prefix}{t.name}-{c.stem}"
            run_out_root_dir = out_dir / run_base_name
            for r in rand_seeds:
                run_suffix = f"rand_seed={r}"
                run_out_dir = run_out_root_dir / run_suffix
                run_name = f"{run_base_name}+{run_suffix}"
                run_args = {
                    **common_params,
                    "additional_args": {"rand_seed": r},
                    "output_dir": run_out_dir,
                }
                runs[run_name] = run_args

    missing_runs: dict[str, dict[str, Any]] = {}
    res: dict[str, dict[str, dict[str, float]]] = {}
    for run_name, run_args in runs.items():
        if (run_out_dir := run_args["output_dir"]).exists():
            # FIXME: the logic here is brittle and feels redundant too
            parsed_dev = next(run_out_dir.glob("*dev.parsed.conllu"), None)
            parsed_test = next(run_out_dir.glob("*test.parsed.conllu"), None)

            if parsed_dev is not None and parsed_test is not None:
                try:
                    # FIXME: don't hardcode the model path this way?
                    prev_metrics = evaluate_model(
                        metrics=metrics,
                        model_path=run_out_dir / "model",
                        parsed_dir=run_out_dir,
                        treebanks={"Dev": run_args["dev_file"], "Test": run_args["test_file"]},
                        reparse=False,
                    )
                except evaluator.UDError as e:
                    raise ValueError(f"Corrupted parsed file for {run_out_dir}") from e

                res[run_name] = prev_metrics
                logger.info(
                    f"{run_out_dir} already exists, skipping run {run_name}."
                    f" Results were {prev_metrics}"
                )
            else:
                logger.warning(
                    f"Incomplete run in {run_out_dir}, skipping it."
                    " You probably want to delete it and rerun."
                )
        else:
            missing_runs[run_name] = run_args

    logger.info(f"Starting {len(missing_runs)} runs.")
    for h in logging_handlers:
        logger.remove(h)
    new_res = run_multi(missing_runs, devices)
    utils.setup_logging(appname="hops_trainer")
    logger.info("Done with training")
    res.update(new_res)

    train_results_df = pol.from_records(
        [
            {
                "run_name": name,
                "additional_args": runs[name]["additional_args"],
                "config": str(runs[name]["config_file"]),
                "output_dir": str(runs[name]["output_dir"]),
                "results": scores,
                "treebank": runs[name]["train_file"].parent.name,
            }
            for name, scores in res.items()
        ]
    )
    train_results_df.write_ndjson(out_dir / "full_report.jsonl")

    best_df = (
        train_results_df.group_by("treebank")
        .agg(pol.all().top_k_by(pol.col("results").struct["Dev"].struct[metrics[-1]], 1))
        .explode(pol.all().exclude("treebank"))
        .sort(pol.col("run_name"))
    )
    best_dir = out_dir / "best"
    best_dir.mkdir(exist_ok=True, parents=True)
    (best_dir / "models.md").write_text(
        tabulate(
            (
                best_df.select(
                    pol.col("output_dir"),
                    pol.col("run_name"),
                    pol.col("treebank").alias("Treebank"),
                    pol.col("run_name").str.splitn("+", n=2).struct[0].alias("Model Name"),
                    *(
                        (
                            pol.col("results").struct[split].struct[metric].round_sig_figs(4) * 100
                        ).alias(f"{split} {metric}")
                        for split in ("Dev", "Test")
                        for metric in metrics
                    ),
                )
                .with_columns(pol.format("[link][{}]", pol.col("Model Name")).alias("Download"))
                .select(pol.all().exclude("output_dir", "run_name"))
                .to_dict(as_series=False)
            ),
            floatfmt="05.2f",
            headers="keys",
            headersglobalalign="center",
            tablefmt="pipe",
        )
    )

    for report in best_df.iter_rows(named=True):
        shutil.copytree(report["output_dir"], best_dir / report["run_name"], dirs_exist_ok=True)


if __name__ == "__main__":
    main()
