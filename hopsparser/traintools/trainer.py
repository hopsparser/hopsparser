import datetime
import os
import pathlib
import shutil
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence, cast

import click
import pydantic
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchmetrics
import transformers
import yaml
from lightning.pytorch.utilities.types import LRSchedulerConfigType
from lightning_utilities.core.rank_zero import rank_zero_only
from loguru import logger
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from rich import box
from rich.console import Console
from rich.table import Column, Table

from hopsparser import evaluator
from hopsparser import utils
from hopsparser.deptree import DepGraph
from hopsparser.parser import (
    BiAffineParser,
    BiaffineParserOutput,
    DependencyBatch,
    DependencyDataset,
    LRSchedule,
    parse,
)
from hopsparser.utils import setup_logging


# TODO: in Python 3.12+, this could be typed much more elegantly:
# <https://typing.readthedocs.io/en/latest/spec/generics.html#user-defined-generic-classes>
class HarmonicMeanAggregateMetric(torchmetrics.Metric):
    """An aggregator of metric replicas that returns their harmonic mean. Only works for metrics
    returning their output as a torch Tensor."""

    is_differentiable: bool | None = False

    def __init__(
        self,
        n_datasets: int,
        metric_class: type[torchmetrics.Metric],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.wrapped_metrics = cast(
            Sequence[torchmetrics.Metric],
            torch.nn.ModuleList([metric_class(*args, **kwargs) for _ in range(n_datasets)]),
        )

    def update(self, dataset_idx: int, *args, **kwargs) -> None:
        self.wrapped_metrics[dataset_idx].update(*args, **kwargs)

    def _agg(self, values: list[torch.Tensor]) -> torch.Tensor:
        if len(values) == 1:
            return values[0]
        with torch.inference_mode():
            # This can result in inf values. It's OK.
            return torch.stack(values, dim=0).reciprocal().nanmean(dim=0).reciprocal()

    def forward(self, dataset_idx: int, *args, **kwargs) -> torch.Tensor:
        """Call underlying forward methods for the targeted metric. The return value does not
        necessarily make a lot of sense."""
        # This method is overridden because we do not need the complex version defined in Metric,
        # that relies on the value of full_state_update, and that also accumulates the results.
        # Here, all computations are handled by the underlying metrics, which all have their own
        # value of full_state_update, and which all accumulate the results by themselves.
        return self.wrapped_metrics[dataset_idx](*args, **kwargs)

    def compute(self) -> torch.Tensor:
        """Compute metrics for all tasks."""
        return self._agg([m.compute() for m in self.wrapped_metrics])

    def reset(self) -> None:
        """Reset all underlying metrics."""
        for metric in self.wrapped_metrics:
            metric.reset()
        super().reset()


class TrainConfig(pydantic.BaseModel):
    batch_size: int = 1
    epochs: int
    lr: LRSchedule


class ParserTrainingModuleForwardOutput(NamedTuple):
    output: BiaffineParserOutput
    loss: torch.Tensor


class ParserTrainingModule(pl.LightningModule):
    def __init__(self, parser: BiAffineParser, config: TrainConfig, n_dev: int = 1):
        super().__init__()
        self.parser = parser
        self.config = config

        self.val_loss = HarmonicMeanAggregateMetric(
            n_datasets=n_dev,
            metric_class=torchmetrics.MeanMetric,
        )

        self.val_tags_accuracy = HarmonicMeanAggregateMetric(
            n_datasets=n_dev,
            metric_class=torchmetrics.Accuracy,
            ignore_index=self.parser.LABEL_PADDING,
            num_classes=len(self.parser.tagset),
            task="multiclass",
        )

        self.val_heads_accuracy = HarmonicMeanAggregateMetric(
            n_datasets=n_dev, metric_class=torchmetrics.MeanMetric
        )
        self.val_deprel_accuracy = HarmonicMeanAggregateMetric(
            n_datasets=n_dev,
            metric_class=torchmetrics.Accuracy,
            ignore_index=self.parser.LABEL_PADDING,
            num_classes=len(self.parser.labels),
            task="multiclass",
        )
        # We need a cast here because ModuleDict does not subclass dict
        self.val_extra_labels_accuracy = cast(
            Mapping[str, torchmetrics.Metric],
            torch.nn.ModuleDict({
                name: HarmonicMeanAggregateMetric(
                    n_datasets=n_dev,
                    metric_class=torchmetrics.Accuracy,
                    ignore_index=self.parser.LABEL_PADDING,
                    num_classes=len(lex),
                    task="multiclass",
                )
                for name, lex in self.parser.annotation_lexicons.items()
            }),
        )
        logger.debug(f"Using train config {config}")
        self.save_hyperparameters("config")

    def forward(self, batch: DependencyBatch) -> ParserTrainingModuleForwardOutput:
        output = self.parser(batch.sentences.encodings, batch.sentences.sent_lengths)
        loss = self.parser.parser_loss(output, batch)
        return ParserTrainingModuleForwardOutput(output=output, loss=loss)

    def training_step(self, batch: DependencyBatch, batch_idx: int) -> torch.Tensor:
        output: ParserTrainingModuleForwardOutput = self(batch)

        self.log(
            "train/loss",
            output.loss,
            batch_size=batch.sentences.sent_lengths.shape[0],
            logger=True,
            on_epoch=True,
            reduce_fx=torch.mean,
            sync_dist=True,
        )

        return output.loss

    # This should be done in the validation_step hook but see
    # <https://github.com/Lightning-AI/pytorch-lightning/issues/11126#issuecomment-1504866597>
    def on_validation_epoch_end(self):
        self.log(
            "validation/loss",
            self.val_loss,
            logger=True,
            on_epoch=True,
        )

        self.log(
            "validation/tags_accuracy",
            self.val_tags_accuracy,
            logger=True,
            on_epoch=True,
        )

        self.log(
            "validation/heads_accuracy",
            self.val_heads_accuracy,
            logger=True,
            on_epoch=True,
        )

        self.log(
            "validation/deprel_accuracy",
            self.val_deprel_accuracy,
            logger=True,
            on_epoch=True,
        )

        for name, metric in self.val_extra_labels_accuracy.items():
            self.log(
                f"validation/{name}_accuracy",
                metric,
                logger=True,
                on_epoch=True,
            )

    def validation_step(self, batch: DependencyBatch, batch_idx: int, dataloader_idx: int = 0):
        output: ParserTrainingModuleForwardOutput = self(batch)

        self.val_loss(dataloader_idx, output.loss)

        tags_pred = output.output.tag_scores.argmax(dim=-1)
        self.val_tags_accuracy(dataloader_idx, tags_pred, batch.tags)

        # greedy head accuracy (without parsing)
        heads_preds = output.output.head_scores.argmax(dim=-1)
        heads_mask = batch.heads.ne(self.parser.LABEL_PADDING)
        n_heads = heads_mask.sum()
        heads_accuracy = (
            heads_preds.eq(batch.heads).logical_and(heads_mask).sum().true_divide(n_heads)
        )
        self.val_heads_accuracy(dataloader_idx, heads_accuracy, n_heads)

        # greedy deprel accuracy (without parsing)
        gold_heads_select = (
            # We need a non-negatif index for gather
            batch.heads.masked_fill(batch.heads.eq(self.parser.LABEL_PADDING), 0)
            # we need to unsqueeze before expanding
            .view(batch.heads.shape[0], batch.heads.shape[1], 1, 1)
            # For every head, we will select the score for all labels
            .expand(
                batch.heads.shape[0],
                batch.heads.shape[1],
                1,
                output.output.deprel_scores.shape[-1],
            )
        )
        # shape: num_padded_deps×num_padded_heads
        gold_head_deprels_scores = torch.gather(
            output.output.deprel_scores, -2, gold_heads_select
        ).squeeze(-2)
        deprels_pred = gold_head_deprels_scores.argmax(dim=-1)
        self.val_deprel_accuracy(dataloader_idx, deprels_pred, batch.labels)

        # extra labels accuracy
        for name, scores in output.output.extra_labels_scores.items():
            gold_annotation = batch.annotations[name]
            annotation_pred = scores.argmax(dim=-1)
            self.val_extra_labels_accuracy[name](dataloader_idx, annotation_pred, gold_annotation)

    def configure_optimizers(self):  # type: ignore[override]
        # TODO: use modern Adam/other opts and allow tweaking the betas
        optimizer = torch.optim.Adam(
            self.parameters(), betas=(0.9, 0.9), lr=self.config.lr.base, eps=1e-09, fused=True
        )

        schedulers: list[LRSchedulerConfigType]
        if self.config.lr.shape == "exponential":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                lr_lambda=(lambda n: 0.95**n),
                optimizer=optimizer,
            )
            schedulers = [{"scheduler": scheduler, "interval": "epoch"}]
        elif self.config.lr.shape == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_warmup_steps=self.config.lr.warmup_steps,
                optimizer=optimizer,
            )
            schedulers = [{"scheduler": scheduler, "interval": "step"}]
        elif self.config.lr.shape == "constant":
            scheduler = transformers.get_constant_schedule_with_warmup(
                num_warmup_steps=self.config.lr.warmup_steps,
                optimizer=optimizer,
            )
            schedulers = [{"scheduler": scheduler, "interval": "step"}]
        else:
            raise ValueError(f"Unkown lr schedule shape {self.config.lr.shape!r}")
        return [optimizer], schedulers


class SaveModelCallback(pl.Callback):
    def __init__(self, save_dir: pathlib.Path):
        self.save_dir = save_dir

    @rank_zero_only
    def on_save_checkpoint(  # type: ignore[override]
        self, trainer: pl.Trainer, pl_module: ParserTrainingModule, checkpoint: dict[str, Any]
    ):
        logger.info(f"Saving model to {self.save_dir}")
        pl_module.parser.save(self.save_dir)


class RichFeedbackCallback(pl_callbacks.Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.sanity_checking:
            utils.log_epoch(
                epoch_name=str(trainer.current_epoch),
                metrics={
                    k: (f"{v:.08f}" if "loss" in k else f"{v:06.2%}"[:-1])
                    for k, v in trainer.logged_metrics.items()
                },
            )


def train(
    accelerator: str,
    config_file: pathlib.Path,
    dev_file: pathlib.Path | list[pathlib.Path] | None,
    output_dir: pathlib.Path,
    rand_seed: int,
    run_name: str,
    train_file: pathlib.Path,
    devices: int | str | list[int] = 0,
    callbacks: Optional[Iterable[pl_callbacks.Callback]] = None,
    overwrite: bool = False,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    # TODO: rollback in case of failure?
    shutil.copy(config_file, output_dir / config_file.name)
    model_path = output_dir / "model"

    logger.info(f"Using random seed {rand_seed}")
    pl.seed_everything(rand_seed, workers=True)
    transformers.set_seed(rand_seed)

    with open(config_file) as in_stream:
        hp = yaml.load(in_stream, Loader=yaml.SafeLoader)

    train_config = TrainConfig.model_validate(hp)

    with open(train_file) as in_stream:
        train_trees = list(DepGraph.read_conll(in_stream))

    if model_path.exists():
        if overwrite:
            logger.info(
                f"Erasing existing trained model in {model_path} since overwrite was asked",
            )
            shutil.rmtree(model_path)
            parser = BiAffineParser.initialize(
                config_path=config_file,
                treebank=train_trees,
            )

        else:
            logger.info(f"Continuing training from {model_path}")
            parser = BiAffineParser.load(model_path)
    else:
        if overwrite:
            logger.warning(f"Overwrite asked but {model_path} does not exist or is empty")
        parser = BiAffineParser.initialize(
            config_path=config_file,
            treebank=train_trees,
        )

    parser.save(output_dir / "model")
    train_set = DependencyDataset(
        parser,
        train_trees,
        skip_unencodable=True,
    )
    # We need this since we use multiple workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_config.batch_size,
        collate_fn=parser.batch_trees,
        shuffle=True,
        num_workers=2,
    )
    dev_loaders: list[torch.utils.data.DataLoader] = []
    if dev_file is not None:
        if isinstance(dev_file, pathlib.Path):
            dev_file = [dev_file]
        for f in dev_file:
            with f.open() as in_stream:
                dev_trees = list(DepGraph.read_conll(in_stream))
            dev_set = DependencyDataset(
                parser,
                dev_trees,
                skip_unencodable=True,
            )
            dev_loaders.append(
                torch.utils.data.DataLoader(
                    dataset=dev_set,
                    batch_size=train_config.batch_size,
                    collate_fn=parser.batch_trees,
                    num_workers=2,
                )
            )
    train_module = ParserTrainingModule(config=train_config, n_dev=len(dev_loaders), parser=parser)
    all_callbacks = [
        pl_callbacks.LearningRateMonitor("step"),
        SaveModelCallback(save_dir=model_path),
    ]
    if callbacks is not None:
        all_callbacks.extend(callbacks)
    # Resist the urge to factor this code. Resist it. Good.
    if len(dev_loaders) > 0:
        all_callbacks.append(
            pl_callbacks.ModelCheckpoint(
                auto_insert_metric_name=False,
                dirpath=output_dir / "lightning_checkpoints",
                filename="epoch={epoch}-dev_heads_acc={validation/heads_accuracy:06.2%}",
                mode="max",
                monitor="validation/heads_accuracy",
                save_top_k=1,
            ),
        )
    else:
        all_callbacks.append(
            pl_callbacks.ModelCheckpoint(
                auto_insert_metric_name=False,
                dirpath=output_dir / "lightning_checkpoints",
                filename="epoch={epoch}-train_loss={train/loss:.6f}",
                mode="max",
                monitor="train/loss",
                save_top_k=1,
            ),
        )
    loggers = [
        CSVLogger(output_dir, version=run_name),
        TensorBoardLogger(output_dir, version=run_name),
    ]
    trainer = pl.Trainer(
        accelerator=accelerator,
        callbacks=all_callbacks,
        default_root_dir=output_dir,
        devices=devices,
        enable_progress_bar=any(isinstance(c, pl_callbacks.ProgressBar) for c in all_callbacks),
        log_every_n_steps=128,
        logger=loggers,
        max_epochs=train_config.epochs,
    )
    trainer.fit(train_module, train_dataloaders=train_loader, val_dataloaders=dev_loaders)


@click.command(help="Train a parsing model")
@click.argument(
    "config_file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "train_file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "output_dir",
    type=click.Path(resolve_path=True, file_okay=False, writable=True, path_type=pathlib.Path),
)
@click.option(
    "--accelerator",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", *AcceleratorRegistry.available_accelerators()]),
    help="The type of devices to use (see pytorch-lightning's doc for details).",
)
@click.option(
    "--dev-file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
    help="A CoNLL-U treebank to use as a development dataset.",
)
@click.option(
    "--run-name",
    default=datetime.datetime.now().replace(microsecond=0).isoformat(),
    help="The name of the experiment. Defaults to the current datetime.",
)
@click.option(
    "--rand-seed",
    default=0,
    help=(
        "Force the random seed for Python and Pytorch (see"
        " <https://pytorch.org/docs/stable/notes/randomness.html> for notes on reproducibility)"
    ),
    show_default=True,
    type=int,
)
@click.option(
    "--test-file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
    help="A CoNLL-U treebank to use as a development dataset.",
)
def main(
    accelerator: str,
    dev_file: Optional[pathlib.Path],
    config_file: pathlib.Path,
    run_name: str,
    output_dir: pathlib.Path,
    rand_seed: int,
    test_file: pathlib.Path,
    train_file: pathlib.Path,
):
    setup_logging()
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / "model"
    shutil.copy(config_file, output_dir / config_file.name)
    train(
        accelerator=accelerator,
        dev_file=dev_file,
        devices=[0],
        config_file=config_file,
        run_name=run_name,
        output_dir=output_dir,
        train_file=train_file,
        rand_seed=rand_seed,
        callbacks=[
            pl_callbacks.RichProgressBar(console_kwargs={"stderr": True}),
            RichFeedbackCallback(),
        ],
    )

    metrics = ("UPOS", "UAS", "LAS")
    metrics_table = Table(
        "Split",
        *(Column(header=m, justify="center") for m in metrics),
        box=box.HORIZONTALS,
        title="Evaluation metrics",
    )

    if dev_file is not None:
        parsed_devset_path = output_dir / f"{dev_file.stem}.parsed.conllu"
        parse(model_path, dev_file, parsed_devset_path)
        gold_devset = evaluator.load_conllu_file(dev_file)
        syst_devset = evaluator.load_conllu_file(parsed_devset_path)
        dev_metrics = evaluator.evaluate(gold_devset, syst_devset)
        metrics_table.add_row("Dev", *(f"{100 * dev_metrics[m].f1:.2f}" for m in metrics))

    if test_file is not None:
        parsed_testset_path = output_dir / f"{test_file.stem}.parsed.conllu"
        parse(model_path, test_file, parsed_testset_path)
        gold_testset = evaluator.load_conllu_file(test_file)
        syst_testset = evaluator.load_conllu_file(parsed_testset_path)
        test_metrics = evaluator.evaluate(gold_testset, syst_testset)
        metrics_table.add_row("Test", *(f"{100 * test_metrics[m].f1:.2f}" for m in metrics))

    if metrics_table.rows:
        console = Console()
        console.print(metrics_table)


if __name__ == "__main__":
    main()
