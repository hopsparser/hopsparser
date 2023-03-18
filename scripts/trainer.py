import datetime
import pathlib
import shutil
from typing import Any, Dict, Literal, NamedTuple, Optional, Union

import click
from lightning_utilities.core.rank_zero import rank_zero_only
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pydantic
import torch
import torch.utils.data
import torchmetrics
import transformers
import yaml

from hopsparser.deptree import DepGraph
from hopsparser.parser import (
    BiAffineParser,
    BiaffineParserOutput,
    DependencyBatch,
    DependencyDataset,
    LRSchedule,
)
from hopsparser.utils import setup_logging


class TrainConfig(pydantic.BaseModel):
    batch_size: int = 1
    epochs: int
    lr_schedule: LRSchedule
    lr: float


class ParserTrainingModuleForwardOutput(NamedTuple):
    output: BiaffineParserOutput
    loss: torch.Tensor


class ParserTrainingModule(pl.LightningModule):
    def __init__(self, parser: BiAffineParser, config: TrainConfig):
        super().__init__()
        self.parser = parser
        self.config = config

        self.val_tags_accuracy = torchmetrics.Accuracy(
            ignore_index=self.parser.LABEL_PADDING,
            num_classes=len(self.parser.tagset),
            task="multiclass",
        )
        self.val_heads_accuracy = torchmetrics.MeanMetric()
        self.val_deprel_accuracy = torchmetrics.Accuracy(
            ignore_index=self.parser.LABEL_PADDING,
            num_classes=len(self.parser.labels),
            task="multiclass",
        )
        self.val_extra_labels_accuracy = {
            name: torchmetrics.Accuracy(
                ignore_index=self.parser.LABEL_PADDING,
                num_classes=len(lex),
                task="multiclass",
            )
            for name, lex in self.parser.annotation_lexicons.items()
        }

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
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )

        return output.loss

    def validation_step(self, batch: DependencyBatch, batch_idx: int):
        output: ParserTrainingModuleForwardOutput = self(batch)

        self.log(
            "validation/loss",
            output.loss,
            batch_size=batch.sentences.sent_lengths.shape[0],
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )

        tags_pred = output.output.tag_scores.argmax(dim=-1)
        self.val_tags_accuracy(tags_pred, batch.tags)

        self.log(
            "validation/tags_accuracy",
            self.val_tags_accuracy,
            on_epoch=True,
        )

        # greedy head accuracy (without parsing)
        heads_preds = output.output.head_scores.argmax(dim=-1)
        heads_mask = batch.heads.ne(self.parser.LABEL_PADDING)
        n_heads = heads_mask.sum()
        heads_accuracy = (
            heads_preds.eq(batch.heads).logical_and(heads_mask).sum().true_divide(n_heads)
        )
        self.val_heads_accuracy(heads_accuracy, n_heads)

        self.log(
            "validation/heads_accuracy",
            self.val_heads_accuracy,
            on_epoch=True,
        )

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
        self.val_deprel_accuracy(deprels_pred, batch.labels)

        self.log(
            "validation/deprel_accuracy",
            self.val_deprel_accuracy,
            on_epoch=True,
        )

        # extra labels accuracy
        for name, scores in output.output.extra_labels_scores.items():
            gold_annotation = batch.annotations[name]
            annotation_pred = scores.argmax(dim=-1)
            self.val_extra_labels_accuracy[name](annotation_pred, gold_annotation)
            self.log(
                f"validation/{name}_accuracy",
                self.val_extra_labels_accuracy[name],
                on_epoch=True,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), betas=(0.9, 0.9), lr=self.config.lr, eps=1e-09
        )

        if self.config.lr_schedule["shape"] == "exponential":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                (lambda n: 0.95**n),
            )
            schedulers = [{"scheduler": scheduler, "interval": "epoch"}]
        elif self.config.lr_schedule["shape"] == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                self.config.lr_schedule["warmup_steps"],
                self.trainer.estimated_stepping_batches,
            )
            schedulers = [{"scheduler": scheduler, "interval": "step"}]
        elif self.config.lr_schedule["shape"] == "constant":
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer, self.config.lr_schedule["warmup_steps"]
            )
            schedulers = [{"scheduler": scheduler, "interval": "step"}]
        else:
            raise ValueError(f"Unkown lr schedule shape {self.config.lr_schedule['shape']!r}")
        return [optimizer], schedulers


class SaveModelCallback(pl.Callback):
    def __init__(self, save_dir: pathlib.Path):
        self.save_dir = save_dir

    @rank_zero_only
    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: ParserTrainingModule, checkpoint: Dict[str, Any]
    ):
        logger.info(f"Saving model to {self.save_dir}")
        pl_module.parser.save(self.save_dir)


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
    default="cpu",
    help="The type device to use for the parsing model. (cpu, gpu, …).",
    show_default=True,
)
@click.option(
    "--dev-file",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, path_type=pathlib.Path),
    help="A CoNLL-U treebank to use as a development dataset.",
)
@click.option(
    "--device",
    help="The device to use for the parsing model.",
    type=click.IntRange(0),
)
@click.option(
    "--name",
    default=datetime.datetime.now().isoformat(),
    help="The name of the experiment. Defaults to the current datetime.",
)
def train(
    accelerator: str,
    dev_file: Optional[pathlib.Path],
    device: Optional[int],
    config_file: pathlib.Path,
    name: str,
    output_dir: pathlib.Path,
    train_file: pathlib.Path,
):
    _device: Union[Literal["auto"], int]
    if device is None:
        _device = "auto"
    else:
        _device = device
    setup_logging()
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / "model"
    shutil.copy(config_file, output_dir / config_file.name)
    with open(config_file) as in_stream:
        hp = yaml.load(in_stream, Loader=yaml.SafeLoader)

    lr_config = hp["lr"]
    train_config = TrainConfig(
        batch_size=hp["batch_size"],
        epochs=hp["epochs"],
        lr=lr_config["base"],
        lr_schedule=lr_config.get("schedule", {"shape": "exponential", "warmup_steps": 0}),
    )

    with open(train_file) as in_stream:
        train_trees = list(DepGraph.read_conll(in_stream))

    parser = BiAffineParser.initialize(
        config_path=config_file,
        treebank=train_trees,
    )

    train_set = DependencyDataset(
        parser,
        train_trees,
        skip_unencodable=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_config.batch_size,
        collate_fn=parser.batch_trees,
        shuffle=True,
    )
    if dev_file is not None:
        with open(dev_file) as in_stream:
            dev_trees = list(DepGraph.read_conll(in_stream))
        dev_set = DependencyDataset(
            parser,
            dev_trees,
            skip_unencodable=True,
        )
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_set,
            batch_size=train_config.batch_size,
            collate_fn=parser.batch_trees,
            shuffle=True,
        )
    else:
        dev_loader = None
    train_module = ParserTrainingModule(parser=parser, config=train_config)
    callbacks = [
        pl_callbacks.RichProgressBar(console_kwargs={"stderr": True}),
        pl_callbacks.LearningRateMonitor("step"),
        SaveModelCallback(save_dir=output_dir / "model"),
    ]
    if dev_loader is not None:
        callbacks.append(
            pl_callbacks.ModelCheckpoint(save_top_k=1, monitor="validation/heads_accuracy")
        )
    else:
        callbacks.append(pl_callbacks.ModelCheckpoint(save_top_k=1, monitor="train/loss"))
    loggers = [
        CSVLogger(output_dir, version=name),
        TensorBoardLogger(output_dir, version=name),
    ]
    trainer = pl.Trainer(
        accelerator=accelerator,
        callbacks=callbacks,
        default_root_dir=output_dir,
        devices=_device,
        log_every_n_steps=1,
        logger=loggers,
        max_epochs=train_config.epochs,
    )
    trainer.fit(train_module, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    parser.save(model_path)


if __name__ == "__main__":
    train()
