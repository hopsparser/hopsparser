import logging
import pathlib
import shutil
from typing import NamedTuple, Optional
import warnings
import click

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
import pydantic
import rich
import torch
import torch.utils.data
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

    def forward(self, x: DependencyBatch) -> ParserTrainingModuleForwardOutput:
        output = self.parser(x.sentences.encodings, x.sentences.sent_lengths)
        loss = self.parser.parser_loss(output, x)
        return ParserTrainingModuleForwardOutput(output=output, loss=loss)

    def training_step(self, batch: DependencyBatch, batch_idx: int) -> torch.Tensor:
        # FIXME: why is batch a list of len 1 here??? See the doc

        output: ParserTrainingModuleForwardOutput = self(batch)
        return output.loss

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


# TODO: small cli that initializes and train a model from the existing initialization file
# so we can test and gradually improve this
# then we can move one to refinments


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
def train(
    config_file: pathlib.Path,
    output_dir: pathlib.Path,
    train_file: pathlib.Path,
):
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
        traintrees = list(DepGraph.read_conll(in_stream))

    parser = BiAffineParser.initialize(
        config_path=config_file,
        treebank=traintrees,
    )

    train_set = DependencyDataset(
        parser,
        traintrees,
        skip_unencodable=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_config.batch_size,
        collate_fn=parser.batch_trees,
        shuffle=True,
    )
    train_module = ParserTrainingModule(parser=parser, config=train_config)
    callbacks = [
        pl_callbacks.RichProgressBar(console_kwargs={"stderr": True}),
        pl_callbacks.LearningRateMonitor("step"),
    ]
    trainer = pl.Trainer(
        callbacks=callbacks, default_root_dir=output_dir, max_epochs=train_config.epochs
    )
    trainer.fit(train_module, train_dataloaders=train_loader)
    parser.save(model_path)


if __name__ == "__main__":
    train()
