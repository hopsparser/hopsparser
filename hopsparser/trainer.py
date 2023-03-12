from typing import NamedTuple

import lightning.pytorch as pl
import pydantic
import torch
import transformers

from hopsparser.parser import (
    BiAffineParser,
    BiaffineParserOutput,
    DependencyBatch,
    LRSchedule,
)


class TrainConfig(pydantic.BaseModel):
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
