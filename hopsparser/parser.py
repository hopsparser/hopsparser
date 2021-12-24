import collections.abc
import json
import math
import pathlib
import random
import shutil
import warnings
from typing import (
    Any,
    Dict,
    IO,
    BinaryIO,
    Callable,
    Final,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pydantic
import torch
import transformers
import yaml
from bidict import bidict, BidirectionalMapping
from boltons import iterutils as itu
from loguru import logger
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from hopsparser import deptree, lexers
from hopsparser.deptree import DepGraph
from hopsparser.lexers import (
    BertLexer,
    CharRNNLexer,
    Lexer,
    SupportsTo,
    WordEmbeddingsLexer,
    FastTextLexer,
    freeze_module,
)
from hopsparser.mst import chuliu_edmonds_one_root as chuliu_edmonds
from hopsparser.utils import smart_open


def gen_tags(treelist: Iterable[DepGraph]) -> List[str]:
    tagset = set([tag for tree in treelist for tag in tree.pos_tags[1:]])
    if tagset.intersection((BiAffineParser.PAD_TOKEN, BiAffineParser.UNK_WORD)):
        raise ValueError("Tag conflict: the treebank contains reserved POS tags")
    return [
        BiAffineParser.PAD_TOKEN,
        DepGraph.ROOT_TOKEN,
        BiAffineParser.UNK_WORD,
        *sorted(tagset),
    ]


def gen_labels(treelist: Iterable[DepGraph]) -> List[str]:
    labels = set([lbl for tree in treelist for lbl in tree.deprels])
    if BiAffineParser.PAD_TOKEN in labels:
        raise ValueError("Tag conflict: the treebank contains reserved dep labels")
    return [BiAffineParser.PAD_TOKEN, *sorted(labels)]


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0
    ):
        super(MLP, self).__init__()
        self.input_dim: Final[int] = input_dim
        self.output_dim: Final[int] = output_dim
        self.w_down = nn.Linear(self.input_dim, hidden_dim)
        self.w_up = nn.Linear(hidden_dim, self.output_dim)
        self.g = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.w_up(self.dropout(self.g(self.w_down(inpt))))


# Note: This is the biaffine layer used in Qi et al. (2018) and Dozat and Manning (2017).
class BiAffine(nn.Module):
    """Biaffine attention layer.

    Inputs
    ------

    - `d` a tensor of shape `batch_size×num_dependents×input_dim
    - `h` a tensor of shape `batch_size×num_heads×input_dim

    Outputs
    -------

    A tensor of shape `batch_size×num_dependents×num_heads×output_dim`.
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool):
        super(BiAffine, self).__init__()
        self.input_dim: Final[int] = input_dim
        self.output_dim: Final[int] = output_dim
        self.bias: Final[bool] = bias
        weight_input = input_dim + 1 if bias else input_dim
        self.weight = nn.Parameter(torch.empty(output_dim, weight_input, weight_input))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, d: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.bias:
            d = torch.cat((d, d.new_ones((*d.shape[:-1], 1))), dim=-1)
            h = torch.cat((h, h.new_ones((*h.shape[:-1], 1))), dim=-1)
        return torch.einsum("bxi,oij,byj->bxyo", d, self.weight, h)


class EncodedSentence(NamedTuple):
    words: Sequence[str]
    encodings: Dict[str, SupportsTo]
    sent_len: int


_T_SentencesBatch = TypeVar("_T_SentencesBatch", bound="SentencesBatch")


class SentencesBatch(NamedTuple):
    """Batched and padded sentences.

    ## Attributes

    - `words` The word forms for every sentence in the batch
    - `encodings` A dict mapping a lexer name to the correponding encoding of the batch of sentences
    - `tags` The gold POS tags (if any) as a `LongTensor` with shape `(batch_size,
      max_sentence_length)`
    - `heads` The gold heads (if any) as a `LongTensor` with shape `(batch_size,
      max_sentence_length)`
    - `labels` The gold dependency labels (if any) as a `LongTensor` with shape `(batch_size,
      max_sentence_length)`
    - `sent_length` The lengths of the sentences in the batch as `LongTensor` with shape
      `(batch_size,)`
    - `content_mask` A `BoolTensor` mask of shape `(batch_size, max_sentence_length)` such that
      `content_mask[i, j]` is true iff the j-th word of the i-th sentence in the batch is neither
      padding not the root (i.e. iff `1 <= j < sent_length[i]`).
    """

    words: Sequence[Sequence[str]]
    encodings: Dict[str, SupportsTo]
    sent_lengths: torch.Tensor
    content_mask: torch.Tensor

    def to(
        self: _T_SentencesBatch, device: Union[str, torch.device]
    ) -> _T_SentencesBatch:
        return type(self)(
            words=self.words,
            encodings={
                lexer_name: encoded_batch.to(device)
                for lexer_name, encoded_batch in self.encodings.items()
            },
            sent_lengths=self.sent_lengths,
            content_mask=self.content_mask.to(device),
        )


class EncodedTree(NamedTuple):
    sentence: EncodedSentence
    heads: torch.Tensor
    labels: torch.Tensor
    tags: torch.Tensor


_T_DependencyBatch = TypeVar("_T_DependencyBatch", bound="DependencyBatch")


class DependencyBatch(NamedTuple):
    """Encoded, padded and batched trees.

    ## Attributes

    - `sentences` The sentences as a `SentencesBatch`
    - `trees` The sentences as `DepGraph`s for rich attribute access.
    - `tags` The gold POS tags (if any) as a `LongTensor` with shape `(batch_size,
      max_sentence_length)`
    - `heads` The gold heads (if any) as a `LongTensor` with shape `(batch_size,
      max_sentence_length)`
    - `labels` The gold dependency labels (if any) as a `LongTensor` with shape `(batch_size,
      max_sentence_length)`
    """

    trees: Sequence[DepGraph]
    sentences: SentencesBatch
    tags: torch.Tensor
    heads: torch.Tensor
    labels: torch.Tensor

    def to(
        self: _T_DependencyBatch, device: Union[str, torch.device]
    ) -> _T_DependencyBatch:
        return type(self)(
            trees=self.trees,
            sentences=self.sentences.to(device),
            tags=self.tags.to(device),
            heads=self.heads.to(device),
            labels=self.labels.to(device),
        )


class LRSchedule(TypedDict):
    shape: Literal["exponential", "linear", "constant"]
    warmup_steps: int


class BiAffineParserConfig(pydantic.BaseModel):
    mlp_input: int
    mlp_tag_hidden: int
    mlp_arc_hidden: int
    mlp_lab_hidden: int
    biased_biaffine: bool
    default_batch_size: int
    encoder_dropout: float
    labels: List[str]
    mlp_dropout: float
    tagset: List[str]
    lexers: Dict[str, str]


_T_BiAffineParser = TypeVar("_T_BiAffineParser", bound="BiAffineParser")


class BiAffineParser(nn.Module):
    """Biaffine Dependency Parser."""

    PAD_IDX: Final[int] = 0
    PAD_TOKEN: Final[str] = "<pad>"
    UNK_WORD: Final[str] = "<unk>"
    # Labels that are -100 are ignored in torch crossentropy (we still set it explicitely)
    LABEL_PADDING: Final[int] = -100

    # FIXME: `mlp_input` here is conterintuitive: the actual MLP input dim will be twice that, this is
    # more accurately the dimension of the outputs of each direction of the LSTM
    # FIXME: `default_batch_size` is discutable, since it's heavily machine-dependent
    def __init__(
        self,
        biased_biaffine: bool,
        default_batch_size: int,
        encoder_dropout: float,  # lstm dropout
        labels: Sequence[str],
        lexers: Dict[str, lexers.Lexer],
        mlp_input: int,
        mlp_tag_hidden: int,
        mlp_arc_hidden: int,
        mlp_lab_hidden: int,
        mlp_dropout: float,
        tagset: Sequence[str],
    ):

        super(BiAffineParser, self).__init__()
        self.default_batch_size = default_batch_size
        self.tagset: BidirectionalMapping[str, int] = bidict(
            (t, i) for i, t in enumerate(tagset)
        )
        self.labels: BidirectionalMapping[str, int] = bidict(
            (l, i) for i, l in enumerate(labels)
        )
        self.mlp_arc_hidden: Final[int] = mlp_arc_hidden
        self.mlp_input: Final[int] = mlp_input
        self.mlp_tag_hidden: Final[int] = mlp_tag_hidden
        self.mlp_lab_hidden: Final[int] = mlp_lab_hidden
        self.mlp_dropout = mlp_dropout

        # TODO: fix typing here, casting works but it's inelegant
        # The issue is `Lexer` is a protocol and we can't require them to be torch Modules (which is
        # a concrete class and you can't ask a protocol to subclass a concrete class or ask for the
        # intersection of a protocol and a concrete class), but we do need them to be torch Modules
        # since the have to wrap them in a `ModuleDict` here`. Also we really don't want to separate
        # lexers that wouldn't be torch Modules since that's really never going to happen in
        # practice.
        self.lexers = cast(
            Dict[str, Lexer], nn.ModuleDict(cast(Dict[str, nn.Module], lexers))
        )
        self.lexers_order = sorted(self.lexers.keys())

        self.dep_rnn = nn.LSTM(
            sum(lex.output_dim for lex in self.lexers.values()),
            mlp_input,
            3,
            batch_first=True,
            dropout=encoder_dropout,
            bidirectional=True,
        )

        # POS tagger & char RNN
        self.pos_tagger = MLP(self.mlp_input * 2, self.mlp_tag_hidden, len(self.tagset))

        # Arc MLPs
        self.arc_mlp_h = MLP(
            self.mlp_input * 2, self.mlp_arc_hidden, self.mlp_input, self.mlp_dropout
        )
        self.arc_mlp_d = MLP(
            self.mlp_input * 2, self.mlp_arc_hidden, self.mlp_input, self.mlp_dropout
        )
        # Label MLPs
        self.lab_mlp_h = MLP(
            self.mlp_input * 2, self.mlp_lab_hidden, self.mlp_input, self.mlp_dropout
        )
        self.lab_mlp_d = MLP(
            self.mlp_input * 2, self.mlp_lab_hidden, self.mlp_input, self.mlp_dropout
        )

        # BiAffine layers
        self.arc_biaffine = BiAffine(self.mlp_input, 1, bias=biased_biaffine)
        self.lab_biaffine = BiAffine(
            self.mlp_input, len(self.labels), bias=biased_biaffine
        )

    def save_params(self, path: Union[str, pathlib.Path, BinaryIO]):
        torch.save(self.state_dict(), path)

    def load_params(self, path: Union[str, pathlib.Path, BinaryIO]):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def forward(
        self,
        encodings: Dict[str, Any],
        sent_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict POS, heads and deprel scores.

        ## Outputs

        `tag_scores, arc_scores, lab_scores` with shapes

        - `tag_score`: $`batch_size×max_sent_length×num_pos_tags`$
        - `arc_scores`: $`batch_size×n_deps×n_possible_heads`$
        - `label_scores`: $`batch_size×n_deps×n_possible_heads×num_deprels`$
        """
        embeddings = [
            self.lexers[lexer_name](encodings[lexer_name])
            for lexer_name in self.lexers_order
        ]
        inpt = torch.cat(embeddings, dim=-1)
        packed_inpt = pack_padded_sequence(
            inpt, sent_lengths, batch_first=True, enforce_sorted=False
        )
        # TODO: everything after this point could be jitted
        # maybe use an auxilary private module?
        packed_dep_embeddings, _ = self.dep_rnn(packed_inpt)
        dep_embeddings, _ = pad_packed_sequence(packed_dep_embeddings, batch_first=True)

        # Tagging
        tag_scores = self.pos_tagger(dep_embeddings)

        arc_h = self.arc_mlp_h(dep_embeddings)
        arc_d = self.arc_mlp_d(dep_embeddings)
        lab_h = self.lab_mlp_h(dep_embeddings)
        lab_d = self.lab_mlp_d(dep_embeddings)

        arc_scores = self.arc_biaffine(arc_d, arc_h).squeeze(-1)
        lab_scores = self.lab_biaffine(lab_d, lab_h)

        return tag_scores, arc_scores, lab_scores

    # TODO: make this an independent function
    # TODO: hardcode the marginal loss for now
    # TODO: JIT this (or split it and jit the marginals?)
    def parser_loss(
        self,
        tagger_scores: torch.Tensor,
        arc_scores: torch.Tensor,
        lab_scores: torch.Tensor,
        batch: DependencyBatch,
        marginal_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        batch_size = batch.heads.shape[0]
        num_padded_deps = batch.heads.shape[1]
        num_deprels = lab_scores.shape[-1]
        # ARC LOSS
        # [batch×num_deps, num_heads]
        arc_scores_flat = arc_scores.view(-1, arc_scores.size(-1))
        arc_loss = marginal_loss(arc_scores_flat, batch.heads.view(-1))

        # TAGGER_LOSS
        # [batch×sent_length, num_tags]
        tagger_scores_flat = tagger_scores.view(-1, tagger_scores.size(-1))
        tagger_loss = marginal_loss(tagger_scores_flat, batch.tags.view(-1))

        # LABEL LOSS
        # We will select the labels for the true heads, so we have to give a true head to
        # the padding tokens (even if they will be ignored in the crossentropy since the true
        # label for that head is set to -100) so we give them the root.
        positive_heads = batch.heads.masked_fill(
            batch.sentences.content_mask.logical_not(), 0
        )
        heads_selection = positive_heads.view(batch_size, num_padded_deps, 1, 1)
        # [batch, n_dependents, 1, n_labels]
        heads_selection = heads_selection.expand(
            batch_size, num_padded_deps, 1, num_deprels
        )
        # [batch, n_dependents, 1, n_labels]
        predicted_labels_scores = torch.gather(lab_scores, -2, heads_selection)

        # [batch×sent_len, n_labels]
        predicted_labels_scores_flat = predicted_labels_scores.view(-1, num_deprels)
        lab_loss = marginal_loss(predicted_labels_scores_flat, batch.labels.view(-1))

        # TODO: see if other loss combination functions wouldn't help here, e.g.
        # <https://arxiv.org/abs/1805.06334> or <https://arxiv.org/abs/1209.2784>
        # tracked at <https://github.com/hopsparser/npdependency/issues/59>
        return tagger_loss + arc_loss + lab_loss

    def eval_model(
        self, dev_set: Iterable[DependencyBatch], batch_size: Optional[int] = None
    ) -> Tuple[float, float, float, float]:
        if batch_size is None:
            batch_size = self.default_batch_size

        loss_fnc = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.LABEL_PADDING)

        self.eval()
        device = next(self.parameters()).device
        # NOTE: the accuracy scoring is approximative and cannot be interpreted as an UAS/LAS score
        # NOTE: fun project: track the correlation between them
        tag_tp = torch.zeros(1, dtype=torch.long, device=device)
        arc_tp = torch.zeros(1, dtype=torch.long, device=device)
        lab_tp = torch.zeros(1, dtype=torch.long, device=device)
        overall_size = torch.zeros(1, dtype=torch.long, device=device)
        gloss = torch.zeros(1, dtype=torch.float, device=device)

        with torch.no_grad():
            for batch in dev_set:
                overall_size += batch.sentences.content_mask.sum()

                batch = batch.to(device)

                # workaround since the design of torch modules makes it hard
                # for static analyzer to find out their return type
                tagger_scores: torch.Tensor
                arc_scores: torch.Tensor
                lab_scores: torch.Tensor
                tagger_scores, arc_scores, lab_scores = self(
                    batch.sentences.encodings,
                    batch.sentences.sent_lengths,
                )

                gloss += self.parser_loss(
                    tagger_scores, arc_scores, lab_scores, batch, loss_fnc
                )

                # greedy arc accuracy (without parsing)
                arc_pred = arc_scores.argmax(dim=-1)
                arc_accuracy = (
                    arc_pred.eq(batch.heads)
                    .logical_and(batch.sentences.content_mask)
                    .sum()
                )
                arc_tp += arc_accuracy

                # tagger accuracy
                tag_pred = tagger_scores.argmax(dim=-1)
                tag_accuracy = (
                    tag_pred.eq(batch.tags)
                    .logical_and(batch.sentences.content_mask)
                    .sum()
                )
                tag_tp += tag_accuracy

                # greedy label accuracy (without parsing)
                gold_heads_select = (
                    batch.heads.masked_fill(
                        batch.sentences.content_mask.logical_not(), 0
                    )
                    .view(batch.heads.shape[0], batch.heads.shape[1], 1, 1)
                    .expand(
                        batch.heads.shape[0],
                        batch.heads.shape[1],
                        1,
                        lab_scores.shape[-1],
                    )
                )
                # shape: num_padded_deps×num_padded_heads
                gold_head_lab_scores = torch.gather(
                    lab_scores, -2, gold_heads_select
                ).squeeze(-2)
                lab_pred = gold_head_lab_scores.argmax(dim=-1)
                lab_accuracy = (
                    lab_pred.eq(batch.labels)
                    .logical_and(batch.sentences.content_mask)
                    .sum()
                )
                lab_tp += lab_accuracy.item()

        return (
            gloss.true_divide(overall_size).item(),
            tag_tp.true_divide(overall_size).item(),
            arc_tp.true_divide(overall_size).item(),
            lab_tp.true_divide(overall_size).item(),
        )

    def train_model(
        self,
        train_set: "DependencyDataset",
        epochs: int,
        lr: float,
        lr_schedule: LRSchedule,
        model_path: Union[str, pathlib.Path],
        batch_size: Optional[int] = None,
        dev_set: Optional["DependencyDataset"] = None,
    ):
        model_path = pathlib.Path(model_path)
        weights_file = model_path / "weights.pt"
        if batch_size is None:
            batch_size = self.default_batch_size
        device = next(self.parameters()).device
        logger.info(f"Start training on {device}")
        loss_fnc = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.LABEL_PADDING)

        # TODO: make these configurable?
        optimizer = torch.optim.Adam(
            self.parameters(), betas=(0.9, 0.9), lr=lr, eps=1e-09
        )

        if lr_schedule["shape"] == "exponential":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                (lambda n: 0.95 ** (n // (math.ceil(len(train_set) / batch_size)))),
            )
        elif lr_schedule["shape"] == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                lr_schedule["warmup_steps"],
                epochs * math.ceil(len(train_set) / batch_size) + 1,
            )
        elif lr_schedule["shape"] == "constant":
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer, lr_schedule["warmup_steps"]
            )
        else:
            raise ValueError(f"Unkown lr schedule shape {lr_schedule['shape']!r}")

        best_arc_acc = 0.0
        for e in range(epochs):
            train_loss = torch.zeros(1, dtype=torch.float, device=device)
            overall_size = torch.zeros(1, dtype=torch.long, device=device)
            train_batches = train_set.make_batches(
                batch_size,
                shuffle_batches=True,
                shuffle_data=True,
            )
            self.train()
            for batch in train_batches:
                overall_size += batch.sentences.content_mask.sum()

                batch = batch.to(device)

                # FORWARD
                tagger_scores, arc_scores, lab_scores = self(
                    batch.sentences.encodings,
                    batch.sentences.sent_lengths,
                )

                loss = self.parser_loss(
                    tagger_scores, arc_scores, lab_scores, batch, loss_fnc
                )

                with torch.no_grad():
                    train_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            if dev_set is not None:
                dev_loss, dev_tag_acc, dev_arc_acc, dev_lab_acc = self.eval_model(
                    dev_set.make_batches(
                        batch_size, shuffle_batches=False, shuffle_data=False
                    ),
                    batch_size=batch_size,
                )
                logger.info(
                    f"Epoch {e}"
                    f" train loss {train_loss.true_divide(overall_size).item():.4f}"
                    f" dev loss {dev_loss:.4f}"
                    f" dev tag acc {dev_tag_acc:.2%}"
                    f" dev head acc {dev_arc_acc:.2%}"
                    f" dev deprel acc {dev_lab_acc:.2%}"
                )

                if dev_arc_acc > best_arc_acc:
                    logger.info(
                        f"New best model: head accuracy {dev_arc_acc:.2%} > {best_arc_acc:.2%}"
                    )
                    self.save_params(weights_file)
                    best_arc_acc = dev_arc_acc
            else:
                self.save_params(weights_file)

        self.load_params(weights_file)

    @overload
    def encode_sentence(
        self, words: Sequence[str], strict: Literal[True] = True
    ) -> EncodedSentence:
        pass

    @overload
    def encode_sentence(
        self, words: Sequence[str], strict: bool
    ) -> Optional[EncodedSentence]:
        pass

    def encode_sentence(
        self, words: Sequence[str], strict: bool = True
    ) -> Optional[EncodedSentence]:
        words_with_root = [DepGraph.ROOT_TOKEN, *words]
        try:
            encodings = {
                lexer_name: lexer.encode(words_with_root)
                for lexer_name, lexer in self.lexers.items()
            }
        except lexers.LexingError as e:
            if strict:
                raise e
            else:
                logger.info(
                    f"Skipping sentence {e.sentence} due to lexing error '{e.message}'.",
                )
                return None
        encoded = EncodedSentence(
            words=words,
            encodings=encodings,
            sent_len=len(words_with_root),
        )
        return encoded

    def batch_sentences(self, sentences: Sequence[EncodedSentence]) -> SentencesBatch:
        words = [sent.words for sent in sentences]
        encodings = {
            lexer_name: lexer.make_batch(
                [sent.encodings[lexer_name] for sent in sentences]
            )
            for lexer_name, lexer in self.lexers.items()
        }

        sent_lengths = torch.tensor(
            [sent.sent_len for sent in sentences], dtype=torch.long
        )

        content_mask = (
            torch.arange(sent_lengths.max().item())
            .unsqueeze(0)
            .lt(sent_lengths.unsqueeze(1))
        )
        # For the root tokens
        content_mask[:, 0] = False

        return SentencesBatch(
            words=words,
            encodings=encodings,
            content_mask=content_mask,
            sent_lengths=sent_lengths,
        )

    def encode_tree(self, tree: DepGraph) -> EncodedTree:
        sentence = self.encode_sentence(tree.words[1:])
        tag_idxes = torch.tensor(
            [self.tagset.get(tag, self.LABEL_PADDING) for tag in tree.pos_tags],
            dtype=torch.long,
        )
        tag_idxes[0] = self.LABEL_PADDING
        heads = torch.tensor(tree.heads, dtype=torch.long)
        heads[0] = self.LABEL_PADDING
        # FIXME: should unk labels be padding?
        labels = torch.tensor(
            [self.labels.get(lab, self.LABEL_PADDING) for lab in tree.deprels],
            dtype=torch.long,
        )
        labels[0] = self.LABEL_PADDING
        return EncodedTree(
            sentence=sentence,
            heads=heads,
            labels=labels,
            tags=tag_idxes,
        )

    def batch_trees(
        self,
        trees: Sequence[DepGraph],
        encoded_trees: Optional[Sequence[EncodedTree]] = None,
    ) -> DependencyBatch:
        if encoded_trees is None:
            encoded_trees = [self.encode_tree(tree) for tree in trees]
        # FIXME: typing err here because we need to constraint that the encoded trees are all
        # encoded by the same lexer
        sentences = self.batch_sentences([tree.sentence for tree in encoded_trees])

        tags = pad_sequence(
            [tree.tags for tree in encoded_trees],
            batch_first=True,
            padding_value=self.LABEL_PADDING,
        )
        heads = pad_sequence(
            [tree.heads for tree in encoded_trees],
            batch_first=True,
            padding_value=self.LABEL_PADDING,
        )
        labels = pad_sequence(
            [tree.labels for tree in encoded_trees],
            batch_first=True,
            padding_value=self.LABEL_PADDING,
        )

        return DependencyBatch(
            sentences=sentences,
            heads=heads,
            labels=labels,
            tags=tags,
            trees=trees,
        )

    def batched_predict(
        self,
        batch_lst: Union[Iterable[DependencyBatch], Iterable[SentencesBatch]],
        greedy: bool = False,
    ) -> Iterable[DepGraph]:
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in batch_lst:
                batch = batch.to(device)

                if isinstance(batch, DependencyBatch):
                    batch_sentences = batch.sentences
                else:
                    batch_sentences = batch
                # batch prediction
                # workaround since the design of torch modules makes it hard
                # for static analyzer to find out their return type
                tagger_scores_batch: torch.Tensor
                arc_scores_batch: torch.Tensor
                lab_scores_batch: torch.Tensor
                tagger_scores_batch, arc_scores_batch, lab_scores_batch = self(
                    batch_sentences.encodings,
                    batch_sentences.sent_lengths,
                )
                arc_scores_batch = arc_scores_batch.cpu()

                if isinstance(batch, DependencyBatch):
                    trees = batch.trees
                else:
                    trees = [DepGraph.from_words(words) for words in batch.words]
                for (
                    tree,
                    sentence_length,
                    tagger_scores,
                    arc_scores,
                    lab_scores,
                ) in zip(
                    trees,
                    batch_sentences.sent_lengths,
                    tagger_scores_batch,
                    arc_scores_batch,
                    lab_scores_batch,
                ):
                    tagger_scores = tagger_scores[:sentence_length, :]
                    arc_scores = arc_scores[:sentence_length, :sentence_length]
                    lab_scores = lab_scores[:sentence_length, :sentence_length, :]
                    # Predict heads
                    probs = arc_scores.cpu().numpy()
                    mst_heads_np = (
                        np.argmax(probs, axis=1) if greedy else chuliu_edmonds(probs)
                    )
                    # shape: num_deps
                    mst_heads = torch.from_numpy(mst_heads_np).to(device)

                    # Predict tags
                    tag_idxes = tagger_scores.argmax(dim=1)
                    # Predict labels
                    # shape: num_deps×1×num_deprel
                    select = mst_heads.view(-1, 1, 1).expand(-1, 1, lab_scores.size(-1))
                    # shape: num_deps×num_deprel
                    selected = torch.gather(lab_scores, 1, select).view(
                        sentence_length, -1
                    )
                    mst_labels = selected.argmax(dim=-1)

                    # `[1:]` to ignore the root node's tag
                    pos_tags = [
                        self.tagset.inverse[idx] for idx in tag_idxes[1:].tolist()
                    ]
                    # `[1:]` to ignore the root node's head
                    heads = mst_heads[1:].tolist()
                    # `[1:]` to ignore the root node's deprel
                    deprels = [
                        self.labels.inverse[lbl] for lbl in mst_labels[1:].tolist()
                    ]
                    edges = [
                        deptree.Edge(head_idx, lbl, dep_idx)
                        for dep_idx, (head_idx, lbl) in enumerate(
                            zip(heads, deprels), start=1
                        )
                    ]
                    result_tree = tree.replace(
                        edges=edges,
                        pos_tags=pos_tags,
                    )
                    yield result_tree

    # FIXME: this is awkward when parsing pre-tokenized input: we should accept something like `inpt:
    # Iterable[Sequence[str]]`
    def parse(
        self,
        inpt: Iterable[str],
        batch_size: Optional[int] = None,
        raw: bool = False,
        strict: bool = True,
    ) -> Iterable[DepGraph]:
        if batch_size is None:
            batch_size = self.default_batch_size
        batches: Union[Iterable[DependencyBatch], Iterable[SentencesBatch]]
        if raw:
            sentences = (
                encoded
                for line in inpt
                if line and not line.isspace()
                for encoded in [
                    self.encode_sentence(line.strip().split(), strict=strict)
                ]
                if encoded is not None
            )
            batches = (
                self.batch_sentences(sentences)
                for sentences in itu.chunked_iter(
                    sentences,
                    size=batch_size,
                )
            )
        else:
            trees = DepGraph.read_conll(inpt)
            batches = (
                self.batch_trees(batch)
                for batch in itu.chunked_iter(trees, size=batch_size)
            )
        yield from self.batched_predict(batches, greedy=False)

    def save(self, model_path: pathlib.Path, save_weights: bool = True):
        model_path.mkdir(exist_ok=True, parents=True)
        config = BiAffineParserConfig(
            biased_biaffine=self.arc_biaffine.bias,
            default_batch_size=self.default_batch_size,
            encoder_dropout=self.dep_rnn.dropout,
            labels=[self.labels.inverse[i] for i in range(len(self.labels))],
            mlp_input=self.mlp_input,
            mlp_tag_hidden=self.mlp_tag_hidden,
            mlp_arc_hidden=self.mlp_arc_hidden,
            mlp_lab_hidden=self.mlp_lab_hidden,
            mlp_dropout=self.mlp_dropout,
            tagset=[self.tagset.inverse[i] for i in range(len(self.tagset))],
            lexers={
                lexer_name: lexers.LEXER_TYPES.inverse[type(lexer)]
                for lexer_name, lexer in self.lexers.items()
            },
        )
        config_file = model_path / "config.json"
        with open(config_file, "w") as out_stream:
            json.dump(config.dict(), out_stream)
        lexers_path = model_path / "lexers"
        for lexer_name, lexer in self.lexers.items():
            lexer.save(model_path=lexers_path / lexer_name, save_weights=False)
        if save_weights:
            torch.save(self.state_dict(), model_path / "weights.pt")

    @classmethod
    def initialize(
        cls: Type[_T_BiAffineParser],
        config_path: pathlib.Path,
        treebank: List[DepGraph],
    ) -> _T_BiAffineParser:
        logger.info(f"Initializing a parser from {config_path}")
        with open(config_path) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.SafeLoader)
        config.setdefault("biased_biaffine", True)
        config.setdefault("batch_size", 1)

        corpus_words = [word for tree in treebank for word in tree.words[1:]]
        parser_lexers: Dict[str, Lexer] = dict()
        lexer: Lexer
        for lexer_config in config["lexers"]:
            lexer_type = lexer_config["type"]
            if lexer_type == "words":
                lexer = WordEmbeddingsLexer.from_words(
                    embeddings_dim=lexer_config["embedding_size"],
                    unk_word=cls.UNK_WORD,
                    word_dropout=lexer_config["word_dropout"],
                    words=(DepGraph.ROOT_TOKEN, cls.UNK_WORD, *corpus_words),
                    words_padding_idx=cls.PAD_IDX,
                )
            elif lexer_config["type"] == "chars_rnn":
                lexer = CharRNNLexer.from_chars(
                    chars=(c for word in corpus_words for c in word),
                    special_tokens=[DepGraph.ROOT_TOKEN],
                    char_embeddings_dim=lexer_config["embedding_size"],
                    output_dim=lexer_config["lstm_output_size"],
                )
            elif lexer_config["type"] == "bert":
                bert_layers = lexer_config.get("layers", "*")
                if bert_layers == "*":
                    bert_layers = None
                lexer = BertLexer.from_pretrained(
                    model_name_or_path=lexer_config["model"],
                    layers=bert_layers,
                    subwords_reduction=lexer_config.get("subwords_reduction", "first"),
                    weight_layers=lexer_config.get("weighted", False),
                )
            elif lexer_config["type"] == "fasttext":
                if (fasttext_model_path := lexer_config.get("source")) is not None:
                    fasttext_model_path = pathlib.Path(fasttext_model_path)
                    if not fasttext_model_path.is_absolute():
                        fasttext_model_path = (
                            config_path.parent / fasttext_model_path
                        ).resolve()
                if fasttext_model_path is None:
                    logger.info("Generating a FastText model from the treebank")
                    lexer = FastTextLexer.from_sents(
                        [tree.words[1:] for tree in treebank],
                        special_tokens=[DepGraph.ROOT_TOKEN],
                    )
                elif fasttext_model_path.exists():
                    try:
                        # ugly, but we have no better way of checking if a file is a valid model
                        lexer = FastTextLexer.from_fasttext_model(
                            fasttext_model_path, special_tokens=[DepGraph.ROOT_TOKEN]
                        )
                    except ValueError:
                        # FastText couldn't load it, so it should be raw text
                        logger.info(
                            f"Generating a FastText model from {fasttext_model_path}"
                        )
                        lexer = FastTextLexer.from_raw(
                            fasttext_model_path, special_tokens=[DepGraph.ROOT_TOKEN]
                        )
                else:
                    raise ValueError(f"{fasttext_model_path} not found")
            else:
                raise ValueError(f"Unknown lexer type: {lexer_type!r}")
            parser_lexers[lexer_config["name"]] = lexer

        itolab = gen_labels(treebank)
        itotag = gen_tags(treebank)

        return cls(
            labels=itolab,
            lexers=parser_lexers,
            tagset=itotag,
            biased_biaffine=config["biased_biaffine"],
            default_batch_size=config["batch_size"],
            encoder_dropout=config["encoder_dropout"],
            mlp_input=config["mlp_input"],
            mlp_tag_hidden=config["mlp_tag_hidden"],
            mlp_arc_hidden=config["mlp_arc_hidden"],
            mlp_lab_hidden=config["mlp_lab_hidden"],
            mlp_dropout=config["mlp_dropout"],
        )

    @classmethod
    def load(
        cls: Type[_T_BiAffineParser],
        model_path: Union[str, pathlib.Path],
    ) -> _T_BiAffineParser:
        model_path = pathlib.Path(model_path)
        config_path = model_path / "config.json"

        with open(config_path) as in_stream:
            config_dict = json.load(in_stream)
        config = BiAffineParserConfig.parse_obj(config_dict)

        lexers_path = model_path / "lexers"

        parser_lexers: Dict[str, Lexer] = dict()
        for lexer_name, lexer_type in config.lexers.items():
            lexer_class = lexers.LEXER_TYPES[lexer_type]
            parser_lexers[lexer_name] = lexer_class.load(lexers_path / lexer_name)
        parser_lexers

        parser = cls(
            biased_biaffine=config.biased_biaffine,
            default_batch_size=config.default_batch_size,
            encoder_dropout=config.encoder_dropout,
            labels=config.labels,
            lexers=parser_lexers,
            mlp_arc_hidden=config.mlp_arc_hidden,
            mlp_dropout=config.mlp_dropout,
            mlp_input=config.mlp_input,
            mlp_lab_hidden=config.mlp_lab_hidden,
            mlp_tag_hidden=config.mlp_tag_hidden,
            tagset=config.tagset,
        )
        weights_file = model_path / "weights.pt"
        if weights_file.exists():
            parser.load_params(str(weights_file))

        return parser


# TODO: replace this by a torch Dataset+Dataloader (or datapipe?)
# FIXME: why are we not requiring a sequence for treelist again?
class DependencyDataset:
    def __init__(
        self,
        parser: BiAffineParser,
        treelist: Iterable[DepGraph],
    ):
        self.parser = parser
        self.treelist = treelist

        self.itolab = self.parser.labels
        self.labtoi = {label: idx for idx, label in enumerate(self.itolab)}

        self.itotag = self.parser.tagset
        self.tagtoi = {tag: idx for idx, tag in enumerate(self.itotag)}

        self.encoded_trees: List[EncodedTree] = []

    def encode(self):
        self.encoded_trees = []
        for tree in self.treelist:
            self.encoded_trees.append(self.parser.encode_tree(tree))

    def make_batches(
        self,
        batch_size: int,
        shuffle_batches: bool = False,
        shuffle_data: bool = True,
    ) -> Iterable[DependencyBatch]:
        if not isinstance(self.treelist, collections.abc.Sequence):
            self.treelist = list(self.treelist)
        if not self.encoded_trees:
            self.encode()
        N = len(self.treelist)
        order = list(range(N))
        if shuffle_data:
            random.shuffle(order)

        batch_order = list(range(0, N, batch_size))
        if shuffle_batches:
            random.shuffle(batch_order)

        for i in batch_order:
            batch_indices = order[i : i + batch_size]
            trees = [self.treelist[j] for j in batch_indices]
            encoded_trees = [self.encoded_trees[j] for j in batch_indices]
            yield self.parser.batch_trees(trees, encoded_trees)

    def __len__(self):
        return len(self.treelist)


def train(
    config_file: pathlib.Path,
    model_path: pathlib.Path,
    train_file: pathlib.Path,
    device: Union[str, torch.device] = "cpu",
    max_tree_length: Optional[int] = None,
    overwrite: bool = False,
    rand_seed: Optional[int] = None,
    dev_file: Optional[pathlib.Path] = None,
):
    if rand_seed is not None:
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)

    with open(config_file) as in_stream:
        hp = yaml.load(in_stream, Loader=yaml.SafeLoader)

    with open(train_file) as in_stream:
        traintrees = list(
            DepGraph.read_conll(in_stream, max_tree_length=max_tree_length)
        )
    if model_path.exists() and not overwrite:
        logger.info(f"Continuing training from {model_path}")
        parser = BiAffineParser.load(model_path)
    else:
        if overwrite:
            if model_path.exists():
                logger.info(
                    f"Erasing existing trained model in {model_path} since overwrite was asked",
                )
                shutil.rmtree(model_path)
            else:
                logger.warning(f"--overwrite asked but {model_path} does not exist")
        parser = BiAffineParser.initialize(
            config_path=config_file,
            treebank=traintrees,
        )
    parser = parser.to(device)
    for lexer_to_freeze_name in hp.get("freeze", []):
        if (lexer := parser.lexers.get(lexer_to_freeze_name)) is not None:
            # TODO: remove the cast once we've figured out how to require our lexers to be modules
            freeze_module(cast(nn.Module, lexer))
        else:
            warnings.warn(
                f"I can't freeze a {lexer_to_freeze_name!r} lexer that does not exist"
            )
    parser.save(model_path)

    trainset = DependencyDataset(
        parser,
        traintrees,
    )
    devset: Optional[DependencyDataset]
    if dev_file is not None:
        with open(dev_file) as in_stream:
            devset = DependencyDataset(
                parser,
                list(DepGraph.read_conll(in_stream)),
            )
    else:
        devset = None

    lr_config = hp["lr"]
    parser.train_model(
        batch_size=hp["batch_size"],
        dev_set=devset,
        epochs=hp["epochs"],
        lr=lr_config["base"],
        lr_schedule=lr_config.get(
            "schedule", {"shape": "exponential", "warmup_steps": 0}
        ),
        model_path=model_path,
        train_set=trainset,
    )


def parse(
    model_path: Union[str, pathlib.Path],
    in_file: Union[str, pathlib.Path, IO[str]],
    out_file: Union[str, pathlib.Path, IO[str]],
    device: Union[str, torch.device] = "cpu",
    batch_size: Optional[int] = None,
    raw: bool = False,
    strict: bool = True,
):
    parser = BiAffineParser.load(model_path)
    parser = parser.to(device)
    with smart_open(in_file) as in_stream, smart_open(out_file, "w") as ostream:
        for tree in parser.parse(
            inpt=in_stream, batch_size=batch_size, raw=raw, strict=strict
        ):
            ostream.write(tree.to_conllu())
            ostream.write("\n\n")
