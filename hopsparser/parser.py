import json
import math
import pathlib
import random
import shutil
import warnings
from typing import (
    IO,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Type,
    TypedDict,
    Union,
    cast,
    overload,
)

import pydantic

# import torch
import torch.utils.data
import transformers
import yaml
from bidict import BidirectionalMapping, bidict
from boltons import iterutils as itu
from loguru import logger
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from typing_extensions import Self

from hopsparser import lexers, utils
from hopsparser.deptree import DepGraph
from hopsparser.lexers import (
    BertLexer,
    CharRNNLexer,
    FastTextLexer,
    Lexer,
    SupportsTo,
    WordEmbeddingsLexer,
    freeze_module,
)
from hopsparser.mst import chuliu_edmonds_one_root as chuliu_edmonds
from hopsparser.utils import smart_open


def gen_tags(treelist: Iterable[DepGraph]) -> List[str]:
    tagset = {tag for tree in treelist for tag in tree.pos_tags[1:] if tag is not None}
    return sorted(tagset)


def gen_annotations_labels(
    treelist: Iterable[DepGraph], annotation_names: Sequence[str]
) -> Dict[str, List[str]]:
    label_sets: Dict[str, Set[str]] = {name: set() for name in annotation_names}
    for tree in treelist:
        for node in tree.nodes:
            for name, labels in label_sets.items():
                if (node_label := node.misc.mapping.get(name)) is not None:
                    labels.add(node_label)
    if (
        name_nolabel := next(
            (name for name, labels in label_sets.items() if len(labels) <= 1), None
        )
    ) is not None:
        # No iterable unpacking for poor walrus :( (<https://bugs.python.org/issue43143>)
        labels = label_sets[name_nolabel]
        if not labels:
            raise ValueError(f"No label found in treebank for annotation {name_nolabel}")
        else:
            logger.warning(
                f"Only one label ({next(iter(labels))} found for annotation {labels}, this is likely an error."
            )
    return {name: sorted(labels) for name, labels in label_sets.items()}


def gen_labels(treelist: Iterable[DepGraph]) -> List[str]:
    label_sets = {lbl for tree in treelist for lbl in tree.deprels if lbl is not None}
    return sorted(label_sets)


# No non-linearity on the output so this can also serve before softmax
# TODO: jit
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super(MLP, self).__init__()
        self.input_dim: Final[int] = input_dim
        self.output_dim: Final[int] = output_dim
        self.w_down = nn.Linear(self.input_dim, hidden_dim)
        self.w_up = nn.Linear(hidden_dim, self.output_dim)
        self.g = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.w_up(self.dropout(self.g(self.w_down(inpt))))


# NOTE: This is the biaffine layer used in Qi et al. (2018) and Dozat and Manning (2017).
class BiAffine(nn.Module):
    """Biaffine attention layer.

    ## Inputs

    - `d` a tensor of shape `batch_size×num_dependents×input_dim
    - `h` a tensor of shape `batch_size×num_heads×input_dim

    ## Outputs

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
            # TODO: can we avoid a copy here
            d = torch.cat((d, d.new_ones((*d.shape[:-1], 1))), dim=-1)
            h = torch.cat((h, h.new_ones((*h.shape[:-1], 1))), dim=-1)
        return torch.einsum("bxi,oij,byj->bxyo", d, self.weight, h)


class EncodedSentence(NamedTuple):
    words: Sequence[str]
    encodings: Dict[str, SupportsTo]
    sent_len: int


class SentencesBatch(NamedTuple):
    """Batched and padded sentences.

    ## Attributes

    - `words` The word forms for every sentence in the batch
    - `encodings` A dict mapping a lexer name to the correponding encoding of the batch of sentences
    - `sent_length` The lengths of the sentences in the batch as `LongTensor` with shape
      `(batch_size,)`
    """

    words: Sequence[Sequence[str]]
    encodings: Dict[str, SupportsTo]
    sent_lengths: torch.Tensor

    def to(self: Self, device: Union[str, torch.device]) -> Self:
        return type(self)(
            encodings={
                lexer_name: encoded_batch.to(device)
                for lexer_name, encoded_batch in self.encodings.items()
            },
            sent_lengths=self.sent_lengths,
            words=self.words,
        )


class EncodedTree(NamedTuple):
    """Annotations for an `EncodedSentence`.

    ## Attributes

    - `sentence`: the sentence in question
    - `tags` The gold POS tags (if any) as a `LongTensor` with shape `(batch_size,
      max_sentence_length)`
    - `heads` The gold heads (if any) as a `LongTensor` with shape `(batch_size,
      max_sentence_length)`
    - `labels` The gold dependency labels (if any) as a `LongTensor` with shape `(batch_size,
      max_sentence_length)`
    """

    sentence: EncodedSentence
    heads: torch.Tensor
    labels: torch.Tensor
    tags: torch.Tensor
    annotations: Dict[str, torch.Tensor]
    tree: DepGraph


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
    - `annotations` The gold labels for extra annotations, as a mapping from annotation names to
      `LongTensor`s with shape `(batch_size, max_sentence_length)`
    """

    trees: Sequence[DepGraph]
    sentences: SentencesBatch
    tags: torch.Tensor
    heads: torch.Tensor
    labels: torch.Tensor
    annotations: Dict[str, torch.Tensor]

    def to(self: Self, device: Union[str, torch.device]) -> Self:
        return type(self)(
            trees=self.trees,
            sentences=self.sentences.to(device),
            tags=self.tags.to(device),
            heads=self.heads.to(device),
            labels=self.labels.to(device),
            annotations={k: v.to(device) for k, v in self.annotations.items()},
        )


class LRSchedule(TypedDict):
    shape: Literal["exponential", "linear", "constant"]
    warmup_steps: int


class AnnotationConfig(pydantic.BaseModel):
    hidden_layer_dim: int
    labels: List[str]
    loss_weight: float = 1.0


# TODO: look into <https://github.com/lincolnloop/goodconf/>
class BiAffineParserConfig(pydantic.BaseModel):
    mlp_input: int
    mlp_tag_hidden: int
    mlp_arc_hidden: int
    mlp_lab_hidden: int
    biased_biaffine: bool
    default_batch_size: int
    encoder_dropout: float
    extra_annotations: Dict[str, AnnotationConfig] = pydantic.Field(default_factory=dict)
    labels: List[str]
    mlp_dropout: float
    tagset: List[str]
    lexers: Dict[str, str]
    multitask_loss: Literal["adaptative", "mean", "sum", "weighted"]


class BiaffineParserOutput(NamedTuple):
    tag_scores: torch.Tensor
    head_scores: torch.Tensor
    deprel_scores: torch.Tensor
    extra_labels_scores: Dict[str, torch.Tensor]

    def unbatch(self, sentence_lengths: Sequence[int]) -> List["BiaffineParserOutput"]:
        """Return individual scores for every sentence in the batch, properly truncated to the
        sentence length."""
        transposed_extra_labels_scores: List[Dict[str, torch.Tensor]] = [
            dict() for _ in sentence_lengths
        ]
        # FIXME: this is too clunky
        if self.extra_labels_scores:
            for name, scores in self.extra_labels_scores.items():
                for scores_dict, label_scores, sent_len in zip(
                    transposed_extra_labels_scores, scores.unbind(0), sentence_lengths
                ):
                    scores_dict[name] = label_scores[:sent_len, :]

        return [
            type(self)(
                tag_scores=tag_scores[:sent_len, :],
                head_scores=head_scores[:sent_len, :sent_len],
                deprel_scores=deprel_scores[:sent_len, :sent_len, :],
                extra_labels_scores=extra_labels_scores,
            )
            for tag_scores, head_scores, deprel_scores, extra_labels_scores, sent_len in zip(
                self.tag_scores.unbind(0),
                self.head_scores.unbind(0),
                self.deprel_scores.unbind(0),
                transposed_extra_labels_scores,
                sentence_lengths,
            )
        ]


class ParserEvalOutput(NamedTuple):
    loss: float
    tag_accuracy: float
    head_accuracy: float
    deprel_accuracy: float
    extra_annotations_accuracy: Dict[str, float]


class BiAffineParser(nn.Module):
    """Biaffine Dependency Parser."""

    UNK_WORD: Final[str] = "<unk>"
    # Labels that are -100 are ignored in torch crossentropy (we still set it explicitely)
    LABEL_PADDING: Final[int] = -100

    # FIXME: `mlp_input` here is conterintuitive: the actual MLP input dim will be twice that, this
    # is more accurately the dimension of the outputs of each direction of the LSTM
    # FIXME: `default_batch_size` is discutable, since it's heavily machine-dependent
    def __init__(
        self,
        biased_biaffine: bool,
        default_batch_size: int,
        encoder_dropout: float,  # lstm dropout
        labels: Sequence[str],
        lexers: Mapping[str, lexers.Lexer],
        mlp_input: int,
        mlp_tag_hidden: int,
        mlp_arc_hidden: int,
        mlp_lab_hidden: int,
        mlp_dropout: float,
        tagset: Sequence[str],
        extra_annotations: Optional[Mapping[str, AnnotationConfig]] = None,
        multitask_loss: Literal["adaptative", "mean", "sum", "weighted"] = "sum",
    ):
        super(BiAffineParser, self).__init__()
        self.default_batch_size = default_batch_size
        self.tagset: BidirectionalMapping[str, int] = bidict((t, i) for i, t in enumerate(tagset))
        self.labels: BidirectionalMapping[str, int] = bidict((l, i) for i, l in enumerate(labels))

        self.mlp_arc_hidden: Final[int] = mlp_arc_hidden
        self.mlp_input: Final[int] = mlp_input
        self.mlp_tag_hidden: Final[int] = mlp_tag_hidden
        self.mlp_lab_hidden: Final[int] = mlp_lab_hidden
        self.mlp_dropout = mlp_dropout

        # TODO: fix typing here, casting works but it's inelegant
        # The issue is `Lexer` is a protocol and we can't require them to be torch Modules (which is
        # a concrete class and you can't ask a protocol to subclass a concrete class or ask for the
        # intersection of a protocol and a concrete class), but we do need them to be torch Modules
        # since we have to wrap them in a `ModuleDict` here`. Also we really don't want to separate
        # lexers that wouldn't be torch Modules since that's really never going to happen in
        # practice.
        self.lexers = cast(Dict[str, Lexer], nn.ModuleDict(cast(Dict[str, nn.Module], lexers)))
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
        # FIXME: why no dropout here?
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
        self.lab_biaffine = BiAffine(self.mlp_input, len(self.labels), bias=biased_biaffine)

        # Extra annotations
        _loss_weights = {
            "tag": 1.0,
            "head": 1.0,
            "deprel": 1.0,
        }
        # This makes us store the list of labels twice but makes it less clunky to __init__ and save
        self.annotation_lexicons: Dict[str, BidirectionalMapping[str, int]]
        self.extra_annotations: Dict[str, AnnotationConfig]
        if extra_annotations is not None:
            if (
                reserved := next(
                    (a for a in {"tag", "head", "deprel"} if a in extra_annotations), None
                )
            ) is not None:
                raise ValueError(f"Reserved name used in extra annotations: {reserved!r}")
            self.extra_annotations = dict(extra_annotations)
            self.annotation_lexicons = {
                name: bidict((l, i) for i, l in enumerate(conf.labels))
                for name, conf in extra_annotations.items()
            }
            self.annotators = nn.ModuleDict(
                {
                    name: MLP(
                        input_dim=self.mlp_input * 2,
                        hidden_dim=conf.hidden_layer_dim,
                        output_dim=len(conf.labels),
                        dropout=self.mlp_dropout,
                    )
                    for name, conf in extra_annotations.items()
                }
            )
            _loss_weights.update(
                {name: conf.loss_weight for name, conf in extra_annotations.items()}
            )
        else:
            self.extra_annotations = dict()
            self.annotation_lexicons = dict()
            self.annotators = nn.ModuleDict(dict())
        self.annotations_order = sorted(self.annotation_lexicons.keys())

        self.marginal_loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.LABEL_PADDING)
        self.multitask_loss: Literal["adaptative", "mean", "sum", "weighted"] = multitask_loss
        self.loss_weights = torch.tensor(
            [_loss_weights[name] for name in ["tag", "head", "deprel", *self.annotations_order]],
        )
        # NOTE: we really don't use the weights in the same way in adaptive mode, see
        # <https://aclanthology.org/2022.findings-acl.190>
        if self.multitask_loss == "adaptative":
            self.loss_weights = torch.nn.Parameter(self.loss_weights)

    def save_params(self, path: Union[str, pathlib.Path, BinaryIO]):
        torch.save(self.state_dict(), path)

    def load_params(self, path: Union[str, pathlib.Path, BinaryIO]):
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict)

    def forward(
        self,
        encodings: Dict[str, Any],
        sent_lengths: torch.Tensor,
    ) -> BiaffineParserOutput:
        """Predict POS, heads and deprel scores.

        ## Outputs

        `tag_scores, arc_scores, lab_scores` with shapes

        - `tag_score`: $`batch_size×max_sent_length×num_pos_tags`$
        - `arc_scores`: $`batch_size×n_deps×n_possible_heads`$
        - `label_scores`: $`batch_size×n_deps×n_possible_heads×num_deprels`$
        """
        embeddings = [
            self.lexers[lexer_name](encodings[lexer_name]) for lexer_name in self.lexers_order
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

        extra_labels_scores = dict()
        for name, annotator in self.annotators.items():
            extra_labels_scores[name] = annotator(dep_embeddings)

        return BiaffineParserOutput(
            tag_scores=tag_scores,
            head_scores=arc_scores,
            deprel_scores=lab_scores,
            extra_labels_scores=extra_labels_scores,
        )

    # TODO: make this an independent function
    # TODO: JIT this (or split it and jit the marginals?)
    def parser_loss(
        self,
        parser_output: BiaffineParserOutput,
        batch: DependencyBatch,
    ) -> torch.Tensor:
        all_loss: dict[str, torch.Tensor] = dict()

        batch_size = batch.heads.shape[0]
        num_padded_deps = batch.heads.shape[1]
        num_deprels = parser_output.deprel_scores.shape[-1]

        # TAGGER_LOSS
        # [batch×sent_length, num_tags]
        tagger_scores_flat = parser_output.tag_scores.view(-1, parser_output.tag_scores.size(-1))
        all_loss["tag"] = self.marginal_loss(tagger_scores_flat, batch.tags.view(-1))

        # ARC LOSS
        # [batch×num_deps, num_heads]
        head_scores_flat = parser_output.head_scores.view(-1, parser_output.head_scores.size(-1))
        all_loss["head"] = self.marginal_loss(head_scores_flat, batch.heads.view(-1))

        # LABEL LOSS We will select the labels for the gold heads, so we have to provide one when
        # there is none (either it is absent from the data or this is a padding token) to even if
        # they will be ignored in the crossentropy since the true label for that head is set to -100
        # so we give them the root.
        positive_heads = batch.heads.masked_fill(batch.heads.eq(self.LABEL_PADDING), 0)
        heads_selection = positive_heads.view(batch_size, num_padded_deps, 1, 1)
        # [batch, n_dependents, 1, n_labels]
        heads_selection = heads_selection.expand(batch_size, num_padded_deps, 1, num_deprels)
        # [batch, n_dependents, 1, n_labels]
        predicted_labels_scores = torch.gather(parser_output.deprel_scores, -2, heads_selection)

        # [batch×sent_len, n_labels]
        predicted_labels_scores_flat = predicted_labels_scores.view(-1, num_deprels)
        all_loss["deprel"] = self.marginal_loss(predicted_labels_scores_flat, batch.labels.view(-1))

        # Extra annotations loss
        for annotation_name, labels in batch.annotations.items():
            labels_scores = parser_output.extra_labels_scores[annotation_name]
            labels_scores_flat = labels_scores.view(-1, labels_scores.size(-1))
            all_loss[annotation_name] = self.marginal_loss(labels_scores_flat, labels.view(-1))

        loss = torch.stack(
            [all_loss[name] for name in ["tag", "head", "deprel", *self.annotations_order]]
        )
        # TODO: see if other loss combination functions wouldn't help here, tracked at
        # <https://github.com/hopsparser/npdependency/issues/59>
        if self.multitask_loss == "sum":
            return loss.sum()
        elif self.multitask_loss == "mean":
            return loss.sum() / len(batch.trees)
        elif self.multitask_loss == "weighted":
            return torch.inner(self.loss_weights, loss) / len(batch.trees)
        elif self.multitask_loss == "adaptative":
            return (
                torch.inner(1 / self.loss_weights**2, loss) + torch.log(self.loss_weights)
            ).sum()
        else:
            raise ValueError(f"Unknown loss type {self.multitask_loss}")

    def eval_model(
        self, dev_set: Iterable[DependencyBatch], batch_size: Optional[int] = None
    ) -> ParserEvalOutput:
        if batch_size is None:
            batch_size = self.default_batch_size

        self.eval()
        device = next(self.parameters()).device
        # NOTE: the accuracy scoring is approximative and cannot be interpreted as an UAS/LAS score
        # NOTE: fun project: track the correlation between them
        tags_tp = torch.zeros(1, dtype=torch.long, device=device)
        heads_tp = torch.zeros(1, dtype=torch.long, device=device)
        deprels_tp = torch.zeros(1, dtype=torch.long, device=device)
        extra_annotations_tp = {
            name: torch.zeros(1, dtype=torch.long, device=device) for name in self.annotations_order
        }
        overall_tags_size = torch.zeros(1, dtype=torch.long, device=device)
        overall_heads_size = torch.zeros(1, dtype=torch.long, device=device)
        overall_deprels_size = torch.zeros(1, dtype=torch.long, device=device)
        overall_extra_annotations_size = {
            name: torch.zeros(1, dtype=torch.long, device=device) for name in self.annotations_order
        }
        gloss = torch.zeros(1, dtype=torch.float, device=device)

        with torch.inference_mode():
            for batch in dev_set:
                batch = batch.to(device)

                # workaround since the design of torch modules makes it hard
                # for static analyzer to find out their return type
                output: BiaffineParserOutput = self(
                    batch.sentences.encodings, batch.sentences.sent_lengths
                )

                gloss += self.parser_loss(output, batch)

                # TODO: make all this batch-level eval a method
                # tagger accuracy
                tags_pred = output.tag_scores.argmax(dim=-1)
                tags_mask = batch.tags.ne(self.LABEL_PADDING)
                overall_tags_size += tags_mask.sum()
                tags_accuracy = tags_pred.eq(batch.tags).logical_and(tags_mask).sum()
                tags_tp += tags_accuracy

                # extra labels accuracy
                for name, scores in output.extra_labels_scores.items():
                    gold_annotation = batch.annotations[name]
                    annotation_pred = scores.argmax(dim=-1)
                    annotation_mask = gold_annotation.ne(self.LABEL_PADDING)
                    overall_extra_annotations_size[name] += annotation_mask.sum()
                    extra_annotations_tp[name] += (
                        annotation_pred.eq(gold_annotation).logical_and(annotation_mask).sum()
                    )

                # greedy head accuracy (without parsing)
                heads_preds = output.head_scores.argmax(dim=-1)
                heads_mask = batch.heads.ne(self.LABEL_PADDING)
                overall_heads_size += heads_mask.sum()
                heads_accuracy = heads_preds.eq(batch.heads).logical_and(heads_mask).sum()
                heads_tp += heads_accuracy

                # greedy deprel accuracy (without parsing)
                gold_heads_select = (
                    batch.heads.masked_fill(batch.heads.eq(self.LABEL_PADDING), 0)
                    .view(batch.heads.shape[0], batch.heads.shape[1], 1, 1)
                    .expand(
                        batch.heads.shape[0],
                        batch.heads.shape[1],
                        1,
                        output.deprel_scores.shape[-1],
                    )
                )
                # shape: num_padded_deps×num_padded_heads
                gold_head_deprels_scores = torch.gather(
                    output.deprel_scores, -2, gold_heads_select
                ).squeeze(-2)
                deprels_pred = gold_head_deprels_scores.argmax(dim=-1)
                deprels_mask = batch.labels.ne(self.LABEL_PADDING)
                overall_deprels_size += deprels_mask.sum()
                deprels_accuracy = deprels_pred.eq(batch.labels).logical_and(deprels_mask).sum()
                deprels_tp += deprels_accuracy.item()

        return ParserEvalOutput(
            loss=gloss.true_divide(
                overall_tags_size + overall_heads_size + overall_deprels_size
            ).item(),
            tag_accuracy=tags_tp.true_divide(overall_tags_size).item(),
            head_accuracy=heads_tp.true_divide(overall_heads_size).item(),
            deprel_accuracy=deprels_tp.true_divide(overall_deprels_size).item(),
            extra_annotations_accuracy={
                name: extra_annotations_tp[name]
                .true_divide(overall_extra_annotations_size[name])
                .item()
                for name in self.annotations_order
            },
        )

    def train_model(
        self,
        epochs: int,
        lr: float,
        lr_schedule: LRSchedule,
        model_path: Union[str, pathlib.Path],
        train_set: "DependencyDataset",
        batch_size: Optional[int] = None,
        dev_set: Optional["DependencyDataset"] = None,
        max_grad_norm: Optional[float] = None,
        log_epoch: Callable[[str, Dict[str, str]], Any] = lambda x, y: None,
    ):
        train_loader = cast(
            torch.utils.data.DataLoader[DepGraph],
            torch.utils.data.DataLoader(
                dataset=train_set, batch_size=batch_size, collate_fn=self.batch_trees, shuffle=True
            ),
        )
        if dev_set is not None:
            dev_loader = cast(
                torch.utils.data.DataLoader[DepGraph],
                torch.utils.data.DataLoader(
                    dataset=dev_set, batch_size=batch_size, collate_fn=self.batch_trees
                ),
            )
        else:
            dev_loader = None
        model_path = pathlib.Path(model_path)
        weights_file = model_path / "weights.pt"
        if batch_size is None:
            batch_size = self.default_batch_size
        device = next(self.parameters()).device
        logger.info(f"Start training on {device}")

        # TODO: make these configurable?
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.9), lr=lr, eps=1e-09)

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

            self.train()
            for batch in train_loader:
                overall_size += (
                    batch.tags.ne(self.LABEL_PADDING).sum()
                    + batch.heads.ne(self.LABEL_PADDING).sum()
                    + batch.labels.ne(self.LABEL_PADDING).sum()
                    + sum(ann.ne(self.LABEL_PADDING).sum() for ann in batch.annotations.values())
                )

                batch = cast(DependencyBatch, batch.to(device))

                # FORWARD
                output: BiaffineParserOutput = self(
                    batch.sentences.encodings, batch.sentences.sent_lengths
                )

                loss = self.parser_loss(output, batch)

                with torch.inference_mode():
                    train_loss += loss

                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()

            if dev_loader is not None:
                dev_scores = self.eval_model(dev_loader, batch_size=batch_size)
                # FIXME: this is not very elegant (2022-07)
                # FIXME: really not (2022-09)
                # FIXME: it's ok, lightning will save us
                log_epoch(
                    str(e),
                    {
                        "train loss": f"{train_loss.true_divide(overall_size).item():.4f}",
                        "dev loss": f"{dev_scores.loss:.4f}",
                        **{
                            k: f"{v:06.2%}"
                            for k, v in (
                                ("dev tag acc", dev_scores.tag_accuracy),
                                ("dev head acc", dev_scores.head_accuracy),
                                ("dev deprel acc", dev_scores.deprel_accuracy),
                                *(
                                    (f"dev {name} acc", annotation_accuracy)
                                    for name, annotation_accuracy in dev_scores.extra_annotations_accuracy.items()
                                ),
                            )
                            if not math.isnan(v)
                        },
                    },
                )

                # FIXME: probably change the logic here, esp. for headless data
                if dev_scores.head_accuracy > best_arc_acc:
                    logger.info(
                        f"New best model: head accuracy {dev_scores.head_accuracy:.2%} > {best_arc_acc:.2%}"
                    )
                    self.save_params(weights_file)
                    best_arc_acc = dev_scores.head_accuracy
                elif math.isnan(dev_scores.head_accuracy):
                    logger.debug("No head annotations in dev: saving model")
                    self.save_params(weights_file)
                    best_arc_acc = dev_scores.head_accuracy
            else:
                self.save_params(weights_file)

        self.load_params(weights_file)

    @overload
    def encode_sentence(
        self, words: Sequence[str], strict: Literal[True] = True
    ) -> EncodedSentence:
        pass

    @overload
    def encode_sentence(self, words: Sequence[str], strict: bool) -> Optional[EncodedSentence]:
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
            lexer_name: lexer.make_batch([sent.encodings[lexer_name] for sent in sentences])
            for lexer_name, lexer in self.lexers.items()
        }

        sent_lengths = torch.tensor([sent.sent_len for sent in sentences], dtype=torch.long)

        return SentencesBatch(
            words=words,
            encodings=encodings,
            sent_lengths=sent_lengths,
        )

    def encode_tree(self, tree: DepGraph) -> EncodedTree:
        sentence = self.encode_sentence(tree.words[1:])
        tag_idxes = torch.tensor(
            [
                self.tagset.get(tag, self.LABEL_PADDING) if tag is not None else self.LABEL_PADDING
                for tag in tree.pos_tags
            ],
            dtype=torch.long,
        )
        heads = torch.tensor(
            [h if h is not None else self.LABEL_PADDING for h in tree.heads],
            dtype=torch.long,
        )
        # FIXME: should unk labels be padding?
        labels = torch.tensor(
            [
                self.labels.get(lab, self.LABEL_PADDING)
                if lab is not None and h is not None
                else self.LABEL_PADDING
                for h, lab in zip(tree.heads, tree.deprels)
            ],
            dtype=torch.long,
        )
        # TODO: at this point we probably want to implement a strict mode to be safe
        # Double get here: this way, if the annotation isn't present in the MISC column OR if it's
        # present but with an unknown value
        # FIXME: maybe not a good idea actually since some labels might be implicit (like
        # SpaceAfter=yes)
        # FIXME: Padding for the root node, but probably a better idea to not even predict any label
        # for it if we can avoid it
        annotations = {
            name: torch.tensor(
                [
                    self.LABEL_PADDING,
                    *(
                        self.annotation_lexicons[name].get(
                            node.misc.mapping.get(name), self.LABEL_PADDING  # type: ignore
                        )
                        for node in tree.nodes
                    ),
                ]
            )
            for name in self.annotations_order
        }

        return EncodedTree(
            sentence=sentence,
            heads=heads,
            labels=labels,
            tags=tag_idxes,
            annotations=annotations,
            tree=tree,
        )

    def batch_trees(
        self,
        encoded_trees: Sequence[EncodedTree],
    ) -> DependencyBatch:
        """Batch encoded trees."""
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
        annotations = {
            name: pad_sequence(
                [tree.annotations[name] for tree in encoded_trees],
                batch_first=True,
                padding_value=self.LABEL_PADDING,
            )
            for name in self.annotations_order
        }

        return DependencyBatch(
            sentences=sentences,
            heads=heads,
            labels=labels,
            tags=tags,
            trees=[t.tree for t in encoded_trees],
            annotations=annotations,
        )

    def batched_predict(
        self,
        batch_lst: Union[Iterable[DependencyBatch], Iterable[SentencesBatch]],
        greedy: bool = False,
    ) -> Iterable[DepGraph]:
        self.eval()
        device = next(self.parameters()).device

        with torch.inference_mode():
            for batch in batch_lst:
                batch = batch.to(device)

                if isinstance(batch, DependencyBatch):
                    batch_sentences = batch.sentences
                else:
                    batch_sentences = batch
                # batch prediction
                output: BiaffineParserOutput = self(
                    batch_sentences.encodings, batch_sentences.sent_lengths
                )

                if isinstance(batch, DependencyBatch):
                    trees = batch.trees
                else:
                    trees = [DepGraph.from_words(words) for words in batch.words]
                for tree, tree_scores in zip(
                    trees, output.unbatch(sentence_lengths=batch_sentences.sent_lengths.tolist())
                ):
                    result_tree = self._scores_to_tree(
                        greedy=greedy,
                        scores=tree_scores,
                        tree=tree,
                    )
                    yield result_tree

    # FIXME: tags and extra annotations argmaxing is very batchable, which would make this faster
    # AND easier to read
    def _scores_to_tree(
        self,
        greedy: bool,
        scores: BiaffineParserOutput,
        tree: DepGraph,
    ):
        sentence_length = scores.deprel_scores.shape[0]

        # Predict heads
        if greedy:
            # shape: num_deps
            mst_heads = scores.head_scores.argmax(dim=1)
        else:
            head_scores_np = scores.head_scores.cpu().numpy()
            mst_heads_np = chuliu_edmonds(head_scores_np)
            # shape: num_deps
            mst_heads = torch.from_numpy(mst_heads_np).to(scores.deprel_scores.device)

        # TODO: we should remove the root here so we have one less node to argmax on
        # Predict labels
        # shape: num_deps×1×num_deprel
        select = mst_heads.view(-1, 1, 1).expand(-1, 1, scores.deprel_scores.size(-1))
        # shape: num_deps×num_deprel
        selected = torch.gather(scores.deprel_scores, 1, select).view(sentence_length, -1)
        mst_labels = selected.argmax(dim=-1)

        # Predict tags
        tag_idxes = scores.tag_scores.argmax(dim=1)

        # Predict extra annotations
        # `[1:]` to ignore the root node's labels
        misc_idx = {n: s.argmax(dim=1).tolist() for n, s in scores.extra_labels_scores.items()}

        # `[1:]` to ignore the root node's tag
        # TODO: use zip strict when we can drop py38 and py39, for this and the following
        # TODO: does tolist slow us down, here?
        # TODO: should we maintain a `node.identifier: index_in_tensor` dict for this? It's
        # unnecsesary right now but would make the management of the root cleaner and allow non-int
        # identifiers in the future
        pos_tags = {
            node.identifier: self.tagset.inverse[idx]
            for node, idx in zip(tree.nodes, tag_idxes[1:].tolist())
        }
        # `[1:]` to ignore the root node's head
        heads = {node.identifier: h for node, h in zip(tree.nodes, mst_heads[1:].tolist())}
        # `[1:]` to ignore the root node's deprel
        deprels = {
            node.identifier: self.labels.inverse[lbl]
            for node, lbl in zip(tree.nodes, mst_labels[1:].tolist())
        }

        misc = {
            node.identifier: {
                n: self.annotation_lexicons[n].inverse[idx[i]] for n, idx in misc_idx.items()
            }
            for i, node in enumerate(tree.nodes, start=1)
        }

        result_tree = tree.replace(
            deprels=deprels,
            heads=heads,
            misc=misc,
            pos_tags=pos_tags,
        )

        return result_tree

    def parse(
        self,
        inpt: Iterable[Union[str, Sequence[str]]],
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
                if line and not (isinstance(line, str) and line.isspace())
                for encoded in [
                    self.encode_sentence(
                        line.strip().split() if isinstance(line, str) else list(line), strict=strict
                    )
                ]
                if encoded is not None
            )
            batches = (
                self.batch_sentences(sentences_slice)
                for sentences_slice in cast(
                    Iterable[List[EncodedSentence]],
                    itu.chunked_iter(
                        sentences,
                        size=batch_size,
                    ),
                )
            )
        else:
            trees = DepGraph.read_conll(cast(Iterable[str], inpt))
            batches = (
                self.batch_trees([self.encode_tree(t) for t in batch])
                for batch in cast(
                    Iterable[List[DepGraph]], itu.chunked_iter(trees, size=batch_size)
                )
            )
        yield from self.batched_predict(batches, greedy=False)

    def save(self, model_path: pathlib.Path, save_weights: bool = True):
        model_path.mkdir(exist_ok=True, parents=True)
        config = BiAffineParserConfig(
            biased_biaffine=self.arc_biaffine.bias,
            default_batch_size=self.default_batch_size,
            encoder_dropout=self.dep_rnn.dropout,
            extra_annotations=self.extra_annotations,
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
            multitask_loss=self.multitask_loss,
        )
        config_file = model_path / "config.json"
        with open(config_file, "w") as out_stream:
            json.dump(config.dict(), out_stream)
        lexers_path = model_path / "lexers"
        for lexer_name, lexer in self.lexers.items():
            lexer.save(model_path=lexers_path / lexer_name, save_weights=False)
        if save_weights:
            torch.save(self.state_dict(), model_path / "weights.pt")

    # FIXME: allow passing a config dict directly
    @classmethod
    def initialize(
        cls: Type[Self],
        config_path: pathlib.Path,
        treebank: List[DepGraph],
    ) -> Self:
        logger.info(f"Initializing a parser from {config_path}")
        with open(config_path) as in_stream:
            config = yaml.load(in_stream, Loader=yaml.SafeLoader)
        config.setdefault("biased_biaffine", True)
        config.setdefault("batch_size", 1)
        config.setdefault("multitask_loss", "sum")

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
                        fasttext_model_path = (config_path.parent / fasttext_model_path).resolve()
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
                        logger.info(f"Generating a FastText model from {fasttext_model_path}")
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
        if (extra_annotations_config := config.get("extra_annotations")) is not None:
            annotations_labels = gen_annotations_labels(
                treebank, sorted(extra_annotations_config.keys())
            )
            extra_annotations = {
                name: AnnotationConfig(labels=annotations_labels[name], **conf)
                for name, conf in extra_annotations_config.items()
            }
        else:
            extra_annotations = None

        return cls(
            labels=itolab,
            lexers=parser_lexers,
            tagset=itotag,
            biased_biaffine=config["biased_biaffine"],
            default_batch_size=config["batch_size"],
            extra_annotations=extra_annotations,
            encoder_dropout=config["encoder_dropout"],
            mlp_input=config["mlp_input"],
            mlp_tag_hidden=config["mlp_tag_hidden"],
            mlp_arc_hidden=config["mlp_arc_hidden"],
            mlp_lab_hidden=config["mlp_lab_hidden"],
            mlp_dropout=config["mlp_dropout"],
            multitask_loss=config["multitask_loss"],
        )

    @classmethod
    def load(
        cls,
        model_path: Union[str, pathlib.Path],
    ) -> Self:
        model_path = pathlib.Path(model_path)
        config_path = model_path / "config.json"

        with open(config_path) as in_stream:
            config_dict = json.load(in_stream)
        config = BiAffineParserConfig.model_validate(config_dict)

        lexers_path = model_path / "lexers"

        parser_lexers: Dict[str, Lexer] = dict()
        for lexer_name, lexer_type in config.lexers.items():
            lexer_class = lexers.LEXER_TYPES[lexer_type]
            parser_lexers[lexer_name] = lexer_class.load(lexers_path / lexer_name)

        parser = cls(
            extra_annotations=config.extra_annotations,
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
            multitask_loss=config.multitask_loss,
            tagset=config.tagset,
        )
        weights_file = model_path / "weights.pt"
        if weights_file.exists():
            parser.load_params(str(weights_file))

        return parser


class DependencyDataset(torch.utils.data.Dataset[DepGraph]):
    def __init__(
        self,
        parser: BiAffineParser,
        treelist: Iterable[DepGraph],
        skip_unencodable: bool = True,
    ):
        self.parser = parser
        self.treelist: List[DepGraph] = []
        self.encoded_trees: List[EncodedTree] = []
        for tree in treelist:
            try:
                encoded = self.parser.encode_tree(tree)
            except lexers.LexingError as e:
                if not skip_unencodable:
                    raise e
                else:
                    logger.info(
                        f"Skipping tree {e.sentence} due to lexing error '{e.message}'.",
                    )
                    continue
            self.encoded_trees.append(encoded)

    def __getitem__(self, index: int) -> EncodedTree:
        return self.encoded_trees[index]

    def __len__(self):
        return len(self.encoded_trees)


def train(
    config_file: pathlib.Path,
    model_path: pathlib.Path,
    train_file: pathlib.Path,
    dev_file: Optional[pathlib.Path] = None,
    skip_unencodable: bool = True,
    device: Union[str, torch.device] = "cpu",
    max_tree_length: Optional[int] = None,
    log_epoch: Callable[[str, Dict[str, str]], Any] = utils.log_epoch,
    overwrite: bool = False,
    rand_seed: Optional[int] = None,
):
    if rand_seed is not None:
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)

    with open(config_file) as in_stream:
        hp = yaml.load(in_stream, Loader=yaml.SafeLoader)

    with open(train_file) as in_stream:
        traintrees = list(DepGraph.read_conll(in_stream, max_tree_length=max_tree_length))
    model_path_not_empty = model_path.exists() and any(model_path.iterdir())
    if model_path_not_empty and not overwrite:
        logger.info(f"Continuing training from {model_path}")
        parser = BiAffineParser.load(model_path)
    else:
        if overwrite:
            if model_path_not_empty:
                logger.info(
                    f"Erasing existing trained model in {model_path} since overwrite was asked",
                )
                shutil.rmtree(model_path)
            else:
                logger.warning(f"--overwrite asked but {model_path} does not exist or is empty")
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
            warnings.warn(f"I can't freeze a {lexer_to_freeze_name!r} lexer that does not exist")
    parser.save(model_path)

    trainset = DependencyDataset(
        parser,
        traintrees,
        skip_unencodable=skip_unencodable,
    )
    devset: Optional[DependencyDataset]
    if dev_file is not None:
        with open(dev_file) as in_stream:
            # NOTE: skip_unencodable here **could** make sense, but in most cases we will want to
            # parse the whole dev set to get comparable global metrics anyway, so it's better to
            # fail here.
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
        log_epoch=log_epoch,
        lr=lr_config["base"],
        lr_schedule=lr_config.get("schedule", {"shape": "exponential", "warmup_steps": 0}),
        max_grad_norm=hp.get("max_grad_norm"),
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
        for tree in parser.parse(inpt=in_stream, batch_size=batch_size, raw=raw, strict=strict):
            ostream.write(tree.to_conllu())
            ostream.write("\n\n")
