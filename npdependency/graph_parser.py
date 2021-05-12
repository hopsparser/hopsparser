import argparse
import math
import os.path
import pathlib
import random
import shutil
import sys
import tempfile
import warnings
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    IO,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import torch
import transformers
import yaml
from boltons import iterutils as itu
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from npdependency import lexers
from npdependency.utils import smart_open
from npdependency import conll2018_eval as evaluator
from npdependency import deptree
from npdependency.deptree import (
    DependencyBatch,
    DependencyDataset,
    DepGraph,
    gen_labels,
    gen_tags,
)
from npdependency.lexers import (
    BertBaseLexer,
    BertLexerBatch,
    CharRNNLexer,
    DefaultLexer,
    FastTextLexer,
    freeze_module,
    make_vocab,
)
from npdependency.mst import chuliu_edmonds_one_root as chuliu_edmonds

# Python 3.7 shim
try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict  # type: ignore


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(MLP, self).__init__()
        self.Wdown = nn.Linear(input_size, hidden_size)
        self.Wup = nn.Linear(hidden_size, output_size)
        self.g = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        return self.Wup(self.dropout(self.g(self.Wdown(input))))


# Note: This is the biaffine layer used in Qi et al. (2018) and Dozat and Manning (2017).
class BiAffine(nn.Module):
    """Biaffine attention layer."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        weight_input = input_dim + 1 if bias else input_dim
        self.weight = nn.Parameter(
            torch.FloatTensor(output_dim, weight_input, weight_input)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.bias:
            h = torch.cat((h, h.new_ones((*h.shape[:-1], 1))), dim=-1)
            d = torch.cat((d, d.new_ones((*d.shape[:-1], 1))), dim=-1)
        return torch.einsum("bxi,oij,byj->boxy", h, self.weight, d)


class Tagger(nn.Module):
    def __init__(self, input_dim, tagset_size):
        super(Tagger, self).__init__()
        self.W = nn.Linear(input_dim, tagset_size)

    def forward(self, input):
        return self.W(input)


class LRSchedule(TypedDict):
    shape: Literal["exponential", "linear", "constant"]
    warmup_steps: int


class EncodedSentence(NamedTuple):
    words: Sequence[str]
    encoded_words: Union[torch.Tensor, lexers.BertLexerSentence]
    subwords: torch.Tensor
    chars: torch.Tensor
    sent_len: int


_T_SentencesBatch = TypeVar("_T_SentencesBatch", bound="SentencesBatch")


class SentencesBatch(NamedTuple):
    """Batched and padded sentences.

    ## Attributes

    - `words` The word forms for every sentence in the batch
    - `encoded_words` The words of the sentences, encoded and batched by a lexer and meant to be consumed by
      it directly. The details stay opaque at this level, see the relevant lexer instead.
    - `subwords` Encoded FastText subwords as a sequence of `LongTensor`. As with `chars`,
      `subwords[i][j, k]` is the k-th subword of the i-th word of the j-th sentence in the batch.
    - `chars` Encoded chars as a sequence of `LongTensor`. `chars[i][j, k]` is the k-th character of
      the i-th word of the j-th sentence in the batch.
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
    encoded_words: Union[torch.Tensor, BertLexerBatch]
    subwords: torch.Tensor
    chars: torch.Tensor
    sent_lengths: torch.Tensor
    content_mask: torch.Tensor

    def to(
        self: _T_SentencesBatch, device: Union[str, torch.device]
    ) -> _T_SentencesBatch:
        return type(self)(
            words=self.words,
            encoded_words=self.encoded_words.to(device),
            chars=self.chars.to(device),
            subwords=self.subwords.to(device),
            sent_lengths=self.sent_lengths,
            content_mask=self.content_mask.to(device),
        )


class BiAffineParser(nn.Module):
    """Biaffine Dependency Parser."""

    def __init__(
        self,
        biased_biaffine: bool,
        chars_lexer: CharRNNLexer,
        default_batch_size: int,
        device: Union[str, torch.device],
        encoder_dropout: float,  # lstm dropout
        ft_lexer: FastTextLexer,
        labels: Sequence[str],
        lexer: Union[DefaultLexer, BertBaseLexer],
        mlp_input: int,
        mlp_tag_hidden: int,
        mlp_arc_hidden: int,
        mlp_lab_hidden: int,
        mlp_dropout: float,
        tagset: Sequence[str],
    ):

        super(BiAffineParser, self).__init__()
        self.default_batch_size = default_batch_size
        self.device = torch.device(device)
        self.tagset = tagset
        self.labels = labels
        self.mlp_arc_hidden = mlp_arc_hidden
        self.mlp_input = mlp_input
        self.mlp_lab_hidden = mlp_lab_hidden

        self.lexer = lexer.to(self.device)

        self.dep_rnn = nn.LSTM(
            self.lexer.embedding_size
            + chars_lexer.embedding_size
            + ft_lexer.embedding_size,
            mlp_input,
            3,
            batch_first=True,
            dropout=encoder_dropout,
            bidirectional=True,
        ).to(self.device)

        # POS tagger & char RNN
        self.pos_tagger = MLP(mlp_input * 2, mlp_tag_hidden, len(self.tagset)).to(
            self.device
        )
        self.char_rnn = chars_lexer.to(self.device)
        self.ft_lexer = ft_lexer.to(self.device)

        # Arc MLPs
        self.arc_mlp_h = MLP(mlp_input * 2, mlp_arc_hidden, mlp_input, mlp_dropout).to(
            self.device
        )
        self.arc_mlp_d = MLP(mlp_input * 2, mlp_arc_hidden, mlp_input, mlp_dropout).to(
            self.device
        )
        # Label MLPs
        self.lab_mlp_h = MLP(mlp_input * 2, mlp_lab_hidden, mlp_input, mlp_dropout).to(
            self.device
        )
        self.lab_mlp_d = MLP(mlp_input * 2, mlp_lab_hidden, mlp_input, mlp_dropout).to(
            self.device
        )

        # BiAffine layers
        self.arc_biaffine = BiAffine(mlp_input, 1, bias=biased_biaffine).to(self.device)
        self.lab_biaffine = BiAffine(
            mlp_input, len(self.labels), bias=biased_biaffine
        ).to(self.device)

    def save_params(self, path: Union[str, pathlib.Path, BinaryIO]):
        torch.save(self.state_dict(), path)

    def load_params(self, path: Union[str, pathlib.Path, BinaryIO]):
        state_dict = torch.load(path, map_location=self.device)
        # Legacy models do not have BERT layer weights, so we inject them here they always use only
        # 4 layers so we don't have to guess the size of the weight vector
        if hasattr(self.lexer, "layers_gamma"):
            state_dict.setdefault(
                "lexer.layer_weights", torch.ones(4, dtype=torch.float)
            )
            state_dict.setdefault(
                "lexer.layers_gamma", torch.ones(1, dtype=torch.float)
            )
        self.load_state_dict(state_dict)

    def forward(
        self,
        words: Union[torch.Tensor, BertLexerBatch],
        chars: torch.Tensor,
        ft_subwords: torch.Tensor,
        sent_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict POS, heads and deprel scores.

        ## Outputs

        `tag_scores, arc_scores, lab_scores` with shapes

        - `tag_score`: $`batch_size×max_sent_length×num_pos_tags`$
        - `arc_scores`: $`batch_size×max_sent_length×max_sent_length`$
        - `label_scores`: $`batch_size×num_deprels×max_sent_length×max_sent_length`$
        """
        # Computes char embeddings
        char_embed = self.char_rnn(chars)
        # Computes fasttext embeddings
        ft_embed = self.ft_lexer(ft_subwords)
        # Computes word embeddings
        lex_emb = self.lexer(words)

        # Encodes input for tagging and parsing
        inpt = torch.cat((lex_emb, char_embed, ft_embed), dim=-1)
        packed_inpt = pack_padded_sequence(
            inpt, sent_lengths, batch_first=True, enforce_sorted=False
        )
        packed_dep_embeddings, _ = self.dep_rnn(packed_inpt)
        dep_embeddings, _ = pad_packed_sequence(packed_dep_embeddings, batch_first=True)

        # Tagging
        tag_scores = self.pos_tagger(dep_embeddings)

        # Compute the score matrices for the arcs and labels.
        arc_h = self.arc_mlp_h(dep_embeddings)
        arc_d = self.arc_mlp_d(dep_embeddings)
        lab_h = self.lab_mlp_h(dep_embeddings)
        lab_d = self.lab_mlp_d(dep_embeddings)

        arc_scores = self.arc_biaffine(arc_h, arc_d).squeeze(1)
        lab_scores = self.lab_biaffine(lab_h, lab_d)

        return tag_scores, arc_scores, lab_scores

    def parser_loss(
        self,
        tagger_scores: torch.Tensor,
        arc_scores: torch.Tensor,
        lab_scores: torch.Tensor,
        batch: DependencyBatch,
        marginal_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        # ARC LOSS
        # [batch, sent_len, sent_len]
        arc_scoresL = arc_scores.transpose(-1, -2)
        # [batch*sent_len, sent_len]
        arc_scoresL = arc_scoresL.reshape(-1, arc_scoresL.size(-1))
        # [batch*sent_len]
        arc_loss = marginal_loss(arc_scoresL, batch.heads.view(-1))

        # TAGGER_LOSS
        tagger_scoresB = tagger_scores.view(-1, tagger_scores.size(-1))
        tagger_loss = marginal_loss(tagger_scoresB, batch.tags.view(-1))

        # LABEL LOSS
        # We will select the labels for the true heads, so we have to give a true head to
        # the padding tokens (even if they will be ignored in the crossentropy since the true
        # label for that head is set to -100) so we give them the root.
        positive_heads = batch.heads.masked_fill(batch.content_mask.logical_not(), 0)
        # [batch, 1, 1, sent_len]
        headsL = positive_heads.unsqueeze(1).unsqueeze(2)
        # [batch, n_labels, 1, sent_len]
        headsL = headsL.expand(-1, lab_scores.size(1), -1, -1)
        # [batch, n_labels, sent_len]
        lab_scoresL = torch.gather(lab_scores, 2, headsL).squeeze(2)
        # [batch, sent_len, n_labels]
        lab_scoresL = lab_scoresL.transpose(-1, -2)
        # [batch*sent_len, n_labels]
        lab_scoresL = lab_scoresL.reshape(-1, lab_scoresL.size(-1))
        # [batch*sent_len]
        labelsL = batch.labels.view(-1)
        lab_loss = marginal_loss(lab_scoresL, labelsL)

        # TODO: see if other loss combination functions wouldn't help here, e.g.
        # <https://arxiv.org/abs/1805.06334>
        return tagger_loss + arc_loss + lab_loss

    def eval_model(self, dev_set: DependencyDataset, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.default_batch_size

        loss_fnc = nn.CrossEntropyLoss(
            reduction="sum", ignore_index=dev_set.LABEL_PADDING
        )

        self.eval()

        dev_batches = dev_set.make_batches(
            batch_size, shuffle_batches=False, shuffle_data=False
        )
        # NOTE: the accuracy scoring is approximative and cannot be interpreted as an UAS/LAS score
        # NOTE: fun project: track the correlation between them
        tag_acc, arc_acc, lab_acc, gloss = 0, 0, 0, 0.0
        overall_size = 0

        with torch.no_grad():
            for batch in dev_batches:
                overall_size += int(batch.content_mask.sum().item())

                batch = batch.to(self.device)

                # preds
                tagger_scores, arc_scores, lab_scores = self(
                    batch.encoded_words, batch.chars, batch.subwords, batch.sent_lengths
                )

                gloss += self.parser_loss(
                    tagger_scores, arc_scores, lab_scores, batch, loss_fnc
                ).item()

                # greedy arc accuracy (without parsing)
                arc_pred = arc_scores.argmax(dim=-2)
                arc_accuracy = (
                    arc_pred.eq(batch.heads).logical_and(batch.content_mask).sum()
                )
                arc_acc += arc_accuracy.item()

                # tagger accuracy
                tag_pred = tagger_scores.argmax(dim=2)
                tag_accuracy = (
                    tag_pred.eq(batch.tags).logical_and(batch.content_mask).sum()
                )
                tag_acc += tag_accuracy.item()

                # greedy label accuracy (without parsing)
                lab_pred = lab_scores.argmax(dim=1)
                lab_pred = torch.gather(
                    lab_pred,
                    1,
                    batch.heads.masked_fill(
                        batch.content_mask.logical_not(), 0
                    ).unsqueeze(1),
                ).squeeze(1)
                lab_accuracy = (
                    lab_pred.eq(batch.labels).logical_and(batch.content_mask).sum()
                )
                lab_acc += lab_accuracy.item()

        return (
            gloss / overall_size,
            tag_acc / overall_size,
            arc_acc / overall_size,
            lab_acc / overall_size,
        )

    def train_model(
        self,
        train_set: DependencyDataset,
        epochs: int,
        lr: float,
        lr_schedule: LRSchedule,
        model_path: Union[str, pathlib.Path],
        batch_size: Optional[int] = None,
        dev_set: Optional[DependencyDataset] = None,
    ):
        model_path = pathlib.Path(model_path)
        weights_file = model_path / "model.pt"
        if batch_size is None:
            batch_size = self.default_batch_size
        print(f"Start training on {self.device}")
        loss_fnc = nn.CrossEntropyLoss(
            reduction="sum", ignore_index=train_set.LABEL_PADDING
        )

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
            scheduler = transformers.get_linear_constant_with_warmup(
                optimizer, lr_schedule["warmup_steps"]
            )
        else:
            raise ValueError(f"Unkown lr schedule shape {lr_schedule['shape']!r}")

        for e in range(epochs):
            train_loss = 0.0
            best_arc_acc = 0.0
            overall_size = 0
            train_batches = train_set.make_batches(
                batch_size,
                shuffle_batches=True,
                shuffle_data=True,
            )
            self.train()
            for batch in train_batches:
                overall_size += int(batch.content_mask.sum().item())

                batch = batch.to(self.device)

                # FORWARD
                tagger_scores, arc_scores, lab_scores = self(
                    batch.encoded_words, batch.chars, batch.subwords, batch.sent_lengths
                )

                loss = self.parser_loss(
                    tagger_scores, arc_scores, lab_scores, batch, loss_fnc
                )
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            if dev_set is not None:
                dev_loss, dev_tag_acc, dev_arc_acc, dev_lab_acc = self.eval_model(
                    dev_set, batch_size=batch_size
                )
                print(
                    f"Epoch {e} train mean loss {train_loss / overall_size}"
                    f" valid mean loss {dev_loss} valid tag acc {dev_tag_acc} valid arc acc {dev_arc_acc} valid label acc {dev_lab_acc}"
                    f" Base LR {scheduler.get_last_lr()[0]}"
                )

                if dev_arc_acc > best_arc_acc:
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
            encoded = EncodedSentence(
                words=words,
                encoded_words=self.lexer.encode(words_with_root),
                subwords=self.ft_lexer.encode(words_with_root),
                chars=self.char_rnn.encode(words_with_root),
                sent_len=len(words_with_root),
            )
        except lexers.LexingError as e:
            if strict:
                raise e
            else:
                print(
                    f"Skipping sentence {e.sentence} due to lexing error '{e.message}'.",
                    file=sys.stderr,
                )
                return None
        return encoded

    def batch_sentences(self, sentences: Sequence[EncodedSentence]) -> SentencesBatch:
        words = [sent.words for sent in sentences]
        # TODO: fix the typing here
        encoded_words = self.lexer.make_batch([sent.encoded_words for sent in sentences])  # type: ignore[misc]
        chars = self.char_rnn.make_batch([sent.chars for sent in sentences])
        subwords = self.ft_lexer.make_batch([sent.subwords for sent in sentences])

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
            encoded_words=encoded_words,
            chars=chars,
            subwords=subwords,
            content_mask=content_mask,
            sent_lengths=sent_lengths,
        )

    def batched_predict(
        self,
        batch_lst: Union[Iterable[DependencyBatch], Iterable[SentencesBatch]],
        ostream: IO[str],
        greedy: bool = False,
    ):
        self.eval()

        with torch.no_grad():
            for batch in batch_lst:
                batch = batch.to(self.device)

                # batch prediction
                tagger_scores_batch, arc_scores_batch, lab_scores_batch = self(
                    batch.encoded_words,
                    batch.chars,
                    batch.subwords,
                    batch.sent_lengths,
                )
                arc_scores_batch = arc_scores_batch.cpu()

                if isinstance(batch, DependencyBatch):
                    trees = batch.trees
                else:
                    trees = [
                        DepGraph(
                            nodes=[
                                deptree.DepNode(
                                    identifier=i,
                                    form=w,
                                    lemma="_",
                                    upos="_",
                                    xpos="_",
                                    feats="_",
                                    head=0,
                                    deprel="_",
                                    deps="_",
                                    misc="_",
                                )
                                for i, w in enumerate(words, start=1)
                            ],
                        )
                        for words in batch.words
                    ]
                for (tree, length, tagger_scores, arc_scores, lab_scores) in zip(
                    trees,
                    batch.sent_lengths,
                    tagger_scores_batch,
                    arc_scores_batch,
                    lab_scores_batch,
                ):
                    # Predict heads
                    probs = arc_scores.numpy().T
                    batch_width = probs.shape[0]
                    mst_heads_np = (
                        np.argmax(probs[:length, :length], axis=1)
                        if greedy
                        else chuliu_edmonds(probs[:length, :length])
                    )
                    mst_heads = torch.from_numpy(
                        np.pad(mst_heads_np, (0, batch_width - length))
                    ).to(self.device)

                    # Predict tags
                    tag_idxes = tagger_scores.argmax(dim=1)
                    pos_tags = [self.tagset[idx] for idx in tag_idxes]
                    # Predict labels
                    select = mst_heads.unsqueeze(0).expand(lab_scores.size(0), -1)
                    selected = torch.gather(lab_scores, 1, select.unsqueeze(1)).squeeze(
                        1
                    )
                    mst_labels = selected.argmax(dim=0)
                    edges = [
                        deptree.Edge(head.item(), self.labels[lbl], dep)
                        for (dep, lbl, head) in zip(
                            list(range(length)), mst_labels, mst_heads
                        )
                    ]
                    result_tree = tree.replace(
                        edges=edges[1:],
                        pos_tags=pos_tags[1:],
                    )
                    print(str(result_tree), file=ostream, end="\n\n")

    @classmethod
    def initialize(
        cls,
        config_path: pathlib.Path,
        model_path: pathlib.Path,
        treebank: List[DepGraph],
        overrides: Optional[Dict[str, Any]] = None,
        fasttext: Optional[pathlib.Path] = None,
    ) -> "BiAffineParser":
        model_path.mkdir(parents=True, exist_ok=False)
        # TODO: remove this once we have a proper full save method?
        model_config_path = model_path / "config.yaml"
        shutil.copy(config_path, model_config_path)
        fasttext_model_path = model_path / "fasttext_model.bin"
        if fasttext is None:
            print("Generating a FastText model from the treebank")
            FastTextLexer.train_model_from_sents(
                [tree.words[1:] for tree in treebank], fasttext_model_path
            )
        elif fasttext.exists():
            try:
                # ugly, but we have no better way of checking if a file is a valid model
                FastTextLexer.load(fasttext)
                print(f"Using the FastText model at {fasttext}")
                shutil.copy(fasttext, fasttext_model_path)
            except ValueError:
                # FastText couldn't load it, so it should be raw text
                print(f"Generating a FastText model from {fasttext}")
                FastTextLexer.train_model_from_raw(fasttext, fasttext_model_path)
        else:
            raise ValueError(f"{fasttext} not found")

        # NOTE: these include the [ROOT] token, which will thus automatically have a dedicated
        # word embeddings in layers based on this vocab
        ordered_vocab = make_vocab(
            [word for tree in treebank for word in tree.words],
            0,
            unk_word=DependencyDataset.UNK_WORD,
            pad_token=DependencyDataset.PAD_TOKEN,
        )
        savelist(ordered_vocab, model_path / "vocab.lst")

        # FIXME: This should be done by the lexer class
        savelist(
            sorted(set((c for word in ordered_vocab for c in word))),
            model_path / "charcodes.lst",
        )

        itolab = gen_labels(treebank)
        savelist(itolab, model_path / "labcodes.lst")

        itotag = gen_tags(treebank)
        savelist(itotag, model_path / "tagcodes.lst")

        return cls.load(model_path, overrides)

    @classmethod
    def load(
        cls,
        model_path: Union[str, pathlib.Path],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "BiAffineParser":
        if overrides is None:
            overrides = dict()
        # TODO: move the initialization code to initialize (even if that duplicates code?)
        model_path = pathlib.Path(model_path)
        if model_path.is_dir():
            config_path = model_path / "config.yaml"
            if not config_path.exists():
                raise ValueError(f"No config in {model_path}")
        else:
            warnings.warn(
                "Loading a model from a YAML file is deprecated and will be removed in a future version.",
                category=FutureWarning,
            )
            config_path = model_path
            model_path = model_path.parent
        print(f"Initializing a parser from {model_path}")

        with open(config_path) as in_stream:
            hp = yaml.load(in_stream, Loader=yaml.SafeLoader)
        hp.update(overrides)
        hp.setdefault("device", "cpu")

        # FIXME: put that in the word lexer class
        ordered_vocab = loadlist(model_path / "vocab.lst")

        # TODO: separate the BERT and word lexers
        lexer: Union[DefaultLexer, BertBaseLexer]
        if hp["lexer"] == "default":
            lexer = DefaultLexer(
                ordered_vocab,
                hp["word_embedding_size"],
                hp["word_dropout"],
                words_padding_idx=DependencyDataset.PAD_IDX,
                unk_word=DependencyDataset.UNK_WORD,
            )
        else:
            bert_config_path = model_path / "bert_config"
            if bert_config_path.exists():
                bert_model = str(bert_config_path)
            else:
                bert_model = hp["lexer"]
            bert_layers = hp.get("bert_layers", [4, 5, 6, 7])
            if bert_layers == "*":
                bert_layers = None
            lexer = BertBaseLexer(
                bert_model=bert_model,
                bert_layers=bert_layers,
                bert_subwords_reduction=hp.get("bert_subwords_reduction", "first"),
                bert_weighted=hp.get("bert_weighted", False),
                embedding_size=hp["word_embedding_size"],
                itos=ordered_vocab,
                unk_word=DependencyDataset.UNK_WORD,
                words_padding_idx=DependencyDataset.PAD_IDX,
                word_dropout=hp["word_dropout"],
            )
            if not bert_config_path.exists():
                lexer.bert.config.save_pretrained(bert_config_path)
                lexer.bert_tokenizer.save_pretrained(
                    bert_config_path, legacy_format=not lexer.bert_tokenizer.is_fast
                )
                # Saving local paths in config is of little use and leaks information
                if os.path.exists(hp["lexer"]):
                    hp["lexer"] = "."

        chars_lexer = CharRNNLexer(
            charset=loadlist(model_path / "charcodes.lst"),
            special_tokens=[DepGraph.ROOT_TOKEN],
            char_embedding_size=hp["char_embedding_size"],
            embedding_size=hp["charlstm_output_size"],
        )

        ft_lexer = FastTextLexer.load(
            str(model_path / "fasttext_model.bin"), special_tokens=[DepGraph.ROOT_TOKEN]
        )

        itolab = loadlist(model_path / "labcodes.lst")
        itotag = loadlist(model_path / "tagcodes.lst")
        parser = cls(
            biased_biaffine=hp.get("biased_biaffine", True),
            device=hp["device"],
            chars_lexer=chars_lexer,
            default_batch_size=hp.get("batch_size", 1),
            encoder_dropout=hp["encoder_dropout"],
            ft_lexer=ft_lexer,
            labels=itolab,
            lexer=lexer,
            mlp_input=hp["mlp_input"],
            mlp_tag_hidden=hp["mlp_tag_hidden"],
            mlp_arc_hidden=hp["mlp_arc_hidden"],
            mlp_lab_hidden=hp["mlp_lab_hidden"],
            mlp_dropout=hp["mlp_dropout"],
            tagset=itotag,
        )
        weights_file = model_path / "model.pt"
        if weights_file.exists():
            parser.load_params(str(weights_file))
        else:
            parser.save_params(str(weights_file))
            # We were actually initializing — rather than loading — the model, let's save the
            # config with our changes
            with open(config_path, "w") as out_stream:
                yaml.dump(hp, out_stream)

        if hp.get("freeze_fasttext", False):
            freeze_module(ft_lexer)
        if hp.get("freeze_bert", False):
            if isinstance(lexer, lexers.BertBaseLexer):
                freeze_module(lexer.bert)
            else:
                warnings.warn(
                    "A non-BERT lexer has no BERT to freeze, ignoring `freeze_bert` hyperparameter"
                )
        return parser


def savelist(strlist, filename):
    with open(filename, "w") as ostream:
        ostream.write("\n".join(strlist))


def loadlist(filename):
    with open(filename) as istream:
        strlist = [line.strip() for line in istream]
    return strlist


def train(
    config_file: pathlib.Path,
    model_path: pathlib.Path,
    train_file: pathlib.Path,
    fasttext: Optional[pathlib.Path],
    max_tree_length: Optional[int] = None,
    overrides: Optional[Dict[str, str]] = None,
    overwrite: bool = False,
    rand_seed: Optional[int] = None,
    dev_file: Optional[pathlib.Path] = None,
):
    if rand_seed is not None:
        random.seed(rand_seed)
        torch.manual_seed(rand_seed)

    with open(config_file) as in_stream:
        hp = yaml.load(in_stream, Loader=yaml.SafeLoader)
    if "device" in hp:
        warnings.warn(
            "Setting a device directly in a configuration file is DEPRECATED and will be silently ignored in a future version.",
            category=FutureWarning,
        )

    traintrees = list(DepGraph.read_conll(train_file, max_tree_length=max_tree_length))
    if model_path.exists() and not overwrite:
        print(f"Continuing training from {model_path}", file=sys.stderr)
        parser = BiAffineParser.load(model_path, overrides)
    else:
        if overwrite:
            print(
                f"Erasing existing trained model in {model_path} since overwrite was asked",
                file=sys.stderr,
            )
            shutil.rmtree(model_path)
        parser = BiAffineParser.initialize(
            config_path=config_file,
            model_path=model_path,
            overrides=overrides,
            treebank=traintrees,
            fasttext=fasttext,
        )

    trainset = DependencyDataset(
        traintrees,
        parser.lexer,
        parser.char_rnn,
        parser.ft_lexer,
        use_labels=parser.labels,
        use_tags=parser.tagset,
    )
    devset: Optional[DependencyDataset]
    if dev_file is not None:
        devset = DependencyDataset(
            list(DepGraph.read_conll(dev_file)),
            parser.lexer,
            parser.char_rnn,
            parser.ft_lexer,
            use_labels=parser.labels,
            use_tags=parser.tagset,
        )
    else:
        devset = None

    parser.train_model(
        batch_size=hp["batch_size"],
        dev_set=devset,
        epochs=hp["epochs"],
        lr=hp["lr"],
        lr_schedule=hp.get("lr_schedule", {"shape": "exponential", "warmup_steps": 0}),
        model_path=model_path,
        train_set=trainset,
    )


def parse(
    model_path: Union[str, pathlib.Path],
    in_file: Union[str, pathlib.Path, IO[str]],
    out_file: Union[str, pathlib.Path, IO[str]],
    batch_size: Optional[int] = None,
    overrides: Optional[Dict[str, str]] = None,
    raw: bool = False,
    strict: bool = True,
):
    parser = BiAffineParser.load(model_path, overrides)
    if batch_size is None:
        batch_size = parser.default_batch_size
    print("Encoding", file=sys.stderr)
    with smart_open(in_file) as in_stream:
        batches: Union[Iterable[DependencyBatch], Iterable[SentencesBatch]]
        if raw:
            sentences = (
                encoded
                for line in in_stream
                if line and not line.isspace()
                for encoded in [
                    parser.encode_sentence(line.strip().split(), strict=strict)
                ]
                if encoded is not None
            )
            batches = (
                parser.batch_sentences(sentences)
                for sentences in itu.chunked_iter(
                    sentences, size=batch_size,
                )
            )
        else:
            test_set = DependencyDataset(
                DepGraph.read_conll(in_file),
                parser.lexer,
                parser.char_rnn,
                parser.ft_lexer,
                use_labels=parser.labels,
                use_tags=parser.tagset,
            )
            batches = (
                test_set.make_single_batch(sentences)
                for sentences in itu.chunked_iter(
                    test_set.treelist, size=parser.default_batch_size
                )
            )
        print("Parsing", file=sys.stderr)
        with smart_open(out_file, "w") as ostream:
            parser.batched_predict(
                batches,
                cast(IO[str], ostream),
                greedy=False,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Graph based Attention based dependency parser/tagger"
    )
    parser.add_argument(
        "config_file", metavar="CONFIG_FILE", type=str, help="the configuration file"
    )
    parser.add_argument(
        "--train_file", metavar="TRAIN_FILE", type=str, help="the conll training file"
    )
    parser.add_argument(
        "--dev_file", metavar="DEV_FILE", type=str, help="the conll development file"
    )
    parser.add_argument(
        "--pred_file", metavar="PRED_FILE", type=str, help="the conll file to parse"
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        help="the (torch) device to use for the parser. Supersedes configuration if given",
    )
    parser.add_argument(
        "--fasttext",
        metavar="PATH",
        help="The path to either an existing FastText model or a raw text file to train one. If this option is absent, a model will be trained from the parsing train set.",
    )
    parser.add_argument(
        "--out_dir",
        metavar="OUT_DIR",
        type=str,
        help="the path of the output directory (defaults to the config dir)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If a model already exists, restart training from scratch instead of continuing.",
    )
    parser.add_argument(
        "--rand_seed",
        metavar="SEED",
        type=int,
        help="Force the random seed fo Python and Pytorch (see <https://pytorch.org/docs/stable/notes/randomness.html> for notes on reproducibility)",
    )
    warnings.warn(
        "The `graph_parser` interface is DEPRECATED and will be removed in a future release, use `hopsparser train` instead",
        category=FutureWarning,
    )
    args = parser.parse_args(argv)

    if args.overwrite and not args.out_dir:
        print(
            "ERROR: overwriting is only supported with --out_dir",
            file=sys.stderr,
        )
        return 1

    if args.device is not None:
        overrides = {"device": args.device}
    else:
        overrides = dict()

    # TODO: warn about unused parameters in config
    if args.train_file and args.out_dir:
        model_dir = pathlib.Path(args.out_dir) / "model"
        config_file = pathlib.Path(args.config_file)
        trained_config_file = model_dir / "config.yaml"
    else:
        model_dir = pathlib.Path(args.config_file).parent
        # We need to give the temp file a name to avoid garbage collection before the method exits
        # this is not very clean but this code path will be removed soon anyway.
        _temp_config_file = tempfile.NamedTemporaryFile()
        shutil.copy(args.config_file, _temp_config_file.name)
        config_file = pathlib.Path(_temp_config_file.name)
        trained_config_file = pathlib.Path(args.config_file)

    if args.train_file and args.dev_file:
        print("Start training", file=sys.stderr)
        train(
            config_file=pathlib.Path(config_file),
            dev_file=pathlib.Path(args.dev_file),
            train_file=pathlib.Path(args.train_file),
            fasttext=(
                pathlib.Path(args.fasttext) if args.fasttext is not None else None
            ),
            max_tree_length=150,
            model_path=model_dir,
            overrides=overrides,
            overwrite=args.overwrite,
            rand_seed=args.rand_seed,
        )
        print("Training done.", file=sys.stderr)
        print("Parsing dev corpus", file=sys.stderr)
        if args.out_dir is not None:
            parsed_devset_path = os.path.join(
                args.out_dir, f"{os.path.basename(args.dev_file)}.parsed"
            )
        else:
            parsed_devset_path = os.path.join(
                os.path.dirname(args.dev_file),
                f"{os.path.basename(args.dev_file)}.parsed",
            )
        parse(model_dir, args.dev_file, parsed_devset_path, overrides=overrides)
        gold_devset = evaluator.load_conllu_file(args.dev_file)
        syst_devset = evaluator.load_conllu_file(parsed_devset_path)
        dev_metrics = evaluator.evaluate(gold_devset, syst_devset)
        print(
            f"Dev-best results: {dev_metrics['UPOS'].f1} UPOS\t{dev_metrics['UAS'].f1} UAS\t{dev_metrics['LAS'].f1} LAS",
            file=sys.stderr,
        )

    if args.pred_file:
        # TEST MODE
        if args.out_dir is not None:
            parsed_testset_path = os.path.join(
                args.out_dir, f"{os.path.basename(args.pred_file)}.parsed"
            )
        else:
            parsed_testset_path = os.path.join(
                os.path.dirname(args.pred_file),
                f"{os.path.basename(args.pred_file)}.parsed",
            )
        print("Parsing test corpus", file=sys.stderr)
        parse(
            trained_config_file,
            args.pred_file,
            parsed_testset_path,
            overrides=overrides,
        )
        print("Parsing done.", file=sys.stderr)


if __name__ == "__main__":
    main()
