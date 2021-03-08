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
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
import transformers
import yaml
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        xwords: Union[torch.Tensor, BertLexerBatch],
        xchars: torch.Tensor,
        xft: Iterable[torch.Tensor],
        sent_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Computes char embeddings
        char_embed = self.char_rnn(xchars)
        # Computes fasttext embeddings
        ft_embed = torch.stack([self.ft_lexer(sent) for sent in xft], dim=0)
        # Computes word embeddings
        lex_emb = self.lexer(xwords)

        # Encodes input for tagging and parsing
        xinput = torch.cat((lex_emb, char_embed, ft_embed), dim=2)
        packed_xinput = pack_padded_sequence(
            xinput, sent_lengths, batch_first=True, enforce_sorted=False
        )
        packed_dep_embeddings, _ = self.dep_rnn(packed_xinput)
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

        # NOTE: the accuracy scoring is approximative and cannot be interpreted as an UAS/LAS score
        # NOTE: fun project: track the correlation between them

        self.eval()

        dev_batches = dev_set.make_batches(
            batch_size, shuffle_batches=False, shuffle_data=False, order_by_length=True
        )
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
        dev_set: DependencyDataset,
        epochs: int,
        lr: float,
        lr_schedule: LRSchedule,
        model_path: Union[str, pathlib.Path],
        batch_size: Optional[int] = None,
    ):
        model_path = pathlib.Path(model_path)
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
                order_by_length=False,
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

            dev_loss, dev_tag_acc, dev_arc_acc, dev_lab_acc = self.eval_model(
                dev_set, batch_size=batch_size
            )
            print(
                f"Epoch {e} train mean loss {train_loss / overall_size}"
                f" valid mean loss {dev_loss} valid tag acc {dev_tag_acc} valid arc acc {dev_arc_acc} valid label acc {dev_lab_acc}"
                f" Base LR {scheduler.get_last_lr()[0]}"
            )

            if dev_arc_acc > best_arc_acc:
                self.save_params(model_path / "model.pt")
                best_arc_acc = dev_arc_acc

        self.load_params(model_path / "model.pt")
        self.save_params(model_path / "model.pt")

    def predict_batch(
        self,
        test_set: DependencyDataset,
        ostream: IO[str],
        batch_size: Optional[int] = None,
        greedy: bool = False,
    ):
        if batch_size is None:
            batch_size = self.default_batch_size
        self.eval()
        # keep natural order here
        test_batches = test_set.make_batches(
            batch_size, shuffle_batches=False, shuffle_data=False, order_by_length=False
        )

        out_trees: List[DepGraph] = []

        with torch.no_grad():
            for batch in test_batches:
                batch = batch.to(self.device)

                # batch prediction
                tagger_scores_batch, arc_scores_batch, lab_scores_batch = self(
                    batch.encoded_words,
                    batch.chars,
                    batch.subwords,
                    batch.sent_lengths,
                )
                arc_scores_batch = arc_scores_batch.cpu()

                for (tree, length, tagger_scores, arc_scores, lab_scores) in zip(
                    batch.trees,
                    batch.sent_lengths,
                    tagger_scores_batch,
                    arc_scores_batch,
                    lab_scores_batch,
                ):
                    # Predict heads
                    probs = arc_scores.numpy().T
                    batch_width, _ = probs.shape
                    mst_heads = (
                        np.argmax(probs[:length, :length], axis=1)
                        if greedy
                        else chuliu_edmonds(probs[:length, :length])
                    )
                    mst_heads = torch.from_numpy(
                        np.pad(mst_heads, (0, batch_width - length))
                    ).to(self.device)

                    # Predict tags
                    tag_idxes = tagger_scores.argmax(dim=1)
                    pos_tags = [test_set.itotag[idx] for idx in tag_idxes]
                    # Predict labels
                    select = mst_heads.unsqueeze(0).expand(lab_scores.size(0), -1)
                    selected = torch.gather(lab_scores, 1, select.unsqueeze(1)).squeeze(
                        1
                    )
                    mst_labels = selected.argmax(dim=0)
                    edges = [
                        deptree.Edge(head.item(), test_set.itolab[lbl], dep)
                        for (dep, lbl, head) in zip(
                            list(range(length)), mst_labels, mst_heads
                        )
                    ]
                    out_trees.append(
                        tree.replace(
                            edges=edges[1:],
                            pos_tags=pos_tags[1:],
                        )
                    )

        for tree in out_trees:
            print(str(tree), file=ostream, end="\n\n")

    @classmethod
    def initialize(
        cls,
        config_path: pathlib.Path,
        model_path: pathlib.Path,
        overrides: Dict[str, Any],
        treebank: List[DepGraph],
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
        cls, model_path: Union[str, pathlib.Path], overrides: Dict[str, Any]
    ) -> "BiAffineParser":
        # TODO: move the initialization code to initialize (even if that duplicates code?)
        model_path = pathlib.Path(model_path)
        if model_path.is_dir():
            config_dir = model_path
            model_path = model_path / "config.yaml"
            if not model_path.exists():
                raise ValueError(f"No config in {model_path.parent}")
        else:
            warnings.warn(
                "Loading a model from a YAML file is deprecated and will be removed in a future version."
            )
            config_dir = model_path.parent
        print(f"Initializing a parser from {model_path}")

        with open(model_path) as in_stream:
            hp = yaml.load(in_stream, Loader=yaml.SafeLoader)
        hp.update(overrides)
        hp.setdefault("device", "cpu")

        # FIXME: put that in the word lexer class
        ordered_vocab = loadlist(config_dir / "vocab.lst")

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
            bert_config_path = config_dir / "bert_config"
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
                lexer.bert_tokenizer.save_pretrained(bert_config_path)
                # Saving local paths in config is of little use and leaks information
                if os.path.exists(hp["lexer"]):
                    hp["lexer"] = "."

        chars_lexer = CharRNNLexer(
            charset=loadlist(config_dir / "charcodes.lst"),
            special_tokens=[DepGraph.ROOT_TOKEN],
            char_embedding_size=hp["char_embedding_size"],
            embedding_size=hp["charlstm_output_size"],
        )

        ft_lexer = FastTextLexer.load(
            str(config_dir / "fasttext_model.bin"), special_tokens=[DepGraph.ROOT_TOKEN]
        )

        itolab = loadlist(config_dir / "labcodes.lst")
        itotag = loadlist(config_dir / "tagcodes.lst")
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
        weights_file = config_dir / "model.pt"
        if weights_file.exists():
            parser.load_params(str(weights_file))
        else:
            parser.save_params(str(weights_file))
            # We were actually initializing — rather than loading — the model, let's save the
            # config with our changes
            with open(model_path, "w") as out_stream:
                yaml.dump(hp, out_stream)

        if hp.get("freeze_fasttext", False):
            freeze_module(ft_lexer)
        if hp.get("freeze_bert", False):
            try:
                freeze_module(lexer.bert)
            except AttributeError:
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


def parse(
    model_path: Union[str, pathlib.Path],
    in_file: Union[str, pathlib.Path, IO[str]],
    out_file: Union[str, pathlib.Path, IO[str]],
    overrides: Optional[Dict[str, str]] = None,
):
    if overrides is None:
        overrides = dict()
    parser = BiAffineParser.load(model_path, overrides)
    testtrees = DependencyDataset.read_conll(in_file)
    testset = DependencyDataset(
        testtrees,
        parser.lexer,
        parser.char_rnn,
        parser.ft_lexer,
        use_labels=parser.labels,
        use_tags=parser.tagset,
    )
    with smart_open(out_file, "w") as ostream:
        parser.predict_batch(testset, cast(IO[str], ostream), greedy=False)


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

    args = parser.parse_args(argv)
    if args.rand_seed is not None:
        random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)

    if args.device is not None:
        overrides = {"device": args.device}
    else:
        overrides = dict()

    # TODO: warn about unused parameters in config
    if args.train_file and args.out_dir:
        model_dir = pathlib.Path(args.out_dir) / "model"
        config_file = pathlib.Path(args.config_file)
    else:
        model_dir = pathlib.Path(args.config_file).parent
        # We need to give the temp file a name to avoid garbage collection before the method exits
        # this is not very clean but this code path will be deprecated soon anyway.
        _temp_config_file = tempfile.NamedTemporaryFile()
        shutil.copy(args.config_file, _temp_config_file.name)
        config_file = pathlib.Path(_temp_config_file.name)

    with open(config_file) as in_stream:
        hp = yaml.load(in_stream, Loader=yaml.SafeLoader)
    if "device" in hp:
        warnings.warn(
            "Setting a device directly in a configuration file is deprecated and will be removed in a future version. Use --device instead."
        )

    if args.train_file and args.dev_file:
        # TRAIN MODE
        traintrees = DependencyDataset.read_conll(args.train_file, max_tree_length=150)
        devtrees = DependencyDataset.read_conll(args.dev_file)
        if os.path.exists(model_dir) and not args.overwrite:
            print(f"Continuing training from {model_dir}", file=sys.stderr)
            parser = BiAffineParser.load(model_dir, overrides)
        else:
            if args.overwrite:
                print(
                    f"Erasing existing trained model in {model_dir} since --overwrite was asked",
                    file=sys.stderr,
                )
                shutil.rmtree(model_dir)
            parser = BiAffineParser.initialize(
                config_path=config_file,
                model_path=model_dir,
                overrides=overrides,
                treebank=traintrees,
                fasttext=args.fasttext,
            )

        trainset = DependencyDataset(
            traintrees,
            parser.lexer,
            parser.char_rnn,
            parser.ft_lexer,
            use_labels=parser.labels,
            use_tags=parser.tagset,
        )
        devset = DependencyDataset(
            devtrees,
            parser.lexer,
            parser.char_rnn,
            parser.ft_lexer,
            use_labels=parser.labels,
            use_tags=parser.tagset,
        )

        parser.train_model(
            train_set=trainset,
            dev_set=devset,
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            lr=hp["lr"],
            lr_schedule=hp.get(
                "lr_schedule", {"shape": "exponential", "warmup_steps": 0}
            ),
            model_path=model_dir,
        )
        print("training done.", file=sys.stderr)
        # Load final params
        parser.load_params(model_dir / "model.pt")
        parser.eval()
        if args.out_dir is not None:
            parsed_devset_path = os.path.join(
                args.out_dir, f"{os.path.basename(args.dev_file)}.parsed"
            )
        else:
            parsed_devset_path = os.path.join(
                os.path.dirname(args.dev_file),
                f"{os.path.basename(args.dev_file)}.parsed",
            )
        with open(parsed_devset_path, "w") as ostream:
            parser.predict_batch(devset, ostream, greedy=False)
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
        parse(model_dir, args.pred_file, parsed_testset_path, overrides=overrides)
        print("parsing done.", file=sys.stderr)


if __name__ == "__main__":
    main()
