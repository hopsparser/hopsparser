import math
import pathlib
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Sequence,
    TextIO,
    Tuple,
    Union,
)
import warnings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import transformers
import yaml
import argparse

import shutil

import os.path
import numpy as np

import torch
from torch import nn
from npdependency import deptree

from npdependency.mst import chuliu_edmonds_one_root as chuliu_edmonds

from npdependency.lexers import (
    BertBaseLexer,
    BertLexerBatch,
    CharDataSet,
    CharRNN,
    DefaultLexer,
    FastTextDataSet,
    FastTextTorch,
    freeze_module,
    make_vocab,
)
from npdependency.deptree import (
    DependencyBatch,
    DependencyDataset,
    DepGraph,
    gen_labels,
    gen_tags,
)
from npdependency import conll2018_eval as evaluator

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
        lexer: Union[DefaultLexer, BertBaseLexer],
        charset: CharDataSet,
        char_rnn: CharRNN,
        ft_lexer: FastTextTorch,
        tagset: Sequence[str],
        encoder_dropout: float,  # lstm dropout
        mlp_input: int,
        mlp_tag_hidden: int,
        mlp_arc_hidden: int,
        mlp_lab_hidden: int,
        mlp_dropout: float,
        labels: Sequence[str],
        biased_biaffine: bool,
        device: Union[str, torch.device],
    ):

        super(BiAffineParser, self).__init__()
        self.device = torch.device(device)
        self.lexer = lexer.to(self.device)
        self.dep_rnn = nn.LSTM(
            self.lexer.embedding_size
            + char_rnn.embedding_size
            + ft_lexer.embedding_size,
            mlp_input,
            3,
            batch_first=True,
            dropout=encoder_dropout,
            bidirectional=True,
        ).to(self.device)

        self.tagset = tagset
        self.labels = labels
        # POS tagger & char RNN
        self.pos_tagger = MLP(mlp_input * 2, mlp_tag_hidden, len(self.tagset)).to(
            self.device
        )
        self.charset = charset
        self.char_rnn = char_rnn.to(self.device)
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

        # hyperparams for saving...
        self.mlp_input, self.mlp_arc_hidden, self.mlp_lab_hidden = (
            mlp_input,
            mlp_arc_hidden,
            mlp_lab_hidden,
        )

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def load_params(self, path: str):
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
        xchars: Iterable[torch.Tensor],
        xft: Iterable[torch.Tensor],
        sent_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Computes char embeddings
        char_embed = torch.stack([self.char_rnn(column) for column in xchars], dim=1)
        # Computes fasttext embeddings
        ft_embed = torch.stack([self.ft_lexer(column) for column in xft], dim=1)
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
        # the padding tokens (even if they will be ignore in the crossentropy since the true
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

    def eval_model(self, dev_set: DependencyDataset, batch_size: int):

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
                overall_size += batch.sent_lengths.sum().item()

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
        batch_size: int,
        lr: float,
        lr_schedule: LRSchedule,
        modelpath="test_model.pt",
    ):

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
                overall_size += batch.sent_lengths.sum().item()

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
                dev_set, batch_size
            )
            print(
                f"Epoch {e} train mean loss {train_loss / overall_size}"
                f" valid mean loss {dev_loss} valid tag acc {dev_tag_acc} valid arc acc {dev_arc_acc} valid label acc {dev_lab_acc}"
                f" Base LR {scheduler.get_last_lr()[0]}"
            )

            if dev_arc_acc > best_arc_acc:
                self.save_params(modelpath)
                best_arc_acc = dev_arc_acc

        self.load_params(modelpath)
        self.save_params(modelpath)

    def predict_batch(
        self,
        test_set: DependencyDataset,
        ostream: TextIO,
        batch_size: int,
        greedy: bool = False,
    ):
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
                        deptree.Edge(head, test_set.itolab[lbl], dep)
                        for (dep, lbl, head) in zip(
                            list(range(length)), mst_labels, mst_heads
                        )
                    ]
                    out_trees.append(
                        DepGraph(
                            edges[1:],
                            wordlist=tree.words[1:],
                            pos_tags=pos_tags[1:],
                            mwe_ranges=tree.mwe_ranges,
                            metadata=tree.metadata,
                        )
                    )

        for tree in out_trees:
            print(str(tree), file=ostream, end="\n\n")

    @classmethod
    def from_config(
        cls, config_path: Union[str, pathlib.Path], overrides: Dict[str, Any]
    ) -> "BiAffineParser":
        config_path = pathlib.Path(config_path)
        with open(config_path) as in_stream:
            hp = yaml.load(in_stream, Loader=yaml.SafeLoader)
        hp.update(overrides)
        hp.setdefault("device", "cpu")

        config_dir = config_path.parent
        ordered_vocab = loadlist(config_dir / "vocab.lst")

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
            lexer = BertBaseLexer(
                itos=ordered_vocab,
                embedding_size=hp["word_embedding_size"],
                word_dropout=hp["word_dropout"],
                bert_modelfile=hp["lexer"],
                bert_layers=hp.get("bert_layers", [4, 5, 6, 7]),
                bert_subwords_reduction=hp.get("bert_subwords_reduction", "first"),
                bert_weighted=hp.get("bert_weighted", False),
                words_padding_idx=DependencyDataset.PAD_IDX,
                unk_word=DependencyDataset.UNK_WORD,
            )

        # char rnn processor
        ordered_charset = CharDataSet(
            loadlist(config_dir / "charcodes.lst"),
            special_tokens=[DepGraph.ROOT_TOKEN],
        )
        char_rnn = CharRNN(
            len(ordered_charset), hp["char_embedding_size"], hp["charlstm_output_size"]
        )

        # fasttext lexer
        ft_lexer = FastTextTorch.loadmodel(str(config_dir / "fasttext_model.bin"))

        itolab = loadlist(config_dir / "labcodes.lst")
        itotag = loadlist(config_dir / "tagcodes.lst")
        parser = cls(
            lexer=lexer,
            charset=ordered_charset,
            char_rnn=char_rnn,
            ft_lexer=ft_lexer,
            tagset=itotag,
            labels=itolab,
            encoder_dropout=hp["encoder_dropout"],
            mlp_input=hp["mlp_input"],
            mlp_tag_hidden=hp["mlp_tag_hidden"],
            mlp_arc_hidden=hp["mlp_arc_hidden"],
            mlp_lab_hidden=hp["mlp_lab_hidden"],
            mlp_dropout=hp["mlp_dropout"],
            biased_biaffine=hp.get("biased_biaffine", True),
            device=hp["device"],
        )
        weights_file = config_dir / "model.pt"
        if weights_file.exists():
            parser.load_params(str(weights_file))
        else:
            parser.save_params(str(weights_file))

        if hp.get("freeze_fasttext", False):
            freeze_module(ft_lexer)
        if hp.get("freeze_bert", False):
            try:
                freeze_module(lexer.bert)
            except AttributeError:
                print(
                    "Warning: a non-BERT lexer has no BERT to freeze, ignoring `freeze_bert` hypereparameter"
                )
        return parser


class GridSearch:
    """ This generates all the possible experiments specified by a yaml config file """

    def __init__(self, yamlparams):

        self.HP = yamlparams

    def generate_setup(self):

        setuplist = []  # init
        K = list(self.HP.keys())
        for key in K:
            value = self.HP[key]
            if type(value) is list:
                if setuplist:
                    setuplist = [elt + [V] for elt in setuplist for V in value]
                else:
                    setuplist = [[V] for V in value]
            else:
                for elt in setuplist:
                    elt.append(value)
        print(f"#{len(setuplist)} runs to be performed")

        for setup in setuplist:
            yield dict(zip(K, setup))

    @staticmethod
    def generate_run_name(base_filename, dict_setup):
        return (
            base_filename
            + "+"
            + "+".join(
                [
                    k + ":" + str(v)
                    for (k, v) in dict_setup.items()
                    if k != "output_path"
                ]
            )
            + ".conll"
        )


def savelist(strlist, filename):
    with open(filename, "w") as ostream:
        ostream.write("\n".join(strlist))


def loadlist(filename):
    with open(filename) as istream:
        strlist = [line.strip() for line in istream]
    return strlist


def main():
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
        "--out_dir",
        metavar="OUT_DIR",
        type=str,
        help="the path of the output directory (defaults to the config dir)",
    )
    parser.add_argument(
        "--fasttext",
        metavar="PATH",
        help="The path to either an existing FastText model or a raw text file to train one. If this option is absent, a model will be trained from the parsing train set.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        help="the (torch) device to use for the parser. Supersedes configuration if given",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If a model already exists, restart training from scratch instead of continuing.",
    )

    args = parser.parse_args()
    if args.device is not None:
        overrides = {"device": args.device}
    else:
        overrides = dict()

    # TODO: warn about unused parameters in config
    config_file = os.path.abspath(args.config_file)
    if args.train_file and args.out_dir:
        model_dir = os.path.join(args.out_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        config_file = shutil.copy(args.config_file, model_dir)
    else:
        model_dir = os.path.dirname(config_file)

    with open(config_file) as in_stream:
        hp = yaml.load(in_stream, Loader=yaml.SafeLoader)
    if "device" in hp:
        warnings.warn(
            "Setting a device directly in a configuration file is deprecated and will be removed in a future version. Use --device instead."
        )

    if args.train_file and args.dev_file:
        # TRAIN MODE
        weights_file = os.path.join(model_dir, "model.pt")
        if os.path.exists(weights_file):
            print(f"Found existing trained model in {model_dir}", file=sys.stderr)
            overwrite = args.overwrite
            if args.overwrite:
                print("Erasing it since --overwrite was asked", file=sys.stderr)
                # Ensure the parser won't load existing weights
                os.remove(weights_file)
                overwrite = True
            else:
                print("Continuing training", file=sys.stderr)
                overwrite = False
        else:
            overwrite = True
        traintrees = DependencyDataset.read_conll(args.train_file, max_tree_length=150)
        devtrees = DependencyDataset.read_conll(args.dev_file)

        if overwrite:
            fasttext_model_path = os.path.join(model_dir, "fasttext_model.bin")
            if args.fasttext is None:
                if os.path.exists(fasttext_model_path) and not args.out_dir:
                    print(f"Using the FastText model at {fasttext_model_path}")
                else:
                    if os.path.exists(fasttext_model_path):
                        print(
                            f"Erasing the FastText model at {fasttext_model_path} since --overwrite was asked",
                            file=sys.stderr,
                        )
                        os.remove(fasttext_model_path)
                    print(f"Generating a FastText model from {args.train_file}")
                    FastTextTorch.train_model_from_sents(
                        [tree.words[1:] for tree in traintrees], fasttext_model_path
                    )
            elif os.path.exists(args.fasttext):
                if os.path.exists(fasttext_model_path):
                    os.remove(fasttext_model_path)
                try:
                    # ugly, but we have no better way of checking if a file is a valid model
                    FastTextTorch.loadmodel(args.fasttext)
                    print(f"Using the FastText model at {args.fasttext}")
                    shutil.copy(args.fasttext, fasttext_model_path)
                except ValueError:
                    # FastText couldn't load it, so it should be raw text
                    print(f"Generating a FastText model from {args.fasttext}")
                    FastTextTorch.train_model_from_raw(
                        args.fasttext, fasttext_model_path
                    )
            else:
                raise ValueError(f"{args.fasttext} not found")

            # NOTE: these include the [ROOT] token, which will thus automatically have a dedicated
            # word embeddings in layers based on this vocab
            ordered_vocab = make_vocab(
                [word for tree in traintrees for word in tree.words],
                0,
                unk_word=DependencyDataset.UNK_WORD,
                pad_token=DependencyDataset.PAD_TOKEN,
            )
            savelist(ordered_vocab, os.path.join(model_dir, "vocab.lst"))

            # FIXME: A better save that can restore special tokens is probably a good idea
            ordered_charset = CharDataSet.from_words(
                ordered_vocab,
                special_tokens=[DepGraph.ROOT_TOKEN],
            )
            savelist(ordered_charset.i2c, os.path.join(model_dir, "charcodes.lst"))

            itolab = gen_labels(traintrees)
            savelist(itolab, os.path.join(model_dir, "labcodes.lst"))

            itotag = gen_tags(traintrees)
            savelist(itotag, os.path.join(model_dir, "tagcodes.lst"))

        parser = BiAffineParser.from_config(config_file, overrides)

        ft_dataset = FastTextDataSet(
            parser.ft_lexer, special_tokens=[DepGraph.ROOT_TOKEN]
        )
        trainset = DependencyDataset(
            traintrees,
            parser.lexer,
            parser.charset,
            ft_dataset,
            use_labels=parser.labels,
            use_tags=parser.tagset,
        )
        devset = DependencyDataset(
            devtrees,
            parser.lexer,
            parser.charset,
            ft_dataset,
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
            modelpath=weights_file,
        )
        print("training done.", file=sys.stderr)
        # Load final params
        parser.load_params(weights_file)
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
            parser.predict_batch(devset, ostream, hp["batch_size"], greedy=False)
        gold_devset = evaluator.load_conllu_file(args.dev_file)
        syst_devset = evaluator.load_conllu_file(parsed_devset_path)
        dev_metrics = evaluator.evaluate(gold_devset, syst_devset)
        print(
            f"Dev-best results: {dev_metrics['UPOS'].f1} UPOS\t{dev_metrics['UAS'].f1} UAS\t{dev_metrics['LAS'].f1} LAS",
            file=sys.stderr,
        )

    if args.pred_file:
        # TEST MODE
        parser = BiAffineParser.from_config(config_file, overrides)
        parser.eval()
        testtrees = DependencyDataset.read_conll(args.pred_file)
        # FIXME: the special tokens should be saved somewhere instead of hardcoded
        ft_dataset = FastTextDataSet(
            parser.ft_lexer, special_tokens=[DepGraph.ROOT_TOKEN]
        )
        testset = DependencyDataset(
            testtrees,
            parser.lexer,
            parser.charset,
            ft_dataset,
            use_labels=parser.labels,
            use_tags=parser.tagset,
        )
        if args.out_dir is not None:
            parsed_testset_path = os.path.join(
                args.out_dir, f"{os.path.basename(args.pred_file)}.parsed"
            )
        else:
            parsed_testset_path = os.path.join(
                os.path.dirname(args.pred_file),
                f"{os.path.basename(args.pred_file)}.parsed",
            )
        with open(parsed_testset_path, "w") as ostream:
            parser.predict_batch(testset, ostream, hp["batch_size"], greedy=False)
        print("parsing done.", file=sys.stderr)


if __name__ == "__main__":
    main()
