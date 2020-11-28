import pathlib
import sys
from typing import Any, Dict, Sequence, Union
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import yaml
import argparse

import shutil

import os.path
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from npdependency.mst import chuliu_edmonds_one_root as chuliu_edmonds

from npdependency.lexers import (
    BertBaseLexer,
    CharDataSet,
    CharRNN,
    DefaultLexer,
    FastTextDataSet,
    FastTextTorch,
    freeze_module,
    make_vocab,
)
from npdependency.deptree import DependencyDataset, DepGraph, gen_labels, gen_tags
from npdependency import conll2018_eval as evaluator


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

    def forward(self, xwords, xchars, xft, sent_lengths: torch.Tensor) -> torch.Tensor:
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

    def eval_model(self, dev_set, batch_size):

        loss_fnc = nn.CrossEntropyLoss(reduction="sum")

        # Note: the accurracy scoring is approximative and cannot be interpreted as an UAS/LAS score
        # Note: fun project: tracke the correlation between them

        self.eval()

        dev_batches = dev_set.make_batches(
            batch_size, shuffle_batches=True, shuffle_data=True, order_by_length=True
        )
        tag_acc, arc_acc, lab_acc, gloss, accZ = 0, 0, 0, 0, 0
        overall_size = 0

        with torch.no_grad():
            for batch in dev_batches:
                (
                    words,
                    mwe,
                    chars,
                    subwords,
                    cats,
                    encoded_words,
                    tags,
                    heads,
                    labels,
                    sent_lengths,
                ) = batch
                encoded_words = encoded_words.to(self.device)
                # bc no masking at training
                overall_size += encoded_words.size(0) * encoded_words.size(1)
                heads, labels, tags = (
                    heads.to(self.device),
                    labels.to(self.device),
                    tags.to(self.device),
                )
                chars = [token.to(self.device) for token in chars]
                subwords = [token.to(self.device) for token in subwords]
                # preds
                tagger_scores, arc_scores, lab_scores = self(
                    encoded_words, chars, subwords, sent_lengths
                )

                # get global loss
                # ARC LOSS
                # [batch, sent_len, sent_len]
                arc_scoresL = arc_scores.transpose(-1, -2)
                # [batch*sent_len, sent_len]
                arc_scoresL = arc_scoresL.reshape(-1, arc_scoresL.size(-1))
                arc_loss = loss_fnc(arc_scoresL, heads.view(-1))  # [batch*sent_len]

                # TAGGER_LOSS
                tagger_scoresB = tagger_scores.reshape(-1, tagger_scores.size(-1))
                tagger_loss = loss_fnc(tagger_scoresB, tags.view(-1))

                # LABEL LOSS
                # [batch, 1, 1, sent_len]
                headsL = heads.unsqueeze(1).unsqueeze(2)
                # [batch, n_labels, 1, sent_len]
                headsL = headsL.expand(-1, lab_scores.size(1), -1, -1)
                # [batch, n_labels, sent_len]
                lab_scoresL = torch.gather(lab_scores, 2, headsL).squeeze(2)
                # [batch, sent_len, n_labels]
                lab_scoresL = lab_scoresL.transpose(-1, -2)
                # [batch*sent_len, n_labels]
                lab_scoresL = lab_scoresL.reshape(-1, lab_scoresL.size(-1))
                # [batch*sent_len]
                labelsL = labels.view(-1)
                lab_loss = loss_fnc(lab_scoresL, labelsL)

                loss = tagger_loss + arc_loss + lab_loss
                gloss += loss.item()

                mask = heads.ne(dev_set.PAD_IDX)
                accZ += mask.sum().item()
                # greedy arc accurracy (without parsing)
                arc_pred = arc_scores.argmax(dim=-2)
                arc_accurracy = arc_pred.eq(heads).logical_and(mask).sum()
                arc_acc += arc_accurracy.item()

                # tagger accurracy
                tag_pred = tagger_scores.argmax(dim=2)
                tag_accurracy = tag_pred.eq(tags).logical_and(mask).sum()
                tag_acc += tag_accurracy.item()

                # greedy label accurracy (without parsing)
                lab_pred = lab_scores.argmax(dim=1)
                lab_pred = torch.gather(lab_pred, 1, heads.unsqueeze(1)).squeeze(1)
                lab_accurracy = lab_pred.eq(labels).logical_and(mask).sum()
                lab_acc += lab_accurracy.item()

        return gloss / overall_size, tag_acc / accZ, arc_acc / accZ, lab_acc / accZ

    def train_model(
        self, train_set, dev_set, epochs, batch_size, lr, modelpath="test_model.pt"
    ):

        print(f"Start training on {self.device}", flush=True)
        loss_fnc = nn.CrossEntropyLoss(reduction="sum")

        optimizer = torch.optim.Adam(
            self.parameters(), betas=(0.9, 0.9), lr=lr, eps=1e-09
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        for e in range(epochs):
            TRAIN_LOSS = 0.0
            BEST_ARC_ACC = 0.0
            # FIXME: why order by length here?
            train_batches = train_set.make_batches(
                batch_size,
                shuffle_batches=True,
                shuffle_data=True,
                order_by_length=True,
            )
            overall_size = 0
            self.train()
            for batch in train_batches:
                (
                    words,
                    mwe,
                    chars,
                    subwords,
                    cats,
                    encoded_words,
                    tags,
                    heads,
                    labels,
                    sent_lengths,
                ) = batch
                encoded_words = encoded_words.to(self.device)
                # bc no masking at training
                overall_size += encoded_words.size(0) * encoded_words.size(1)
                heads, labels, tags = (
                    heads.to(self.device),
                    labels.to(self.device),
                    tags.to(self.device),
                )
                chars = [token.to(self.device) for token in chars]
                subwords = [token.to(self.device) for token in subwords]

                # FORWARD
                tagger_scores, arc_scores, lab_scores = self(
                    encoded_words, chars, subwords, sent_lengths
                )

                # ARC LOSS
                # [batch, sent_len, sent_len]
                arc_scores = arc_scores.transpose(-1, -2)
                # [batch*sent_len, sent_len]
                arc_scores = arc_scores.reshape(-1, arc_scores.size(-1))
                # [batch*sent_len]
                arc_loss = loss_fnc(arc_scores, heads.view(-1))

                # TAGGER_LOSS
                tagger_scores = tagger_scores.reshape(-1, tagger_scores.size(-1))
                tagger_loss = loss_fnc(tagger_scores, tags.view(-1))

                # LABEL LOSS
                # [batch, 1, 1, sent_len]
                heads = heads.unsqueeze(1).unsqueeze(2)
                # [batch, n_labels, 1, sent_len]
                heads = heads.expand(-1, lab_scores.size(1), -1, -1)
                # [batch, n_labels, sent_len]
                lab_scores = torch.gather(lab_scores, 2, heads).squeeze(2)
                # [batch, sent_len, n_labels]
                lab_scores = lab_scores.transpose(-1, -2)
                # [batch*sent_len, n_labels]
                lab_scores = lab_scores.reshape(-1, lab_scores.size(-1))
                # [batch*sent_len]
                labels = labels.view(-1)
                lab_loss = loss_fnc(lab_scores, labels)

                loss = tagger_loss + arc_loss + lab_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                TRAIN_LOSS += loss.item()

            DEV_LOSS, DEV_TAG_ACC, DEV_ARC_ACC, DEV_LAB_ACC = self.eval_model(
                dev_set, batch_size
            )
            print(
                "Epoch ",
                e,
                "train mean loss",
                TRAIN_LOSS / overall_size,
                "valid mean loss",
                DEV_LOSS,
                "valid tag acc",
                DEV_TAG_ACC,
                "valid arc acc",
                DEV_ARC_ACC,
                "valid label acc",
                DEV_LAB_ACC,
                "Base LR",
                scheduler.get_lr()[0],
                flush=True,
            )

            if DEV_ARC_ACC > BEST_ARC_ACC:
                self.save_params(modelpath)
                BEST_ARC_ACC = DEV_ARC_ACC

            scheduler.step()

        self.load_params(modelpath)
        self.save_params(modelpath)

    def predict_batch(self, test_set, ostream, batch_size, greedy=False):

        self.eval()
        test_batches = test_set.make_batches(
            batch_size, shuffle_batches=False, shuffle_data=False, order_by_length=False
        )  # keep natural order here

        with torch.no_grad():
            for batch in test_batches:
                (
                    words,
                    mwe,
                    chars,
                    subwords,
                    cats,
                    encoded_words,
                    tags,
                    heads,
                    labels,
                    sent_lengths,
                ) = batch
                encoded_words = encoded_words.to(self.device)
                sent_lengths = [len(s) for s in words]
                heads, labels, tags = (
                    heads.to(self.device),
                    labels.to(self.device),
                    tags.to(self.device),
                )
                chars = [token.to(self.device) for token in chars]
                subwords = [token.to(self.device) for token in subwords]

                # batch prediction
                tagger_scores_batch, arc_scores_batch, lab_scores_batch = self(
                    encoded_words,
                    chars,
                    subwords,
                    sent_lengths,
                )
                tagger_scores_batch, arc_scores_batch, lab_scores_batch = (
                    tagger_scores_batch.cpu(),
                    arc_scores_batch.cpu(),
                    lab_scores_batch.cpu(),
                )

                for (
                    tokens,
                    mwe_range,
                    length,
                    tagger_scores,
                    arc_scores,
                    lab_scores,
                ) in zip(
                    words,
                    mwe,
                    sent_lengths,
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
                    mst_heads = np.pad(mst_heads, (0, batch_width - length))

                    # Predict tags
                    tag_probs = tagger_scores.numpy()
                    tag_idxes = np.argmax(tag_probs, axis=1)
                    pos_tags = [test_set.itotag[idx] for idx in tag_idxes]
                    # Predict labels
                    select = (
                        torch.LongTensor(mst_heads)
                        .unsqueeze(0)
                        .expand(lab_scores.size(0), -1)
                    )
                    select = Variable(select)
                    selected = torch.gather(lab_scores, 1, select.unsqueeze(1)).squeeze(
                        1
                    )
                    _, mst_labels = selected.max(dim=0)
                    mst_labels = mst_labels.data.numpy()
                    # edges = [ (head,test_set.itolab[lbl],dep) for (dep,head,lbl) in zip(list(range(length)),mst_heads[:length], mst_labels[:length]) ]
                    edges = [
                        (head, test_set.itolab[lbl], dep)
                        for (dep, head, lbl) in zip(
                            list(range(length)), mst_heads, mst_labels
                        )
                    ]
                    dg = DepGraph(
                        edges[1:],
                        wordlist=tokens[1:],
                        pos_tags=pos_tags[1:],
                        mwe_range=mwe_range,
                    )
                    print(dg, file=ostream)
                    print(file=ostream)

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
        print("#%d" % (len(setuplist)), "runs to be performed")

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
        traintrees = DependencyDataset.read_conll(args.train_file)
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
            trainset,
            devset,
            hp["epochs"],
            hp["batch_size"],
            hp["lr"],
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
