import sys
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
    make_vocab,
)
from npdependency.deptree import DependencyDataset, DepGraph


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(MLP, self).__init__()
        self.Wdown = nn.Linear(input_size, hidden_size)
        self.Wup = nn.Linear(hidden_size, output_size)
        self.g = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        return self.Wup(self.dropout(self.g(self.Wdown(input))))


class BiAffine(nn.Module):
    """Biaffine attention layer."""

    def __init__(self, input_dim, output_dim):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(
            torch.FloatTensor(output_dim, input_dim, input_dim)
        )  # check init
        nn.init.xavier_uniform_(self.U)

    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)


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
        lexer,
        char_rnn,
        ft_lexer,
        tagset_size,
        encoder_dropout,  # lstm dropout
        mlp_input,
        mlp_tag_hidden,
        mlp_arc_hidden,
        mlp_lab_hidden,
        mlp_dropout,
        num_labels,
        device="cuda:1",
    ):

        super(BiAffineParser, self).__init__()
        self.device = torch.device(device) if type(device) == str else device
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

        # POS tagger & char RNN
        self.pos_tagger = MLP(mlp_input * 2, mlp_tag_hidden, tagset_size).to(
            self.device
        )
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
        self.arc_biaffine = BiAffine(mlp_input, 1).to(self.device)
        self.lab_biaffine = BiAffine(mlp_input, num_labels).to(self.device)

        # hyperparams for saving...
        self.tagset_size = tagset_size
        self.mlp_input, self.mlp_arc_hidden, self.mlp_lab_hidden = (
            mlp_input,
            mlp_arc_hidden,
            mlp_lab_hidden,
        )
        self.num_labels = num_labels

    def save_params(self, path):

        torch.save(self.state_dict(), path)

    def load_params(self, path):

        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, xwords, xchars, xft):
        """Computes char embeddings"""
        char_embed = torch.stack([self.char_rnn(column) for column in xchars], dim=1)
        """ Computes fasttext embeddings """
        ft_embed = torch.stack([self.ft_lexer(column) for column in xft], dim=1)
        """ Computes word embeddings """
        lex_emb = self.lexer(xwords)

        """ Encodes input for tagging and parsing """
        xinput = torch.cat((lex_emb, char_embed, ft_embed), dim=2)
        dep_embeddings, _ = self.dep_rnn(xinput)

        """ Tagging """
        tag_scores = self.pos_tagger(dep_embeddings)

        """Compute the score matrices for the arcs and labels."""
        arc_h = self.arc_mlp_h(dep_embeddings)
        arc_d = self.arc_mlp_d(dep_embeddings)
        lab_h = self.lab_mlp_h(dep_embeddings)
        lab_d = self.lab_mlp_d(dep_embeddings)

        arc_scores = self.arc_biaffine(arc_h, arc_d)
        lab_scores = self.lab_biaffine(lab_h, lab_d)

        return tag_scores, arc_scores, lab_scores

    def eval_model(self, dev_set, batch_size):

        loss_fnc = nn.CrossEntropyLoss(reduction="sum")

        # Note: the accurracy scoring is approximative and cannot be interpreted as an UAS/LAS score !

        self.eval()
        self.lexer.eval_mode()

        dev_batches = dev_set.make_batches(
            batch_size, shuffle_batches=True, shuffle_data=True, order_by_length=True
        )
        tag_acc, arc_acc, lab_acc, gloss, taggerZ, arcsZ = 0, 0, 0, 0, 0, 0
        overall_size = 0

        with torch.no_grad():
            for batch in dev_batches:
                words, mwe, chars, subwords, cats, deps, tags, heads, labels = batch
                if type(deps) == tuple:
                    depsA, depsB = deps
                    deps = (depsA.to(self.device), depsB.to(self.device))
                    overall_size += depsA.size(0) * depsA.size(
                        1
                    )  # bc no masking at training
                else:
                    deps = deps.to(self.device)
                    overall_size += deps.size(0) * deps.size(
                        1
                    )  # bc no masking at training
                heads, labels, tags = (
                    heads.to(self.device),
                    labels.to(self.device),
                    tags.to(self.device),
                )
                chars = [token.to(self.device) for token in chars]
                subwords = [token.to(self.device) for token in subwords]
                # preds
                tagger_scores, arc_scores, lab_scores = self.forward(
                    deps, chars, subwords
                )

                # get global loss
                # ARC LOSS
                arc_scoresL = arc_scores.transpose(
                    -1, -2
                )  # [batch, sent_len, sent_len]
                arc_scoresL = arc_scoresL.contiguous().view(
                    -1, arc_scoresL.size(-1)
                )  # [batch*sent_len, sent_len]
                arc_loss = loss_fnc(arc_scoresL, heads.view(-1))  # [batch*sent_len]

                # TAGGER_LOSS
                tagger_scoresB = tagger_scores.contiguous().view(
                    -1, tagger_scores.size(-1)
                )
                tagger_loss = loss_fnc(tagger_scoresB, tags.view(-1))

                # LABEL LOSS
                headsL = heads.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, sent_len]
                headsL = headsL.expand(
                    -1, lab_scores.size(1), -1, -1
                )  # [batch, n_labels, 1, sent_len]
                lab_scoresL = torch.gather(lab_scores, 2, headsL).squeeze(
                    2
                )  # [batch, n_labels, sent_len]
                lab_scoresL = lab_scoresL.transpose(
                    -1, -2
                )  # [batch, sent_len, n_labels]
                lab_scoresL = lab_scoresL.contiguous().view(
                    -1, lab_scoresL.size(-1)
                )  # [batch*sent_len, n_labels]
                labelsL = labels.view(-1)  # [batch*sent_len]
                lab_loss = loss_fnc(lab_scoresL, labelsL)

                loss = tagger_loss + arc_loss + lab_loss
                gloss += loss.item()

                # greedy arc accurracy (without parsing)
                _, pred = arc_scores.max(dim=-2)
                mask = (heads != DependencyDataset.PAD_IDX).float()
                arc_accurracy = torch.sum((pred == heads).float() * mask, dim=-1)
                arc_acc += torch.sum(arc_accurracy).item()

                # tagger accurracy
                _, tag_pred = tagger_scores.max(dim=2)
                mask = (tags != DependencyDataset.PAD_IDX).float()
                tag_accurracy = torch.sum((tag_pred == tags).float() * mask, dim=-1)
                tag_acc += torch.sum(tag_accurracy).item()
                taggerZ += torch.sum(mask).item()

                # greedy label accurracy (without parsing)
                _, pred = lab_scores.max(dim=1)
                pred = torch.gather(pred, 1, heads.unsqueeze(1)).squeeze(1)
                mask = (heads != DependencyDataset.PAD_IDX).float()
                lab_accurracy = torch.sum((pred == labels).float() * mask, dim=-1)
                lab_acc += torch.sum(lab_accurracy).item()
                arcsZ += torch.sum(mask).item()

        return gloss / overall_size, tag_acc / taggerZ, arc_acc / arcsZ, lab_acc / arcsZ

    def train_model(
        self, train_set, dev_set, epochs, batch_size, lr, modelpath="test_model.pt"
    ):

        print("start training", flush=True)
        loss_fnc = nn.CrossEntropyLoss(reduction="sum")

        optimizer = torch.optim.Adam(
            self.parameters(), betas=(0.9, 0.9), lr=lr, eps=1e-09
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        for e in range(epochs):
            TRAIN_LOSS = 0
            BEST_ARC_ACC = 0
            self.lexer.train_mode()
            train_batches = train_set.make_batches(
                batch_size,
                shuffle_batches=True,
                shuffle_data=True,
                order_by_length=True,
            )
            overall_size = 0
            for batch in train_batches:
                self.train()
                words, mwe, chars, subwords, cats, deps, tags, heads, labels = batch
                if type(deps) == tuple:
                    depsA, depsB = deps
                    deps = (depsA.to(self.device), depsB.to(self.device))
                    overall_size += depsA.size(0) * depsA.size(
                        1
                    )  # bc no masking at training
                else:
                    deps = deps.to(self.device)
                    overall_size += deps.size(0) * deps.size(
                        1
                    )  # bc no masking at training
                heads, labels, tags = (
                    heads.to(self.device),
                    labels.to(self.device),
                    tags.to(self.device),
                )
                chars = [token.to(self.device) for token in chars]
                subwords = [token.to(self.device) for token in subwords]

                # FORWARD
                tagger_scores, arc_scores, lab_scores = self.forward(
                    deps, chars, subwords
                )

                # ARC LOSS
                arc_scores = arc_scores.transpose(-1, -2)  # [batch, sent_len, sent_len]
                arc_scores = arc_scores.contiguous().view(
                    -1, arc_scores.size(-1)
                )  # [batch*sent_len, sent_len]
                arc_loss = loss_fnc(arc_scores, heads.view(-1))  # [batch*sent_len]

                # TAGGER_LOSS
                tagger_scores = tagger_scores.contiguous().view(
                    -1, tagger_scores.size(-1)
                )
                tagger_loss = loss_fnc(tagger_scores, tags.view(-1))

                # LABEL LOSS
                heads = heads.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, sent_len]
                heads = heads.expand(
                    -1, lab_scores.size(1), -1, -1
                )  # [batch, n_labels, 1, sent_len]
                lab_scores = torch.gather(lab_scores, 2, heads).squeeze(
                    2
                )  # [batch, n_labels, sent_len]
                lab_scores = lab_scores.transpose(-1, -2)  # [batch, sent_len, n_labels]
                lab_scores = lab_scores.contiguous().view(
                    -1, lab_scores.size(-1)
                )  # [batch*sent_len, n_labels]
                labels = labels.view(-1)  # [batch*sent_len]
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

        self.lexer.eval_mode()
        test_batches = test_set.make_batches(
            batch_size, shuffle_batches=False, shuffle_data=False, order_by_length=False
        )  # keep natural order here

        with torch.no_grad():
            for batch in test_batches:
                self.eval()
                words, mwe, chars, subwords, cats, deps, tags, heads, labels = batch
                if type(deps) == tuple:
                    depsA, depsB = deps
                    deps = (depsA.to(self.device), depsB.to(self.device))
                    SLENGTHS = (depsA != DependencyDataset.PAD_IDX).long().sum(-1)
                else:
                    deps = deps.to(self.device)
                    SLENGTHS = (deps != DependencyDataset.PAD_IDX).long().sum(-1)
                heads, labels, tags = (
                    heads.to(self.device),
                    labels.to(self.device),
                    tags.to(self.device),
                )
                chars = [token.to(self.device) for token in chars]
                subwords = [token.to(self.device) for token in subwords]

                # batch prediction
                tagger_scores_batch, arc_scores_batch, lab_scores_batch = self.forward(
                    deps, chars, subwords
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
                    SLENGTHS,
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
                    mst_heads = np.pad(mst_heads, (0, batch_width - length.item()))

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
                    # edges          = [ (head,test_set.itolab[lbl],dep) for (dep,head,lbl) in zip(list(range(length)),mst_heads[:length], mst_labels[:length]) ]
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

    args = parser.parse_args()
    hp = yaml.load(open(args.config_file).read(), Loader=yaml.FullLoader)

    CONFIG_FILE = os.path.abspath(args.config_file)
    if args.out_dir:
        MODEL_DIR = os.path.join(args.out_dir, "model")
        os.makedirs(MODEL_DIR, exist_ok=True)
        shutil.copy(args.config_file, MODEL_DIR)
    else:
        MODEL_DIR = os.path.dirname(CONFIG_FILE)

    if args.train_file and args.dev_file:
        # TRAIN MODE
        traintrees = DependencyDataset.read_conll(args.train_file)
        devtrees = DependencyDataset.read_conll(args.dev_file)

        bert_modelfile = hp["lexer"].split("/")[-1]
        ordered_vocab = make_vocab(traintrees, 0)

        savelist(ordered_vocab, os.path.join(MODEL_DIR, bert_modelfile + "-vocab"))

        if hp["lexer"] == "default":
            lexer = DefaultLexer(
                ordered_vocab, hp["word_embedding_size"], hp["word_dropout"]
            )
        else:
            if "cased" in hp:
                cased = True
            else:
                cased = "uncased" not in bert_modelfile
            lexer = BertBaseLexer(
                ordered_vocab,
                hp["word_embedding_size"],
                hp["word_dropout"],
                cased=cased,
                bert_modelfile=hp["lexer"],
            )

        # char rnn lexer
        ordered_charset = CharDataSet.make_vocab(ordered_vocab)
        savelist(
            ordered_charset.i2c, os.path.join(MODEL_DIR, bert_modelfile + "-charcodes")
        )
        char_rnn = CharRNN(
            len(ordered_charset), hp["char_embedding_size"], hp["charlstm_output_size"]
        )

        # fasttext lexer
        ft_lexer = FastTextTorch.train_model(
            traintrees, os.path.join(MODEL_DIR, "fasttext_model.bin")
        )
        ft_dataset = FastTextDataSet(ft_lexer)

        trainset = DependencyDataset(traintrees, lexer, ordered_charset, ft_dataset)
        itolab, itotag = trainset.itolab, trainset.itotag
        savelist(itolab, os.path.join(MODEL_DIR, bert_modelfile + "-labcodes"))
        savelist(itotag, os.path.join(MODEL_DIR, bert_modelfile + "-tagcodes"))
        devset = DependencyDataset(
            devtrees,
            lexer,
            ordered_charset,
            ft_dataset,
            use_labels=itolab,
            use_tags=itotag,
        )

        parser = BiAffineParser(
            lexer,
            char_rnn,
            ft_lexer,
            len(itotag),
            hp["encoder_dropout"],
            hp["mlp_input"],
            hp["mlp_tag_hidden"],
            hp["mlp_arc_hidden"],
            hp["mlp_lab_hidden"],
            hp["mlp_dropout"],
            len(itolab),
            hp["device"],
        )
        parser.train_model(
            trainset,
            devset,
            hp["epochs"],
            hp["batch_size"],
            hp["lr"],
            modelpath=os.path.join(MODEL_DIR, bert_modelfile + "-model.pt"),
        )
        print("training done.", file=sys.stderr)

    if args.pred_file:
        # TEST MODE
        testtrees = DependencyDataset.read_conll(args.pred_file)
        bert_modelfile = hp["lexer"].split("/")[-1]
        ordered_vocab = loadlist(os.path.join(MODEL_DIR, bert_modelfile + "-vocab"))

        if hp["lexer"] == "default":
            lexer = DefaultLexer(
                ordered_vocab, hp["word_embedding_size"], hp["word_dropout"]
            )
        else:
            if "cased" in hp:
                cased = True
            else:
                cased = "uncased" not in bert_modelfile
            lexer = BertBaseLexer(
                ordered_vocab,
                hp["word_embedding_size"],
                hp["word_dropout"],
                cased=cased,
                bert_modelfile=hp["lexer"],
            )

        # char rnn processor
        ordered_charset = CharDataSet(
            loadlist(os.path.join(MODEL_DIR, bert_modelfile + "-charcodes"))
        )
        char_rnn = CharRNN(
            len(ordered_charset), hp["char_embedding_size"], hp["charlstm_output_size"]
        )

        # fasttext lexer
        ft_lexer = FastTextTorch.loadmodel(
            os.path.join(MODEL_DIR, "fasttext_model.bin")
        )
        ft_dataset = FastTextDataSet(ft_lexer)

        itolab = loadlist(os.path.join(MODEL_DIR, bert_modelfile + "-labcodes"))
        itotag = loadlist(os.path.join(MODEL_DIR, bert_modelfile + "-tagcodes"))
        testset = DependencyDataset(
            testtrees,
            lexer,
            ordered_charset,
            ft_dataset,
            use_labels=itolab,
            use_tags=itotag,
        )
        parser = BiAffineParser(
            lexer,
            char_rnn,
            ft_lexer,
            len(itotag),
            hp["encoder_dropout"],
            hp["mlp_input"],
            hp["mlp_tag_hidden"],
            hp["mlp_arc_hidden"],
            hp["mlp_lab_hidden"],
            hp["mlp_dropout"],
            len(itolab),
            hp["device"],
        )
        parser.load_params(os.path.join(MODEL_DIR, bert_modelfile + "-model.pt"))
        parsed_testset_path = os.path.join(
            args.out_dir, f"{os.path.basename(args.pred_file)}.parsed"
        )
        with open(parsed_testset_path, "w") as ostream:
            parser.predict_batch(testset, ostream, hp["batch_size"], greedy=False)
        print("parsing done.", file=sys.stderr)


if __name__ == "__main__":
    main()
