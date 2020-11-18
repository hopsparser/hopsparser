from typing import List, Sequence, Tuple
import torch
import fasttext
import os.path
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer
from collections import Counter
from random import random  # nosec:B311
from tempfile import gettempdir

from npdependency.deptree import DependencyDataset, DepGraph


def word_sampler(word_idx, unk_idx, dropout):
    return unk_idx if random() < dropout else word_idx


def make_vocab(treelist, threshold):
    """
    Extracts the set of tokens found in the data and orders it
    """
    vocab = Counter([word for tree in treelist for word in tree.words])
    vocab = set([tok for (tok, counts) in vocab.most_common() if counts > threshold])
    vocab.update([DependencyDataset.UNK_WORD])

    itos = [DependencyDataset.PAD_TOKEN] + list(vocab)
    return itos


class CharDataSet:
    """
    Namespace for simulating a char dataset
    """

    def __init__(self, charlist):
        self.i2c = charlist
        self.c2idx = dict([(c, idx) for idx, c in enumerate(charlist)])

    def __len__(self):
        return len(self.i2c)

    def word2charcodes(self, token):
        """
        Turns a string into a list of char codes.
        If the string is <pad> or <unk> returns an empty list of char codes
        """
        return (
            []
            if token in [DependencyDataset.PAD_TOKEN, DependencyDataset.UNK_WORD]
            else [self.c2idx[c] for c in token if c in self.c2idx]
        )

    def batchedtokens2codes(self, toklist):
        """
        Codes a list of tokens as a batch of lists of charcodes and pads them if needed
        :param toklist:
        :return:
        """
        charcodes = [self.word2charcodes(token) for token in toklist]
        sent_lengths = list(map(len, charcodes))
        max_len = max(sent_lengths)
        padded_codes = []
        for k, seq in zip(sent_lengths, charcodes):
            padded = seq + (max_len - k) * [DependencyDataset.PAD_IDX]
            padded_codes.append(padded)
        return torch.tensor(padded_codes)

    def batch_chars(self, sent_batch):
        """
        Batches a list of sentences such that each sentence is padded with the same word length.
        :yields: the character encodings for each word position in this batch of sentences
                 (yields the columns of the batch)
        """
        sent_lengths = list(map(len, sent_batch))
        max_sent_len = max(sent_lengths)

        batched_sents = []
        for k, seq in zip(sent_lengths, sent_batch):
            padded = seq + (max_sent_len - k) * [DependencyDataset.PAD_TOKEN]
            batched_sents.append(padded)

        for idx in range(max_sent_len):
            yield self.batchedtokens2codes(
                [sentence[idx] for sentence in batched_sents]
            )

    @staticmethod
    def make_vocab(wordlist):
        charset = set()
        for token in wordlist:
            charset.update(list(token))

        return CharDataSet([DependencyDataset.PAD_TOKEN] + list(charset))


class CharRNN(nn.Module):
    def __init__(self, charset_size, char_embedding_size, embedding_size):

        super(CharRNN, self).__init__()

        self.embedding_size = embedding_size
        self.char_embedding = nn.Embedding(
            charset_size, char_embedding_size, padding_idx=DependencyDataset.PAD_IDX
        )
        self.char_bilstm = nn.LSTM(
            char_embedding_size,
            int(self.embedding_size / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, xinput):
        """
        Predicts the word embedding from the token characters.
        :param xinput: is a tensor of char indexes encoding a batch of tokens [batch,token_len,char_seq_idx]
        :return: a word embedding tensor
        """
        embeddings = self.char_embedding(xinput)
        outputs, (_, cembedding) = self.char_bilstm(embeddings)
        result = cembedding.view(-1, self.embedding_size)
        return result


class FastTextDataSet:
    """
    Namespace for simulating a fasttext encoded dataset.
    By convention, the padding vector is the last element of the embedding matrix
    """

    def __init__(self, fasttextmodel):
        self.fasttextmodel = fasttextmodel
        self.PAD_IDX = self.fasttextmodel.vocab_size

    def word2subcodes(self, token):
        """
        Turns a string into a list of subword codes.
        """
        if token == DependencyDataset.PAD_TOKEN:
            return [self.PAD_IDX]
        else:
            return list(self.fasttextmodel.subwords_idxes(token))

    def batch_tokens(self, token_sequence):
        """
        Batches a list of tokens as a padded matrix of subword codes.
        :param token_sequence : a sequence of strings
        :return: a list of of list of codes (matrix with padding)
        """
        subcodes = [self.word2subcodes(token) for token in token_sequence]
        code_lengths = list(map(len, subcodes))
        max_len = max(code_lengths)
        padded_codes = []
        for k, seq in zip(code_lengths, subcodes):
            padded = seq + (max_len - k) * [self.PAD_IDX]
            padded_codes.append(padded)
        return torch.tensor(padded_codes)

    def batch_sentences(self, sent_batch):
        """
        Batches a list of sentences such that each sentence is padded with the same word length.
        :yields: the subword encodings for each word position in this batch of sentences
                 (yields the columns of the batch)
        """
        sent_lengths = list(map(len, sent_batch))
        max_sent_len = max(sent_lengths)
        batched_sents = []
        for k, seq in zip(sent_lengths, sent_batch):
            padded = seq + (max_sent_len - k) * [DependencyDataset.PAD_TOKEN]
            batched_sents.append(padded)

        for idx in range(max_sent_len):
            yield self.batch_tokens([sentence[idx] for sentence in batched_sents])


class FastTextTorch(nn.Module):
    """
    This is subword model using FastText as backend.
    It follows the same interface as the CharRNN
    """

    def __init__(self, fasttextmodel):
        super(FastTextTorch, self).__init__()
        self.fasttextmodel = fasttextmodel
        weights = fasttextmodel.get_input_matrix()
        self.vocab_size, self.embedding_size = weights.shape
        weights = np.vstack((weights, np.zeros(self.embedding_size)))
        self.embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(weights), padding_idx=self.vocab_size
        ).to(torch.float)

    def subwords_idxes(self, token):
        """
        Returns a list of ft subwords indexes for the token
        :param tok_sequence:
        :return:
        """
        return self.fasttextmodel.get_subwords(token)[1]

    def forward(self, xinput):
        """
        :param xinput: a batch of subwords
        :return: the fasttext embeddings for this batch
        """
        return self.embeddings(xinput).sum(
            dim=1
        )  # maybe compute true means or sums, currently uses <pad> tokens into this mean...

    @classmethod
    def loadmodel(cls, modelfile: str) -> "FastTextTorch":
        return cls(fasttext.load_model(modelfile))

    @classmethod
    def train_model_from_trees(
        cls, source_trees: Sequence[DepGraph], target_file: str
    ) -> "FastTextTorch":
        if os.path.exists(target_file):
            raise ValueError(f"{target_file} already exists!")
        else:
            source_file = os.path.join(gettempdir(), "source.ft")
            with open(source_file, "w") as source_stream:
                print(
                    "\n".join(
                        [" ".join(tree.words[1:]) for tree in reversed(source_trees)]
                    ),
                    file=source_stream,
                )

            print("Training fasttext model...")
            model = fasttext.train_unsupervised(
                source_file, model="skipgram", neg=10, minCount=5, epoch=10
            )
            model.save_model(target_file)
        return cls(model)

    @classmethod
    def train_model_from_raw(
        cls, raw_text_path: str, target_file: str
    ) -> "FastTextTorch":
        if os.path.exists(target_file):
            raise ValueError(f"{target_file} already exists!")
        else:
            print("Training fasttext model...")
            model = fasttext.train_unsupervised(
                raw_text_path, model="skipgram", neg=10, minCount=5, epoch=10
            )
            model.save_model(target_file)
        return cls(model)


class DefaultLexer(nn.Module):
    """
    This is the basic lexer wrapping an embedding layer.
    """

    def __init__(self, itos, embedding_size, word_dropout):
        super(DefaultLexer, self).__init__()
        self.embedding = nn.Embedding(
            len(itos), embedding_size, padding_idx=DependencyDataset.PAD_IDX
        )
        self.embedding_size = embedding_size  # thats the interface property
        self.itos = itos
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.word_dropout = word_dropout
        self._dpout = 0.0

    def train(self, mode: bool = True) -> "DefaultLexer":
        if mode:
            self._dpout = self.word_dropout
        else:
            self._dpout = 0.0
        return super().train(mode)

    def forward(self, word_sequences):
        """
        Takes words sequences codes as integer sequences and returns the embeddings
        """
        return self.embedding(word_sequences)

    def tokenize(self, tok_sequence):
        """
        This maps word tokens to integer indexes.
        Args:
           tok_sequence: a sequence of strings
        Returns:
           a list of integers
        """
        word_idxes = [
            self.stoi.get(token, self.stoi[DependencyDataset.UNK_WORD])
            for token in tok_sequence
        ]
        if self._dpout > 0:
            word_idxes = [
                word_sampler(widx, self.stoi[DependencyDataset.UNK_WORD], self._dpout)
                for widx in word_idxes
            ]
        return word_idxes


def freeze_module(module, freezing: bool = True):
    """Make a `torch.nn.Module` either finetunable 🔥 or frozen ❄.

    **WARNINGS**

    - Freezing a module will put it in eval mode (since a frozen module can't be in training mode),
      but unfreezing it will not put it back in training mode (since an unfrozen module still has an
      eval mode that you might want to use), you have to do that yourself.
    - Manually setting the submodules of a frozen module to train is not disabled, but if you want
      to do that, writing a custom freezing function is probably a better idea.
    """

    # This will replace the module's train function when freezing
    def no_train(model, mode=True):
        return model

    if freezing:
        module.eval()
        module.train = no_train
        for p in module.parameters():
            p.requires_grad = False
    else:
        for p in module.parameters():
            p.requires_grad = True
        module.train = type(module).train


class BertBaseLexer(nn.Module):
    """
    This Lexer performs tokenization and embedding mapping with BERT
    style models. It concatenates a standard embedding with a BERT
    embedding.
    """

    def __init__(
        self,
        default_itos: Sequence[str],
        default_embedding_size: int,
        word_dropout: float,
        bert_modelfile: str,
        bert_layers: Sequence[int],
        bert_weighted: bool,
        cased: bool = False,
    ):

        super(BertBaseLexer, self).__init__()
        self.itos = default_itos
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        self.bert = AutoModel.from_pretrained(bert_modelfile, output_hidden_states=True)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            bert_modelfile, use_fast=True
        )

        self.BERT_PAD_IDX = self.bert_tokenizer.pad_token_id
        self.BERT_UNK_IDX = self.bert_tokenizer.unk_token_id
        self.embedding_size = default_embedding_size + self.bert.config.hidden_size

        self.embedding = nn.Embedding(
            len(self.itos),
            default_embedding_size,
            padding_idx=DependencyDataset.PAD_IDX,
        )

        # ! FIXME: this is still somewhat brittle, since the BERT models have not been trained with
        # ! the root token at sentence beginning. Maybe we could use the BOS token for that purpose
        # ! instead?
        self.bert_tokenizer.add_tokens([DepGraph.ROOT_TOKEN], special_tokens=True)
        self.bert.resize_token_embeddings(len(self.bert_tokenizer))

        self.word_dropout = word_dropout
        self._dpout = 0.0
        self.cased = cased

        self.bert_layers = bert_layers
        self.bert_weighted = bert_weighted
        self.layer_weights = nn.Parameter(
            torch.ones(len(bert_layers), dtype=torch.float),
            requires_grad=self.bert_weighted,
        )
        self.layers_gamma = nn.Parameter(
            torch.ones(1, dtype=torch.float),
            requires_grad=self.bert_weighted,
        )

    def train(self, mode: bool = True) -> "BertBaseLexer":
        if mode:
            self._dpout = self.word_dropout
        else:
            self._dpout = 0.0
        return super().train(mode)

    def forward(self, coupled_sequences):
        """
        Takes words sequences codes as integer sequences and returns
        the embeddings from the last (top) BERT layer.
        """
        word_idxes, bert_idxes = coupled_sequences
        bert_layers = self.bert(bert_idxes, return_dict=True).hidden_states
        # Shape: layers×batch×sequence×features
        selected_bert_layers = torch.stack(
            [bert_layers[i] for i in self.bert_layers], 0
        )
        if self.bert_weighted:
            # Torch has no equivalent to `np.average` so this is somewhat annoying
            # ! FIXME: recomputing the softmax for every batch is needed at train time but is wasting
            # ! time in eval
            # Shape: layers
            normal_weights = self.layer_weights.softmax(dim=0)
            # shape: batch×sequence×features
            bertE = self.layers_gamma * torch.einsum(
                "l,lbsf->bsf", normal_weights, selected_bert_layers
            )
        else:
            bertE = selected_bert_layers.mean(dim=0)
        wordE = self.embedding(word_idxes)
        return torch.cat((wordE, bertE), dim=2)

    def tokenize(self, tok_sequence: Sequence[str]) -> Tuple[List[int], List[int]]:
        """
        This maps word tokens to integer indexes.
        When a word decomposes as multiple BPE units, we keep only the first (!)
        Args:
           tok_sequence: a sequence of strings
        Returns:
           a list of integers
        """
        word_idxes = [
            self.stoi.get(token, self.stoi[DependencyDataset.UNK_WORD])
            for token in tok_sequence
        ]
        # ? COMBAK: lowercasing should be done by the loaded tokenizer or am I missing something
        # ? here? (2020-11-08)
        if self.cased:
            bert_tokens = [
                self.bert_tokenizer.tokenize(token) for token in tok_sequence
            ]
        else:
            bert_tokens = [
                self.bert_tokenizer.tokenize(token.lower()) for token in tok_sequence
            ]
        bert_idxes = [
            self.bert_tokenizer.convert_tokens_to_ids(token)[0] for token in bert_tokens
        ]

        if self._dpout:
            word_idxes = [
                word_sampler(widx, self.stoi[DependencyDataset.UNK_WORD], self._dpout)
                for widx in word_idxes
            ]

        # ensure that first index is <root> and not an <unk>
        word_idxes[0] = self.stoi[DependencyDataset.UNK_WORD]
        bert_idxes[0] = self.bert_tokenizer.convert_tokens_to_ids(DepGraph.ROOT_TOKEN)
        return (word_idxes, bert_idxes)
