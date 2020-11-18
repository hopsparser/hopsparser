from typing import Iterable, List, Optional, Sequence, Tuple, Union
import torch
import fasttext
import os.path
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from collections import Counter
from random import random  # nosec:B311
from tempfile import gettempdir

# Python 3.7 shim
try:
    from typing import Final
except ImportError:
    from typing_extensions import Final


def word_sampler(word_idx, unk_idx, dropout):
    return unk_idx if random() < dropout else word_idx


def make_vocab(
    words: Iterable[str], threshold: int, unk_word: str, pad_token: str
) -> List[str]:
    """
    Extracts the set of tokens found in the data and orders it
    """
    vocab_counter = Counter(words)
    vocab = set(
        [tok for (tok, counts) in vocab_counter.most_common() if counts > threshold]
    )
    vocab.add(unk_word)

    itos = [pad_token, *sorted(vocab)]
    return itos


class CharDataSet:
    """
    Namespace for simulating a char dataset.

    ## Padding and special tokens

    - Char index `0` is for padding
    - Char index `1` is for special tokens that should not be split into individual characters

    These are **not** configurable in order to allow a simpler handling of the internal vocabulary
    """

    PAD_IDX: Final[int] = 0
    SPECIAL_TOKENS_IDX: Final[int] = 1

    def __init__(
        self, charlist: Iterable[str], special_tokens: Optional[Iterable[str]] = None
    ):
        self.i2c = ["<pad>", "<special>", *charlist]
        self.c2idx = {c: idx for idx, c in enumerate(charlist)}
        self.special_tokens = set([] if special_tokens is None else special_tokens)

    def __len__(self):
        return len(self.i2c)

    def word2charcodes(self, token: str) -> List[int]:
        """
        Turns a string into a list of char codes.
        If the string is <pad> returns an empty list of char codes
        """
        if token in self.special_tokens:
            return [self.SPECIAL_TOKENS_IDX]
        return [self.c2idx[c] for c in token if c in self.c2idx]

    def batchedtokens2codes(self, toklist: List[str]) -> torch.Tensor:
        """
        Codes a list of tokens as a batch of lists of charcodes and pads them if needed
        :param toklist:
        :return:
        """
        charcodes = [
            torch.tensor(self.word2charcodes(token), dtype=torch.long)
            for token in toklist
        ]
        return pad_sequence(charcodes, padding_value=self.PAD_IDX, batch_first=True)

    def batch_chars(self, sent_batch: List[str]) -> List[torch.Tensor]:
        """
        Batches a list of sentences such that each sentence is padded with the same word length.
        :yields: the character encodings for each word position in this batch of sentences
                 (yields the columns of the batch)
        """
        sent_lengths = [len(s) for s in sent_batch]
        max_sent_len = max(sent_lengths)

        # We use empty strings for padding, so they will be correctly filled with the padding value
        # when we tensorize
        batched_sents = [["" for _ in range(max_sent_len)] for _ in sent_batch]
        for batch_sent, l, sent in zip(batched_sents, sent_lengths, sent_batch):
            batch_sent[:l] = sent

        for idx in range(max_sent_len):
            yield self.batchedtokens2codes(
                [sentence[idx] for sentence in batched_sents]
            )

    @classmethod
    def from_words(
        cls,
        wordlist: Iterable[str],
        special_tokens: Optional[Iterable[str]] = None,
    ) -> "CharDataSet":
        charset = set((c for word in wordlist for c in word))
        return cls(sorted(charset), special_tokens=special_tokens)


class CharRNN(nn.Module):
    def __init__(self, charset_size, char_embedding_size, embedding_size):

        super(CharRNN, self).__init__()

        self.embedding_size = embedding_size
        self.char_embedding = nn.Embedding(
            charset_size, char_embedding_size, padding_idx=CharDataSet.PAD_IDX
        )
        self.char_bilstm = nn.LSTM(
            char_embedding_size,
            int(self.embedding_size / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, xinput: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, fasttextmodel: "FastTextTorch"):
        self.fasttextmodel = fasttextmodel
        self.PAD_IDX = self.fasttextmodel.vocab_size

    def word2subcodes(self, token: str) -> torch.Tensor:
        """
        Turns a string into a list of subword codes.
        """
        if token == "":
            return torch.tensor([self.PAD_IDX], dtype=torch.long)
        return self.fasttextmodel.subwords_idxes(token)

    def batch_tokens(self, token_sequence):
        """
        Batches a list of tokens as a padded matrix of subword codes.
        :param token_sequence : a sequence of strings
        :return: a list of of list of codes (matrix with padding)
        """
        subcodes = [self.word2subcodes(token) for token in token_sequence]
        return pad_sequence(subcodes, padding_value=self.PAD_IDX, batch_first=True)

    def batch_sentences(self, sent_batch: List[str]) -> List[torch.Tensor]:
        """
        Batches a list of sentences such that each sentence is padded with the same word length.
        :yields: the subword encodings for each word position in this batch of sentences
                 (yields the columns of the batch)
        """
        sent_lengths = [len(sent) for sent in sent_batch]
        max_sent_len = max(sent_lengths)

        # The empty string here serves as padding, which, contrarily to CharsDataSet is a bit ugly,
        # since we intercept it instead of passing it to FastText
        batched_sents = [["" for _ in range(max_sent_len)] for _ in sent_batch]
        for batch_sent, l, sent in zip(batched_sents, sent_lengths, sent_batch):
            batch_sent[:l] = sent

        for idx in range(max_sent_len):
            yield self.batch_tokens([sentence[idx] for sentence in batched_sents])


class FastTextTorch(nn.Module):
    """
    This is subword model using FastText as backend.
    It follows the same interface as the CharRNN
    """

    def __init__(self, fasttextmodel: fasttext.FastText):
        super(FastTextTorch, self).__init__()
        self.fasttextmodel = fasttextmodel
        weights = fasttextmodel.get_input_matrix()
        self.vocab_size, self.embedding_size = weights.shape
        weights = np.vstack((weights, np.zeros(self.embedding_size)))
        self.embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(weights), padding_idx=self.vocab_size
        ).to(torch.float)

    def subwords_idxes(self, token: str) -> torch.Tensor:
        """
        Returns a list of ft subwords indexes for the token
        :param tok_sequence:
        :return:
        """
        return torch.from_numpy(self.fasttextmodel.get_subwords(token)[1])

    def forward(self, xinput: torch.Tensor) -> torch.Tensor:
        """
        :param xinput: a batch of subwords
        :return: the fasttext embeddings for this batch
        """
        # maybe compute true means or sums, currently uses <pad> tokens into this mean...
        return self.embeddings(xinput).sum(dim=1)

    @classmethod
    def loadmodel(cls, modelfile: str) -> "FastTextTorch":
        return cls(fasttext.load_model(modelfile))

    @classmethod
    def train_model_from_sents(
        cls, source_sents: Iterable[List[str]], target_file: str
    ) -> "FastTextTorch":
        if os.path.exists(target_file):
            raise ValueError(f"{target_file} already exists!")
        else:
            source_file = os.path.join(gettempdir(), "source.ft")
            with open(source_file, "w") as source_stream:
                print(
                    "\n".join([" ".join(sent) for sent in source_sents]),
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

    def __init__(
        self,
        itos: Sequence[str],
        embedding_size: int,
        word_dropout: float,
        words_padding_idx: int,
        unk_word: str,
    ):
        super(DefaultLexer, self).__init__()
        self.embedding = nn.Embedding(
            len(itos), embedding_size, padding_idx=words_padding_idx
        )
        self.embedding_size = embedding_size  # thats the interface property
        self.itos = itos
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.unk_word_idx = self.stoi[unk_word]
        self.word_dropout = word_dropout
        self._dpout = 0.0

    def train(self, mode: bool = True) -> "DefaultLexer":
        if mode:
            self._dpout = self.word_dropout
        else:
            self._dpout = 0.0
        return super().train(mode)

    def forward(self, word_sequences: torch.Tensor) -> torch.Tensor:
        """
        Takes words sequences codes as integer sequences and returns the embeddings
        """
        return self.embedding(word_sequences)

    def tokenize(self, tok_sequence: Sequence[str]) -> List[int]:
        """
        This maps word tokens to integer indexes.
        Args:
           tok_sequence: a sequence of strings
        Returns:
           a list of integers
        """
        word_idxes = [self.stoi.get(token, self.unk_word_idx) for token in tok_sequence]
        if self._dpout > 0:
            word_idxes = [
                word_sampler(widx, self.unk_word_idx, self._dpout)
                for widx in word_idxes
            ]
        return word_idxes

    def pad_batch(self, batch: Sequence[Sequence[int]]) -> torch.Tensor:
        """Pad a batch of sentences."""
        tensorized_sents = [torch.tensor(sent, dtype=torch.long) for sent in batch]
        return pad_sequence(
            tensorized_sents,
            padding_value=self.embedding.padding_idx,
            batch_first=True,
        )


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
        itos: Sequence[str],
        embedding_size: int,
        word_dropout: float,
        bert_modelfile: str,
        bert_layers: Sequence[int],
        bert_weighted: bool,
        words_padding_idx: int,
        unk_word: str,
        cased: bool = False,
    ):

        super(BertBaseLexer, self).__init__()
        self.itos = itos
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.unk_word_idx = self.stoi[unk_word]

        self.bert = AutoModel.from_pretrained(bert_modelfile, output_hidden_states=True)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            bert_modelfile, use_fast=True
        )

        self.BERT_PAD_IDX = self.bert_tokenizer.pad_token_id
        self.BERT_UNK_IDX = self.bert_tokenizer.unk_token_id
        self.embedding_size = embedding_size + self.bert.config.hidden_size

        self.embedding = nn.Embedding(
            len(self.itos),
            embedding_size,
            padding_idx=words_padding_idx,
        )

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

    def pad_batch(
        self,
        batch: Sequence[Tuple[Sequence[int], Sequence[int]]],
        padding_value: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a batch of sentences."""
        words_batch, bert_batch = [], []
        for words_sent, bert_subwords_sent in batch:
            words_batch.append(torch.tensor(words_sent, dtype=torch.long))
            bert_batch.append(torch.tensor(bert_subwords_sent, dtype=torch.long))
        return (
            pad_sequence(words_batch, batch_first=True, padding_value=padding_value),
            pad_sequence(
                bert_batch, batch_first=True, padding_value=self.embedding.padding_idx
            ),
        )

    def tokenize(self, tok_sequence: Sequence[str]) -> Tuple[List[int], List[int]]:
        """
        This maps word tokens to integer indexes.
        When a word decomposes as multiple BPE units, we keep only the first (!)
        Args:
           tok_sequence: a sequence of strings
        Returns:
           a list of integers
        """
        word_idxes = [self.stoi.get(token, self.unk_word_idx) for token in tok_sequence]
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
                word_sampler(widx, self.unk_word_idx, self._dpout)
                for widx in word_idxes
            ]

        # TODO: in the two line below, change unk to a spe
        word_idxes[0] = self.unk_word_idx
        bert_idxes[0] = 0
        return (word_idxes, bert_idxes)


Lexer = Union[DefaultLexer, BertBaseLexer]
