import json
import pathlib
from abc import abstractmethod
from collections import Counter
import tempfile
from typing import (
    Final,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)

import fasttext
import torch
import torch.jit
import transformers
from boltons.dictutils import OneToOne
from loguru import logger
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import BatchEncoding, TokenSpan


class LexingError(Exception):
    def __init__(self, message: str, sentence: Optional[str] = None):
        self.message = message
        self.sentence = sentence
        super().__init__(self.message)


@torch.jit.script
def integer_dropout(t: torch.Tensor, fill_value: int, p: float) -> torch.Tensor:
    mask = torch.empty_like(t, dtype=torch.bool).bernoulli_(p)
    return t.masked_fill(mask, fill_value)


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


_T_LEXER_SENT = TypeVar("_T_LEXER_SENT")
_T_LEXER_BATCH = TypeVar("_T_LEXER_BATCH")


class Lexer(Protocol[_T_LEXER_SENT, _T_LEXER_BATCH]):
    @abstractmethod
    def encode(self, tokens_sequence: Sequence[str]) -> _T_LEXER_SENT:
        raise NotImplementedError

    @abstractmethod
    def make_batch(self, batch: Sequence[_T_LEXER_SENT]) -> _T_LEXER_BATCH:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, inpt: _T_LEXER_BATCH) -> torch.Tensor:
        raise NotImplementedError


_T_CharRNNLexer = TypeVar("_T_CharRNNLexer", bound="CharRNNLexer")


class CharRNNLexer(nn.Module):
    """A lexer encoding words by running a bi-RNN on their characters

    ## Padding and special tokens

    - Char index `0` is for padding
    - Char index `1` is for special tokens that should not be split into individual characters

    These are **not** configurable in order to allow a simpler handling of the internal vocabulary
    """

    pad_idx: Final[int] = 0
    special_tokens_idx: Final[int] = 1

    def __init__(
        self,
        char_embeddings_dim: int,
        charset: Sequence[str],
        output_dim: int,
        special_tokens: Optional[Iterable[str]] = None,
    ):
        """Create a new CharRNNLexer.

        `charset` should not have any duplicates. Note also that the elements at `pad_idx` and
        `special_tokens idx` will have a special meaning. Building a new `CharRNNLexer` is
        preferably done via `from_chars`, which takes care of deduplicating and inserting special
        tokens.
        """
        super().__init__()
        try:
            self.vocabulary: OneToOne[str, int] = OneToOne.unique(
                (c, i) for i, c in enumerate(charset)
            )
        except ValueError as e:
            raise ValueError("Duplicated characters in charset") from e
        self.special_tokens = set([] if special_tokens is None else special_tokens)

        if output_dim % 2:
            raise ValueError("`output_dim` must be a multiple of 2")
        self.char_embeddings_dim: Final[int] = char_embeddings_dim
        self.output_dim: Final[int] = output_dim
        self.embedding = nn.Embedding(
            len(self.vocabulary), self.char_embeddings_dim, padding_idx=self.pad_idx
        )
        # FIXME: this supposes an even output dim
        self.char_bilstm = nn.LSTM(
            batch_first=True,
            bidirectional=True,
            hidden_size=self.output_dim // 2,
            input_size=self.char_embeddings_dim,
            num_layers=1,
        )

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        """
        Predicts the word embedding from the token characters.
        :param inpt: is a tensor of char indexes encoding a batch of tokens *×token_len
        :return: a word embedding tensor
        """
        # FIXME: there is probably a better way to do this since this results in tokens that are
        # full padding We need them of course (they will be cat to other padding embeddings in
        # graph_parser), but running the RNN on them is frustrating
        flattened_inputs = inpt.view(-1, inpt.shape[-1])
        embeddings = self.embedding(flattened_inputs)
        # ! FIXME: this does not take the padding into account
        _, (_, cembedding) = self.char_bilstm(embeddings)
        # TODO: why use the cell state and not the output state here?
        result = cembedding.view(*inpt.shape[:-1], self.output_dim)
        return result

    def word2charcodes(self, token: str) -> torch.Tensor:
        """Turn a string into a list of char codes.

        ## Notes

        Unknown chars are simply skipped. If a word only consists of unknow chars, the charcode
        corresponding to padding is used. This is not ideal but it's easy. In the future it
        **might** be marginally interesting to use UNK codes instead and apply dropout.
        """
        if token in self.special_tokens:
            res = [self.special_tokens_idx]
        else:
            res = [
                idx
                for idx in (self.vocabulary.get(c) for c in token)
                if idx is not None
            ]
            if not res:
                res = [self.pad_idx]
        return torch.tensor(res, dtype=torch.long)

    def encode(self, tokens_sequence: Sequence[str]) -> torch.Tensor:
        """Map word tokens to integer indices."""
        subword_indices = [self.word2charcodes(token) for token in tokens_sequence]
        # shape: sentence_length×num_chars_in_longest_word
        return pad_sequence(
            subword_indices, padding_value=self.pad_idx, batch_first=True
        )

    def make_batch(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        """Pad a batch of sentences."""
        # We need to pad manually because `pad_sequence` only accepts tensors that have all the same
        # dimensions except for one and we differ both in the sentence lengths dimension and in the
        # word length dimension
        res = torch.full(
            (
                len(batch),
                max(t.shape[0] for t in batch),
                max(t.shape[1] for t in batch),
            ),
            fill_value=self.pad_idx,
            dtype=torch.long,
        )
        for i, sent in enumerate(batch):
            res[i, : sent.shape[0], : sent.shape[1]] = sent
        return res

    def save(self, model_dir: pathlib.Path, save_weights: bool = True):
        model_dir.mkdir(exist_ok=True, parents=True)
        config_file = model_dir / "config.json"
        with open(config_file, "w") as out_stream:
            json.dump(
                {
                    "char_embeddings_dim": self.char_embeddings_dim,
                    "output_dim": self.output_dim,
                    "special_tokens": list(self.special_tokens),
                    "charset": [
                        self.vocabulary.inv[i] for i in range(len(self.vocabulary))
                    ],
                },
                out_stream,
            )
        if save_weights:
            torch.save(self.state_dict(), model_dir / "weights.pt")

    @classmethod
    def load(cls: Type[_T_CharRNNLexer], model_dir: pathlib.Path) -> _T_CharRNNLexer:
        with open(model_dir / "config.json") as in_stream:
            config = json.load(in_stream)
        res = cls(**config)
        weight_file = model_dir / "weights.pt"
        if weight_file.exists():
            res.load_state_dict(torch.load(weight_file))
        return res

    @classmethod
    def from_chars(
        cls: Type[_T_CharRNNLexer], chars: Iterable[str], **kwargs
    ) -> _T_CharRNNLexer:
        charset = sorted(set(chars))
        if wrong := [c for c in charset if len(c) > 1]:
            raise ValueError(f"Characters of length > 1 in charset: {wrong}")
        return cls(charset=["<pad>", "<special>", *charset], **kwargs)


_T_FastTextLexer = TypeVar("_T_FastTextLexer", bound="FastTextLexer")


class FastTextLexer(nn.Module):
    """
    This is subword model using FastText as backend.
    It follows the same interface as the CharRNN
    By convention, the padding vector is the last element of the embedding matrix
    """

    def __init__(
        self,
        fasttext_model: fasttext.FastText._FastText,
        special_tokens: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self.fasttext_model = fasttext_model
        weights = torch.from_numpy(fasttext_model.get_input_matrix())
        # Note: `vocab_size` is the size of the actual fasttext vocabulary. In pratice, the
        # embeddings here have two more tokens in their vocabulary: one for padding (embedding fixed
        # at 0, since the padding embedding never receive gradient in `nn.Embedding`) and one for
        # the special (root) tokens, with values sampled accross the vocabulary
        self.vocab_size: Final[int] = weights.shape[0]
        self.output_dim: Final[int] = weights.shape[1]
        # FIXME: this should really be called `special_tokens_embedding`
        # NOTE: I haven't thought too hard about this, maybe it's a bad idea
        root_embedding = weights[
            torch.randint(high=self.vocab_size, size=(self.output_dim,)),
            torch.arange(self.output_dim),
        ].unsqueeze(0)
        weights = torch.cat(
            (weights, torch.zeros((1, self.output_dim)), root_embedding), dim=0
        ).to(torch.float)
        weights.requires_grad = True
        self.embeddings = nn.Embedding.from_pretrained(
            weights, padding_idx=self.vocab_size
        )
        self.special_tokens: Set = set([] if special_tokens is None else special_tokens)
        self.special_tokens_idx: Final[int] = self.vocab_size + 1
        self.pad_idx: Final[int] = self.embeddings.padding_idx

    def subwords_idxes(self, token: str) -> torch.Tensor:
        """Returns a list of ft subwords indexes a token"""
        return torch.from_numpy(self.fasttext_model.get_subwords(token)[1])

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        """
        :param inpt: a batch of subwords of shape `$*×subwords_dim×features_dim$`
        :return: the fasttext embeddings for this batch
        """
        # NOTE: the padding embedding is 0 and should not be modified during training (as per the
        # `torch.nn.Embedding` doc) so the mean here does not include padding subwords
        # NOTE: we use the *mean* of the *input* vectors of the subwords, following [the original
        # FastText
        # implementation](https://github.com/facebookresearch/fastText/blob/a20c0d27cd0ee88a25ea0433b7f03038cd728459/src/fasttext.cc#L117)
        # instead of using the *sum* of *unspecified* (either input or output) vectors as per the
        # original FastText paper (“Enriching Word Vectors with Subword Information”, Bojanowski et
        # al., 2017)
        return self.embeddings(inpt).mean(dim=-2)

    def word2subcodes(self, token: str) -> torch.Tensor:
        """
        Turns a string into a list of subword codes.
        """
        if token in self.special_tokens:
            return torch.tensor([self.special_tokens_idx], dtype=torch.long)
        return self.subwords_idxes(token)

    def encode(self, tokens_sequence: Sequence[str]) -> torch.Tensor:
        """Map word tokens to integer indices."""
        subword_indices = [self.word2subcodes(token) for token in tokens_sequence]
        # shape: sentence_length×num_subwords_in_longest_word
        return pad_sequence(
            subword_indices, padding_value=self.pad_idx, batch_first=True
        )

    def make_batch(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        """Pad a batch of sentences."""
        # We need to pad manually because `pad_sequence` only accepts tensors that have all the same
        # dimensions except for one and we differ both in the sentence lengths dimension and in the
        # word length dimension
        res = torch.full(
            (
                len(batch),
                max(t.shape[0] for t in batch),
                max(t.shape[1] for t in batch),
            ),
            fill_value=self.pad_idx,
            dtype=torch.long,
        )
        for i, sent in enumerate(batch):
            res[i, : sent.shape[0], : sent.shape[1]] = sent
        # shape: batch_size×max_sentence_length×num_subwords_in_longest_word
        return res

    def save(self, model_dir: pathlib.Path, save_weights: bool = True):
        model_dir.mkdir(exist_ok=True, parents=True)
        config_file = model_dir / "config.json"
        with open(config_file, "w") as out_stream:
            json.dump(
                {
                    "special_tokens": list(self.special_tokens),
                },
                out_stream,
            )
        # Not necessarily very useful (since it doesn't save the fine-tuned special tokens embedding
        # so if you want to save the model you should still use save_weights) but nice: set the
        # FastText model weights to the fine-tuned ones
        self.fasttext_model.set_matrices(
            self.embeddings.weight[:-2].numpy(), self.fasttext_model.get_output_matrix()
        )
        self.fasttext_model.save_model(str(model_dir / "fasttext_model.bin"))
        if save_weights:
            torch.save(self.state_dict(), model_dir / "weights.pt")

    @classmethod
    def load(
        cls: Type[_T_FastTextLexer],
        model_dir: pathlib.Path,
    ) -> _T_FastTextLexer:
        with open(model_dir / "config.json") as in_stream:
            config = json.load(in_stream)
        res = cls.from_fasttext_model(model_dir / "fasttext_model.bin", **config)
        weight_file = model_dir / "weights.pt"
        if weight_file.exists():
            res.load_state_dict(torch.load(weight_file))
        return res

    @classmethod
    def from_fasttext_model(
        cls: Type[_T_FastTextLexer], model_file: Union[str, pathlib.Path], **kwargs
    ) -> _T_FastTextLexer:
        return cls(fasttext.load_model(str(model_file)), **kwargs)

    @classmethod
    def from_raw(
        cls: Type[_T_FastTextLexer],
        raw_text_path: Union[str, pathlib.Path],
        **kwargs,
    ) -> _T_FastTextLexer:
        logger.info("Training fasttext model")
        # TODO: make the hyperparameters here configurable?
        model = fasttext.train_unsupervised(
            str(raw_text_path), model="skipgram", neg=10, minCount=5, epoch=10
        )
        return cls(model, **kwargs)

    @classmethod
    def from_sents(
        cls: Type[_T_FastTextLexer],
        sents: Iterable[List[str]],
        **kwargs,
    ) -> _T_FastTextLexer:
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_file = pathlib.Path(tmp_dir) / "train.txt"
            with open(train_file, "w") as out_stream:
                for s in sents:
                    out_stream.write(" ".join(s))
                    out_stream.write("\n")
            return cls.from_raw(train_file, **kwargs)


_T_WordEmbeddingsLexer = TypeVar("_T_WordEmbeddingsLexer", bound="WordEmbeddingsLexer")


class WordEmbeddingsLexer(nn.Module):
    """
    This is the basic lexer wrapping an embedding layer.
    """

    def __init__(
        self,
        vocabulary: Sequence[str],
        embeddings_dim: int,
        word_dropout: float,
        words_padding_idx: int,
        unk_word: str,
    ):
        """Create a new WordEmbeddingsLexer.

        `vocabulary` should not have any duplicates. Note also that the elements at `pad_idx` and
        `special_tokens idx` will have a special meaning. Building a new `WordEmbeddingsLexer` is
        preferably done via `from_words`, which takes care of deduplicating and inserting special
        tokens.
        """
        super().__init__()
        try:
            self.vocabulary: OneToOne[str, int] = OneToOne.unique(
                (c, i) for i, c in enumerate(vocabulary)
            )
        except ValueError as e:
            raise ValueError("Duplicated words in vocabulary") from e
        self.embedding = nn.Embedding(
            len(self.vocabulary), embeddings_dim, padding_idx=words_padding_idx
        )
        self.output_dim = embeddings_dim
        self.unk_word_idx = self.vocabulary[unk_word]
        self.word_dropout = word_dropout
        self._dpout = 0.0

    def train(
        self: _T_WordEmbeddingsLexer, mode: bool = True
    ) -> _T_WordEmbeddingsLexer:
        if mode:
            self._dpout = self.word_dropout
        else:
            self._dpout = 0.0
        return super().train(mode)

    def forward(self, word_sequences: torch.Tensor) -> torch.Tensor:
        """
        Takes words sequences codes as integer sequences and returns the embeddings
        """
        if self._dpout:
            word_sequences = integer_dropout(
                word_sequences, self.unk_word_idx, self._dpout
            )
        return self.embedding(word_sequences)

    def encode(self, tokens_sequence: Sequence[str]) -> torch.Tensor:
        """Map word tokens to integer indices."""
        word_idxes = [
            self.vocabulary.get(token, self.unk_word_idx) for token in tokens_sequence
        ]
        return torch.tensor(word_idxes)

    def make_batch(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        """Pad a batch of sentences."""
        return pad_sequence(
            batch,
            padding_value=self.embedding.padding_idx,
            batch_first=True,
        )

    def save(self, model_dir: pathlib.Path, save_weights: bool = True):
        model_dir.mkdir(exist_ok=True, parents=True)
        config_file = model_dir / "config.json"
        with open(config_file, "w") as out_stream:
            json.dump(
                {
                    "embeddings_dim": self.output_dim,
                    "unk_word": self.vocabulary.inv[self.unk_word_idx],
                    "vocabulary": [
                        self.vocabulary.inv[i] for i in range(len(self.vocabulary))
                    ],
                    "word_dropout": self.word_dropout,
                    "words_padding_idx": self.embedding.padding_idx,
                },
                out_stream,
            )
        if save_weights:
            torch.save(self.state_dict(), model_dir / "weights.pt")

    @classmethod
    def load(
        cls: Type[_T_WordEmbeddingsLexer], model_dir: pathlib.Path
    ) -> _T_WordEmbeddingsLexer:
        with open(model_dir / "config.json") as in_stream:
            config = json.load(in_stream)
        res = cls(**config)
        weight_file = model_dir / "weights.pt"
        if weight_file.exists():
            res.load_state_dict(torch.load(weight_file))
        return res

    @classmethod
    def from_words(
        cls: Type[_T_WordEmbeddingsLexer], words: Iterable[str], **kwargs
    ) -> _T_WordEmbeddingsLexer:
        vocabulary = sorted(set(words))
        return cls(vocabulary=vocabulary, **kwargs)


def freeze_module(module: nn.Module, freezing: bool = True):
    """Make a `torch.nn.Module` either finetunable 🔥 or frozen ❄.

    **WARNINGS**

    - Freezing a module will put it in eval mode (since a frozen module can't be in training mode),
      but unfreezing it will not put it back in training mode (since an unfrozen module still has an
      eval mode that you might want to use), you have to do that yourself.
    - Manually setting the submodules of a frozen module to train is not disabled, but if you want
      to do that, writing a custom freezing function is probably a better idea.
    - Freezing does not save the parameters `requires_grad`, so if some parameters do not require
      grad even at training, this will mess that up. Again, in that case, write a custom function.
    """

    # This will replace the module's train function when freezing
    def no_train(model, mode=True):
        return model

    if freezing:
        module.eval()
        module.train = no_train  # type: ignore[assignment]
        module.requires_grad_(False)
    else:
        module.requires_grad_(True)
        module.train = type(module).train  # type: ignore[assignment]


_T_BertLexerBatch = TypeVar("_T_BertLexerBatch", bound="BertLexerBatch")


class BertLexerBatch(NamedTuple):
    word_indices: torch.Tensor
    bert_encoding: BatchEncoding
    subword_alignments: Sequence[Sequence[TokenSpan]]

    def to(
        self: _T_BertLexerBatch, device: Union[str, torch.device]
    ) -> _T_BertLexerBatch:
        return type(self)(
            self.word_indices.to(device=device),
            self.bert_encoding.to(device=device),
            self.subword_alignments,
        )

    def size(self, *args, **kwargs):
        return self.word_indices.size(*args, **kwargs)


class BertLexerSentence(NamedTuple):
    word_indices: Sequence[int]
    bert_encoding: BatchEncoding
    subwords_alignments: Sequence[TokenSpan]


def align_with_special_tokens(
    word_lengths: Sequence[int],
    mask=Sequence[int],
    special_tokens_code: int = 1,
    sequence_tokens_code: int = 0,
) -> List[TokenSpan]:
    """Provide a word→subwords alignements using an encoded sentence special tokens mask.

    This is only useful for the non-fast 🤗 tokenizers, since the fast ones have native APIs to do
    that, we also return 🤗 `TokenSpan`s for compatibility with this API.
    """
    res: List[TokenSpan] = []
    pos = 0
    for length in word_lengths:
        while mask[pos] == special_tokens_code:
            pos += 1
        word_end = pos + length
        if any(token_type != sequence_tokens_code for token_type in mask[pos:word_end]):
            raise ValueError(
                "mask incompatible with tokenization:"
                f" needed {length} true tokens (1) at position {pos},"
                f" got {mask[pos:word_end]} instead"
            )
        res.append(TokenSpan(pos, word_end))
        pos = word_end

    return res


class BertBaseLexer(nn.Module):
    """
    This Lexer performs tokenization and embedding mapping with BERT
    style models. It concatenates a standard embedding with a BERT
    embedding.
    """

    def __init__(
        self,
        itos: Sequence[str],
        unk_word: str,
        embedding_dim: int,
        word_dropout: float,
        bert_layers: Optional[Sequence[int]],
        bert_model: str,
        bert_subwords_reduction: Literal["first", "mean"],
        bert_weighted: bool,
        words_padding_idx: int,
    ):

        super().__init__()
        self.itos = itos
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.unk_word_idx = self.stoi[unk_word]

        try:
            self.bert = transformers.AutoModel.from_pretrained(
                bert_model, output_hidden_states=True
            )
        except OSError:
            config = transformers.AutoConfig.from_pretrained(bert_model)
            self.bert = transformers.AutoModel.from_config(config)

        self.bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
            bert_model, use_fast=True
        )
        # Shim for the weird idiosyncrasies of the RoBERTa tokenizer
        if isinstance(self.bert_tokenizer, transformers.GPT2TokenizerFast):
            self.bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
                bert_model, use_fast=True, add_prefix_space=True
            )

        self.max_length = min(
            self.bert_tokenizer.max_len_single_sentence,
            getattr(self.bert.config, "max_position_embeddings", float("inf"))
            - self.bert_tokenizer.num_special_tokens_to_add(pair=False),
        )

        self.output_dim = embedding_dim + self.bert.config.hidden_size

        self.embedding = nn.Embedding(
            len(self.itos),
            embedding_dim,
            padding_idx=words_padding_idx,
        )

        self.word_dropout = word_dropout
        self._dpout = 0.0

        # 🤗 has no unified API for the number of layers
        num_layers = next(
            n
            for param_name in ("num_layers", "n_layers", "num_hidden_layers")
            for n in [getattr(self.bert.config, param_name, None)]
            if n is not None
        )
        if bert_layers is None:
            bert_layers = list(range(num_layers))
        elif not all(
            -num_layers <= layer_idx < num_layers for layer_idx in bert_layers
        ):
            raise ValueError(
                f"Wrong BERT layer selections for a model with {num_layers} layers: {bert_layers}"
            )
        self.bert_layers = bert_layers
        # Deactivate layerdrop if available
        if hasattr(self.bert, "layerdrop"):
            self.bert.layerdrop = 0.0
        # TODO: check if the value is allowed?
        self.bert_subwords_reduction = bert_subwords_reduction
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

    def forward(self, inpt: BertLexerBatch) -> torch.Tensor:
        word_indices = inpt.word_indices
        if self._dpout:
            word_indices = integer_dropout(word_indices, self.unk_word_idx, self._dpout)
        word_embeddings = self.embedding(word_indices)

        bert_layers = self.bert(
            input_ids=inpt.bert_encoding["input_ids"], return_dict=True
        ).hidden_states
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
            # shape: batch×subwords_sequence×features
            bert_subword_embeddings = self.layers_gamma * torch.einsum(
                "l,lbsf->bsf", normal_weights, selected_bert_layers
            )
        else:
            bert_subword_embeddings = selected_bert_layers.mean(dim=0)
        # We already know the shape the BERT embeddings should have and we pad with zeros
        # shape: batch×sentence×features
        bert_embeddings = word_embeddings.new_zeros(
            (
                word_embeddings.shape[0],
                word_embeddings.shape[1],
                bert_subword_embeddings.shape[2],
            )
        )
        bert_embeddings[:, 0, ...] = bert_subword_embeddings.mean(dim=1)
        # FIXME: this loop is embarassingly parallel, there must be a way to parallelize it
        for sent_n, alignment in enumerate(inpt.subword_alignments):
            # TODO: If we revise the alignment format, this could probably be made faster using
            # <https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html>
            # The word indices start at 1 because word 0 is the root token, for which we have no
            # bert embedding so we use the average of all subword embeddings
            for word_n, span in enumerate(alignment, start=1):
                # shape: `span.end-span.start×features`
                bert_word_embeddings = bert_subword_embeddings[
                    sent_n, span.start : span.end, ...
                ]
                if self.bert_subwords_reduction == "first":
                    reduced_bert_word_embedding = bert_word_embeddings[0, ...]
                elif self.bert_subwords_reduction == "mean":
                    reduced_bert_word_embedding = bert_word_embeddings.mean(dim=0)
                else:
                    raise ValueError(
                        f"Unknown reduction {self.bert_subwords_reduction}"
                    )
                bert_embeddings[sent_n, word_n, ...] = reduced_bert_word_embedding

        return torch.cat((word_embeddings, bert_embeddings), dim=2)

    def make_batch(
        self,
        batch: Sequence[BertLexerSentence],
    ) -> BertLexerBatch:
        """Pad a batch of sentences."""
        words_batch, bert_batch, alignments = [], [], []
        for sent in batch:
            words_batch.append(torch.tensor(sent.word_indices, dtype=torch.long))
            bert_batch.append(sent.bert_encoding)
            alignments.append(sent.subwords_alignments)
        bert_encoding = self.bert_tokenizer.pad(bert_batch)
        bert_encoding.convert_to_tensors("pt")
        return BertLexerBatch(
            pad_sequence(
                words_batch, batch_first=True, padding_value=self.embedding.padding_idx
            ),
            bert_encoding,
            alignments,
        )

    def encode(self, tokens_sequence: Sequence[str]) -> BertLexerSentence:
        """
        This maps word tokens to integer indexes.
        Args:
           tok_sequence: a sequence of strings
        """
        word_idxes = [
            self.stoi.get(token, self.unk_word_idx) for token in tokens_sequence
        ]

        # The root token is remved here since the BERT model has no reason to know of it
        unrooted_tok_sequence = tokens_sequence[1:]
        # NOTE: for now the 🤗 tokenizer interface is not unified between fast and non-fast
        # tokenizers AND not all tokenizers support the fast mode, so we have to do this little
        # awkward dance. Eventually we should be able to remove the non-fast branch here.
        if self.bert_tokenizer.is_fast:
            bert_encoding = self.bert_tokenizer(
                unrooted_tok_sequence,
                is_split_into_words=True,
                return_special_tokens_mask=True,
            )
            if len(bert_encoding.data["input_ids"]) > self.max_length:
                raise LexingError(
                    f"Sentence too long for this transformer model ({len(bert_encoding.data['input_ids'])} tokens > {self.max_length})",
                    str(unrooted_tok_sequence),
                )
            # TODO: there might be a better way to do this?
            alignments = [
                bert_encoding.word_to_tokens(i)
                for i in range(len(unrooted_tok_sequence))
            ]
            i = next((i for i, a in enumerate(alignments) if a is None), None)
            if i is not None:
                raise LexingError(
                    f"Unencodable token {unrooted_tok_sequence[i]!r} at {i}",
                    str(unrooted_tok_sequence),
                )
        else:
            bert_tokens = [
                self.bert_tokenizer.tokenize(token) for token in unrooted_tok_sequence
            ]
            i = next((i for i, s in enumerate(bert_tokens) if not s), None)
            if i is not None:
                raise LexingError(
                    f"Unencodable token {unrooted_tok_sequence[i]!r} at {i}",
                    str(unrooted_tok_sequence),
                )
            subtokens_sequence = [
                subtoken for token in bert_tokens for subtoken in token
            ]
            if len(subtokens_sequence) > self.max_length:
                raise LexingError(
                    f"Sentence too long for this transformer model ({len(subtokens_sequence)} tokens > {self.max_length})",
                    str(unrooted_tok_sequence),
                )
            bert_encoding = self.bert_tokenizer.encode_plus(
                subtokens_sequence,
                return_special_tokens_mask=True,
            )
            bert_word_lengths = [len(word) for word in bert_tokens]
            alignments = align_with_special_tokens(
                bert_word_lengths,
                bert_encoding["special_tokens_mask"],
            )

        return BertLexerSentence(word_idxes, bert_encoding, alignments)
