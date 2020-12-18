from typing import (
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)
import torch
import torch.jit
import transformers
import fasttext
import os.path
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TokenSpan
from collections import Counter
from tempfile import gettempdir

# Python 3.7 shim
try:
    from typing import Final, Literal
except ImportError:
    from typing_extensions import Final, Literal  # type: ignore


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
        """
        if token in self.special_tokens:
            return [self.SPECIAL_TOKENS_IDX]
        res = [self.c2idx[c] for c in token if c in self.c2idx]
        if not res:
            return [self.PAD_IDX]
        return res

    def batchedtokens2codes(self, toklist: List[str]) -> torch.Tensor:
        """
        Codes a list of tokens as a batch of lists of charcodes and pads them if needed
        """
        charcodes = [
            torch.tensor(self.word2charcodes(token), dtype=torch.long)
            for token in toklist
        ]
        return pad_sequence(charcodes, padding_value=self.PAD_IDX, batch_first=True)

    def batch_chars(self, sent_batch: List[List[str]]) -> Iterable[torch.Tensor]:
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
        # FIXME: minor issue, but this way, if the special tokens appear in the word list, their
        # character will be in the charset (and that might not be a good idea)
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
            self.embedding_size // 2,
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
        # ! FIXME: this does not take the padding into account
        outputs, (_, cembedding) = self.char_bilstm(embeddings)
        # TODO: why use the cell state and not the output state here?
        result = cembedding.view(-1, self.embedding_size)
        return result


class FastTextDataSet:
    """
    Namespace for simulating a fasttext encoded dataset.
    By convention, the padding vector is the last element of the embedding matrix
    """

    def __init__(
        self,
        fasttextmodel: "FastTextTorch",
        special_tokens: Optional[Iterable[str]] = None,
    ):
        self.fasttextmodel = fasttextmodel
        self.special_tokens = set([] if special_tokens is None else special_tokens)
        self.special_tokens_idx: Final[int] = self.fasttextmodel.vocab_size
        self.pad_idx: Final[int] = self.fasttextmodel.vocab_size + 1

    def word2subcodes(self, token: str) -> torch.Tensor:
        """
        Turns a string into a list of subword codes.
        """
        if token == "":
            return torch.tensor([self.pad_idx], dtype=torch.long)
        elif token in self.special_tokens:
            return torch.tensor([self.special_tokens_idx], dtype=torch.long)
        return self.fasttextmodel.subwords_idxes(token)

    def batch_tokens(self, token_sequence):
        """
        Batches a list of tokens as a padded matrix of subword codes.
        :param token_sequence : a sequence of strings
        :return: a list of of list of codes (matrix with padding)
        """
        subcodes = [self.word2subcodes(token) for token in token_sequence]
        return pad_sequence(subcodes, padding_value=self.pad_idx, batch_first=True)

    def batch_sentences(self, sent_batch: List[List[str]]) -> Iterable[torch.Tensor]:
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
        weights = torch.from_numpy(fasttextmodel.get_input_matrix())
        # Note: `vocab_size` is the size of the actual fasttext vocabulary. In pratice, the
        # embeddings here have two more tokens in their vocabulary: one for padding (embedding fixed
        # at 0, since the padding embedding never receive gradient in `nn.Embedding`) and one for
        # the special (root) tokens, with values sampled accross the vocabulary
        self.vocab_size, self.embedding_size = weights.shape
        root_embedding = weights[
            torch.randint(high=self.vocab_size, size=(self.embedding_size,)),
            torch.arange(self.embedding_size),
        ].unsqueeze(0)
        weights = torch.cat(
            (weights, torch.zeros((1, self.embedding_size)), root_embedding), dim=0
        ).to(torch.float)
        weights.requires_grad = True
        self.embeddings = nn.Embedding.from_pretrained(
            weights, padding_idx=self.vocab_size + 1
        )

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
        # Note: the padding embedding is 0 and should not be modified during training (as per the
        # `torch.nn.Embedding` doc) so the mean here does not include padding tokens
        return self.embeddings(xinput).mean(dim=1)

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
            # TODO: make the hyperparameters here configurable
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
            # TODO: make the hyperparameters here configurable
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
        if self._dpout:
            word_sequences = integer_dropout(
                word_sequences, self.unk_word_idx, self._dpout
            )
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
    - Freezing does not save the parameters `requires_grad`, so if some parameters do not require
      grad even at training, this will mess that up. Again, in that case, write a custom function.
    """

    # This will replace the module's train function when freezing
    def no_train(model, mode=True):
        return model

    if freezing:
        module.eval()
        module.train = no_train
        module.requires_grad_(False)
    else:
        module.requires_grad_(True)
        module.train = type(module).train


class BertLexerBatch(NamedTuple):
    word_indices: torch.Tensor
    bert_encoding: BatchEncoding
    subword_alignments: Sequence[Sequence[TokenSpan]]

    def to(self, device: Union[str, torch.device]) -> "BertLexerBatch":
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
        embedding_size: int,
        word_dropout: float,
        bert_layers: Sequence[int],
        bert_modelfile: str,
        bert_subwords_reduction: Literal["first", "mean"],
        bert_weighted: bool,
        words_padding_idx: int,
    ):

        super(BertBaseLexer, self).__init__()
        self.itos = itos
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.unk_word_idx = self.stoi[unk_word]

        self.bert = AutoModel.from_pretrained(bert_modelfile, output_hidden_states=True)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            bert_modelfile, use_fast=True
        )
        # Shim for the weird idiosyncrasies of the RoBERTa tokenizer
        if isinstance(self.bert_tokenizer, transformers.GPT2TokenizerFast):
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                bert_modelfile, use_fast=True, add_prefix_space=True
            )

        self.embedding_size = embedding_size + self.bert.config.hidden_size

        self.embedding = nn.Embedding(
            len(self.itos),
            embedding_size,
            padding_idx=words_padding_idx,
        )

        self.word_dropout = word_dropout
        self._dpout = 0.0

        self.bert_layers = bert_layers
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
        for sent_n, alignment in enumerate(inpt.subword_alignments):
            # The word indices start at 1 because word 0 is the root token, for which we have no
            # bert embedding so we use the average of all subword embeddings
            for word_n, span in enumerate(alignment, start=1):
                # shape: `span.end-span.start×features`
                # it along the first dimension
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

    def pad_batch(
        self,
        batch: Sequence[BertLexerSentence],
        padding_value: int = 0,
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
            pad_sequence(words_batch, batch_first=True, padding_value=padding_value),
            bert_encoding,
            alignments,
        )

    def tokenize(self, tok_sequence: Sequence[str]) -> BertLexerSentence:
        """
        This maps word tokens to integer indexes.
        Args:
           tok_sequence: a sequence of strings
        """
        word_idxes = [self.stoi.get(token, self.unk_word_idx) for token in tok_sequence]

        # We deal with the root token separately since the BERT model has no reason to know of it
        unrooted_tok_sequence = tok_sequence[1:]
        # NOTE: for now the 🤗 tokenizer interface is not unified between fast and non-fast
        # tokenizers AND not all tokenizers support the fast mode, so we have to do this little
        # awkward dance. Eventually we should be able to remove the non-fast branch here.
        if self.bert_tokenizer.is_fast:
            bert_encoding = self.bert_tokenizer(
                unrooted_tok_sequence,
                is_split_into_words=True,
                return_special_tokens_mask=True,
            )
            # TODO: there might be a better way to do this?
            alignments = [
                bert_encoding.word_to_tokens(i)
                for i in range(len(unrooted_tok_sequence))
            ]
        else:
            bert_tokens = [
                self.bert_tokenizer.tokenize(token) for token in unrooted_tok_sequence
            ]
            bert_encoding = self.bert_tokenizer.encode_plus(
                [subtoken for token in bert_tokens for subtoken in token],
                return_special_tokens_mask=True,
            )
            bert_word_lengths = [len(word) for word in bert_tokens]
            alignments = align_with_special_tokens(
                bert_word_lengths,
                bert_encoding["special_tokens_mask"],
            )

        return BertLexerSentence(word_idxes, bert_encoding, alignments)


Lexer = Union[DefaultLexer, BertBaseLexer]
