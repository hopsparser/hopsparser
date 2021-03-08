from dataclasses import dataclass
import pathlib
from random import shuffle
from typing import (
    IO,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

import torch
from torch.nn.utils.rnn import pad_sequence
from typing_extensions import Final

from npdependency import lexers
from npdependency.lexers import BertLexerBatch, BertLexerSentence
from npdependency.utils import smart_open


class MWERange(NamedTuple):
    start: int
    end: int
    form: str

    def to_conll(self) -> str:
        return f"{self.start}-{self.end}\t{self.form}\t_\t_\t_\t_\t_\t_\t_\t_"


class Edge(NamedTuple):
    gov: int
    label: str
    dep: int


@dataclass(eq=False)
class DepNode:
    identifier: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str

    def to_conll(self) -> str:
        return f"{self.identifier}\t{self.form}\t{self.lemma}\t{self.upos}\t{self.xpos}\t{self.feats}\t{self.head}\t{self.deprel}\t{self.deps}\t{self.misc}"


class DepGraph:

    ROOT_TOKEN = "<root>"

    def __init__(
        self,
        nodes: Iterable[DepNode],
        mwe_ranges: Optional[Iterable[MWERange]] = None,
        metadata: Optional[Iterable[str]] = None,
    ):

        self.nodes = list(nodes)

        govs = {n.identifier: n.head for n in self.nodes}
        if 0 not in govs.values():
            raise ValueError("Malformed tree: no root")
        if len(set(govs.values()).difference(govs.keys())) > 1:
            raise ValueError("Malformed tree: non-connex")

        self.mwe_ranges = [] if mwe_ranges is None else list(mwe_ranges)
        self.metadata = [] if metadata is None else list(metadata)

    @property
    def words(self) -> List[str]:
        """
        A list where each element list[i] is the form of
        the position of the governor of the word at position i.
        """
        return [self.ROOT_TOKEN, *(n.form for n in self.nodes)]

    @property
    def pos_tags(self) -> List[str]:
        """
        A list where each element list[i] is the upos of
        the position of the governor of the word at position i.
        """
        return [self.ROOT_TOKEN, *(n.upos for n in self.nodes)]

    @property
    def heads(self) -> List[int]:
        """
        A list where each element list[i] is the index of
        the position of the governor of the word at position i.
        """
        return [0, *(n.head for n in self.nodes)]

    @property
    def deprels(self) -> List[str]:
        """
        A list where each element list[i] is the label of
        the position of the governor of the word at position i.
        """
        return [self.ROOT_TOKEN, *(n.deprel for n in self.nodes)]

    def replace(
        self, edges: Optional[Iterable[Edge]], pos_tags: Optional[Iterable[str]]
    ) -> "DepGraph":
        """Return a new `DepGraph`, identical to `self` except for its dependencies and pos tags (if specified).

        If neither `edges` nor `pos_tags` is provided, this returns a shallow copy of `self`.
        """
        if edges is None:
            govs = {n.identifier: n.head for n in self.nodes}
            labels = {n.identifier: n.deprel for n in self.nodes}
        else:
            govs = {e.dep: e.gov for e in edges}
            labels = {e.dep: e.label for e in edges}
        if pos_tags is None:
            pos_tags = self.pos_tags[1:]
        pos = {i: tag for i, tag in enumerate(pos_tags, start=1)}
        new_nodes = [
            DepNode(
                identifier=node.identifier,
                form=node.form,
                lemma=node.lemma,
                upos=pos[node.identifier],
                xpos=node.xpos,
                feats=node.feats,
                head=govs[node.identifier],
                deprel=labels[node.identifier],
                deps=node.deps,
                misc=node.misc,
            )
            for node in self.nodes
        ]
        return type(self)(
            nodes=new_nodes,
            metadata=self.metadata[:],
            mwe_ranges=self.mwe_ranges[:],
        )

    @classmethod
    def from_conllu(cls, istream: Iterable[str]) -> "DepGraph":
        """
        Reads a conll tree from input stream
        """
        conll = []
        metadata = []
        for line in istream:
            if line.startswith("#"):
                metadata.append(line.strip())
                continue
            conll.append(line.strip().split("\t"))

        mwe_ranges = []
        nodes = []
        for cols in conll:
            if "-" in cols[0]:
                mwe_start, mwe_end = cols[0].split("-")
                mwe_ranges.append(MWERange(int(mwe_start), int(mwe_end), cols[1]))
                continue
            if len(cols) < 2:
                raise ValueError("Too few columns to build a DepNode")
            elif len(cols) < 10:
                cols = [*cols, *("_" for _ in range(10 - len(cols)))]
            if cols[6] == "_":
                cols[6] = "0"
            node = DepNode(
                identifier=int(cols[0]),
                form=cols[1],
                lemma=cols[2],
                upos=cols[3],
                xpos=cols[4],
                feats=cols[5],
                head=int(cols[6]),
                deprel=cols[7],
                deps=cols[8],
                misc=cols[9],
            )
            nodes.append(node)
        return cls(
            nodes=nodes,
            mwe_ranges=mwe_ranges,
            metadata=metadata,
        )

    def __str__(self):
        """
        CoNLL-U string for the dep tree
        """
        lines = self.metadata
        for n in self.nodes:
            mwe_list = [mwe for mwe in self.mwe_ranges if mwe.start == n.identifier]
            for mwe in mwe_list:
                lines.append(mwe.to_conll())
            lines.append(n.to_conll())
        return "\n".join(lines)

    def __len__(self):
        return len(self.words)


class DependencyBatch(NamedTuple):
    """Batched and padded sentences.

    ## Attributes

    - `trees` The sentences as `DepGraph`s for rich attribute access.
    - `chars` Encoded chars as a sequence of `LongTensor`. `chars[i][j, k]` is the k-th character of
      the i-th word of the j-th sentence in the batch.
    - `subwords` Encoded FastText subwords as a sequence of `LongTensor`. As with `chars`,
      `subwords[i][j, k]` is the k-th subword of the i-th word of the j-th sentence in the batch.
    - `encoded_words` The words of the sentences, encoded and batched by a lexer and meant to be
      consumed by it directly. The details stay opaque at this level, see the relevant lexer
      instead.
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

    trees: Sequence[DepGraph]
    chars: Sequence[torch.Tensor]
    subwords: Sequence[torch.Tensor]
    encoded_words: Union[torch.Tensor, BertLexerBatch]
    tags: torch.Tensor
    heads: torch.Tensor
    labels: torch.Tensor
    sent_lengths: torch.Tensor
    content_mask: torch.Tensor

    def to(self, device: Union[str, torch.device]) -> "DependencyBatch":
        encoded_words = self.encoded_words.to(device)
        chars = [token.to(device) for token in self.chars]
        subwords = [token.to(device) for token in self.subwords]
        return type(self)(
            trees=self.trees,
            chars=chars,
            subwords=subwords,
            encoded_words=encoded_words,
            tags=self.tags.to(device),
            heads=self.heads.to(device),
            labels=self.labels.to(device),
            sent_lengths=self.sent_lengths,
            content_mask=self.content_mask.to(device),
        )


class DependencyDataset:
    """
    A representation of the DepBank for efficient processing.
    This is a sorted dataset.
    """

    PAD_IDX: Final[int] = 0
    PAD_TOKEN: Final[str] = "<pad>"
    UNK_WORD: Final[str] = "<unk>"
    # Labels that are -100 are ignored in torch crossentropy (we still set it explicitely in
    # `graph_parser`)
    LABEL_PADDING: Final[int] = -100

    @staticmethod
    def read_conll(
        filename: Union[str, pathlib.Path, IO[str]],
        max_tree_length: Optional[int] = None,
    ) -> List[DepGraph]:
        print(f"Reading treebank from {filename}")
        with smart_open(filename) as istream:
            trees = []
            current_tree_lines: List[str] = []
            # Add a dummy empty line to flush the last tree even if the CoNLL-U mandatory empty last
            # line is absent
            for line in (*istream, ""):
                if not line or line.isspace():
                    if current_tree_lines:
                        if (
                            max_tree_length is None
                            or len(current_tree_lines) <= max_tree_length
                        ):
                            trees.append(DepGraph.from_conllu(current_tree_lines))
                            current_tree_lines = []
                        else:
                            print(
                                f"Dropped tree with length {len(current_tree_lines)} > {max_tree_length}",
                            )
                else:
                    current_tree_lines.append(line)
        return trees

    def __init__(
        self,
        treelist: List[DepGraph],
        lexer: lexers.Lexer,
        chars_lexer: lexers.CharRNNLexer,
        ft_lexer: lexers.FastTextLexer,
        use_labels: Optional[Sequence[str]] = None,
        use_tags: Optional[Sequence[str]] = None,
    ):
        self.lexer = lexer
        self.chars_lexer = chars_lexer
        self.ft_lexer = ft_lexer
        self.treelist = treelist
        if use_labels:
            self.itolab = use_labels
            self.labtoi = {label: idx for idx, label in enumerate(self.itolab)}
        else:
            self.init_labels(self.treelist)
        if use_tags:
            self.itotag = use_tags
            self.tagtoi = {tag: idx for idx, tag in enumerate(self.itotag)}
        else:
            self.init_tags(self.treelist)
        self.encoded_words: List[Union[torch.Tensor, BertLexerSentence]] = []
        self.encoded_ft: List[torch.Tensor] = []
        self.encoded_chars: List[torch.Tensor] = []
        self.heads: List[List[int]] = []
        self.labels: List[List[int]] = []
        self.tags: List[List[int]] = []
        self.encode()

    # TODO: use an `encoded_tree` object instead of hardcoding the lexers here
    def encode(self):
        self.encoded_words = []
        self.encoded_ft = []
        self.encoded_chars = []
        self.heads = []
        self.labels = []
        self.tags = []

        for tree in self.treelist:
            self.encoded_words.append(self.lexer.encode(tree.words))
            self.encoded_ft.append(self.ft_lexer.encode(tree.words))
            self.encoded_chars.append(self.chars_lexer.encode(tree.words))
            tag_idxes = [
                self.tagtoi.get(tag, self.tagtoi[self.UNK_WORD])
                for tag in tree.pos_tags
            ]
            tag_idxes[0] = self.LABEL_PADDING
            self.tags.append(tag_idxes)
            heads = tree.heads
            heads[0] = self.LABEL_PADDING
            self.heads.append(heads)
            labels = [self.labtoi.get(lab, 0) for lab in tree.deprels]
            labels[0] = self.LABEL_PADDING
            self.labels.append(labels)

    def make_batches(
        self,
        batch_size: int,
        shuffle_batches: bool = False,
        shuffle_data: bool = True,
        order_by_length: bool = False,
    ) -> Iterable[DependencyBatch]:
        N = len(self.treelist)
        order = list(range(N))
        if shuffle_data:
            shuffle(order)

        # shuffling then ordering is relevant : it change the way ties are resolved and thus batch
        # construction
        if order_by_length:
            order.sort(key=lambda i: len(self.treelist[i]))

        batch_order = list(range(0, N, batch_size))
        if shuffle_batches:
            shuffle(batch_order)

        for i in batch_order:
            batch_indices = order[i : i + batch_size]
            trees = tuple(self.treelist[j] for j in batch_indices)

            encoded_words = self.lexer.make_batch([self.encoded_words[j] for j in batch_indices])  # type: ignore
            chars = self.chars_lexer.make_batch(
                [self.encoded_ft[j] for j in batch_indices]
            )
            subwords = self.ft_lexer.make_batch(
                [self.encoded_ft[j] for j in batch_indices]
            )

            tags = self.pad(
                [self.tags[j] for j in batch_indices], padding_value=self.LABEL_PADDING
            )
            heads = self.pad(
                [self.heads[j] for j in batch_indices], padding_value=self.LABEL_PADDING
            )
            labels = self.pad(
                [self.labels[j] for j in batch_indices],
                padding_value=self.LABEL_PADDING,
            )

            sent_lengths = torch.tensor([len(t) for t in trees])
            # NOTE: this is equivalent to and faster and clearer but less pure than
            # `torch.arange(sent_lengths.max()).unsqueeze(0).lt(sent_lengths.unsqueeze(1).logical_and(torch.arange(sent_lengths.max()).gt(0))`
            content_mask = labels.ne(self.LABEL_PADDING)

            yield DependencyBatch(
                chars=chars,
                encoded_words=encoded_words,
                heads=heads,
                labels=labels,
                content_mask=content_mask,
                sent_lengths=sent_lengths,
                subwords=subwords,
                tags=tags,
                trees=trees,
            )

    def pad(
        self, batch: List[List[int]], padding_value: Optional[int] = None
    ) -> torch.Tensor:
        if padding_value is None:
            padding_value = self.PAD_IDX
        tensorized_seqs = [torch.tensor(sent, dtype=torch.long) for sent in batch]
        return pad_sequence(
            tensorized_seqs,
            padding_value=padding_value,
            batch_first=True,
        )

    def init_labels(self, treelist: Iterable[DepGraph]):
        self.itolab = gen_labels(treelist)
        self.labtoi = {label: idx for idx, label in enumerate(self.itolab)}

    def init_tags(self, treelist: Iterable[DepGraph]):
        self.itotag = gen_tags(treelist)
        self.tagtoi = {tag: idx for idx, tag in enumerate(self.itotag)}

    def __len__(self):
        return len(self.treelist)


def gen_tags(treelist: Iterable[DepGraph]) -> List[str]:
    tagset = set([tag for tree in treelist for tag in tree.pos_tags])
    return [
        DependencyDataset.PAD_TOKEN,
        DepGraph.ROOT_TOKEN,
        DependencyDataset.UNK_WORD,
        *sorted(tagset),
    ]


def gen_labels(treelist: Iterable[DepGraph]) -> List[str]:
    labels = set([lbl for tree in treelist for lbl in tree.deprels])
    return [DependencyDataset.PAD_TOKEN, *sorted(labels)]
