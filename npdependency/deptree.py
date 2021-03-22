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
    Type,
    TypeVar,
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


_T_DEPGRAPH = TypeVar("_T_DEPGRAPH", bound="DepGraph")


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
    def from_conllu(cls: Type[_T_DEPGRAPH], istream: Iterable[str]) -> _T_DEPGRAPH:
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

    @classmethod
    def read_conll(
        cls: Type[_T_DEPGRAPH],
        filename: Union[str, pathlib.Path, IO[str]],
        max_tree_length: Optional[int] = None,
    ) -> List[_T_DEPGRAPH]:
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
                            trees.append(cls.from_conllu(current_tree_lines))
                        else:
                            print(
                                f"Dropped tree with length {len(current_tree_lines)} > {max_tree_length}",
                            )
                        current_tree_lines = []
                else:
                    current_tree_lines.append(line)
        return trees


class EncodedTree(NamedTuple):
    words: Union[torch.Tensor, BertLexerSentence]
    subwords: torch.Tensor
    chars: torch.Tensor
    heads: torch.Tensor
    labels: torch.Tensor
    tags: torch.Tensor


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
    chars: torch.Tensor
    subwords: torch.Tensor
    encoded_words: Union[torch.Tensor, BertLexerBatch]
    tags: torch.Tensor
    heads: torch.Tensor
    labels: torch.Tensor
    sent_lengths: torch.Tensor
    content_mask: torch.Tensor

    def to(self, device: Union[str, torch.device]) -> "DependencyBatch":
        encoded_words = self.encoded_words.to(device)
        chars = self.chars.to(device)
        subwords = self.subwords.to(device)
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

    def __init__(
        self,
        treelist: List[DepGraph],
        lexer: lexers.Lexer,
        chars_lexer: lexers.CharRNNLexer,
        ft_lexer: lexers.FastTextLexer,
        use_labels: Sequence[str],
        use_tags: Sequence[str],
    ):
        self.lexer = lexer
        self.chars_lexer = chars_lexer
        self.ft_lexer = ft_lexer
        self.treelist = treelist

        self.itolab = use_labels
        self.labtoi = {label: idx for idx, label in enumerate(self.itolab)}

        self.itotag = use_tags
        self.tagtoi = {tag: idx for idx, tag in enumerate(self.itotag)}

        self.encoded_trees: List[EncodedTree] = []
        self.encode()

    # TODO: use an `encoded_tree` object instead of hardcoding the lexers here
    def encode(self):
        self.encoded_trees = []

        for tree in self.treelist:
            tag_idxes = torch.tensor(
                [
                    self.tagtoi.get(tag, self.tagtoi[self.UNK_WORD])
                    for tag in tree.pos_tags
                ],
                dtype=torch.long,
            )
            tag_idxes[0] = self.LABEL_PADDING
            heads = torch.tensor(tree.heads, dtype=torch.long)
            heads[0] = self.LABEL_PADDING
            # FIXME: should unk labels be padding?
            labels = torch.tensor(
                [self.labtoi.get(lab, self.LABEL_PADDING) for lab in tree.deprels],
                dtype=torch.long,
            )
            labels[0] = self.LABEL_PADDING
            self.encoded_trees.append(
                EncodedTree(
                    words=self.lexer.encode(tree.words),
                    subwords=self.ft_lexer.encode(tree.words),
                    chars=self.chars_lexer.encode(tree.words),
                    heads=heads,
                    labels=labels,
                    tags=tag_idxes,
                )
            )

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
            encoded_trees = [self.encoded_trees[j] for j in batch_indices]

            words = self.lexer.make_batch([tree.words for tree in encoded_trees])
            chars = self.chars_lexer.make_batch([tree.chars for tree in encoded_trees])
            subwords = self.ft_lexer.make_batch(
                [tree.subwords for tree in encoded_trees]
            )

            tags = self.pad(
                [self.encoded_trees[j].tags for j in batch_indices],
                padding_value=self.LABEL_PADDING,
            )
            heads = self.pad(
                [self.encoded_trees[j].heads for j in batch_indices],
                padding_value=self.LABEL_PADDING,
            )
            labels = self.pad(
                [self.encoded_trees[j].labels for j in batch_indices],
                padding_value=self.LABEL_PADDING,
            )

            sent_lengths = torch.tensor([len(t) for t in trees])
            # NOTE: this is equivalent to and faster and clearer but less pure than
            # `torch.arange(sent_lengths.max()).unsqueeze(0).lt(sent_lengths.unsqueeze(1).logical_and(torch.arange(sent_lengths.max()).gt(0))`
            content_mask = labels.ne(self.LABEL_PADDING)

            yield DependencyBatch(
                chars=chars,
                encoded_words=words,
                heads=heads,
                labels=labels,
                content_mask=content_mask,
                sent_lengths=sent_lengths,
                subwords=subwords,
                tags=tags,
                trees=trees,
            )

    def pad(
        self, batch: List[torch.Tensor], padding_value: Optional[int] = None
    ) -> torch.Tensor:
        if padding_value is None:
            padding_value = self.PAD_IDX
        return pad_sequence(
            batch,
            padding_value=padding_value,
            batch_first=True,
        )

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
