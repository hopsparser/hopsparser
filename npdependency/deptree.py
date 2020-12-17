from typing_extensions import Final
from npdependency.lexers import BertLexerBatch, BertLexerSentence
import pathlib
from random import shuffle
from typing import (
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    TextIO,
    Union,
)

import torch
from torch.nn.utils.rnn import pad_sequence

from npdependency import lexers


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


class DepGraph:

    ROOT_TOKEN = "<root>"

    def __init__(
        self,
        edges: Iterable[Edge],
        wordlist: Optional[Iterable[str]] = None,
        pos_tags: Optional[Iterable[str]] = None,
        with_root: bool = False,
        mwe_ranges: Optional[Iterable[MWERange]] = None,
        metadata: Optional[Iterable[str]] = None,
    ):

        self.gov2dep: Dict[int, List[Edge]] = {}
        self.has_gov: Set[int] = set()  # set of nodes with a governor

        for e in edges:
            self.add_arc(e)

        if with_root:
            self.add_root()

        self.words = [self.ROOT_TOKEN, *(wordlist if wordlist is not None else [])]
        self.pos_tags = [
            self.ROOT_TOKEN,
            *(pos_tags if pos_tags is not None else []),
        ]
        self.mwe_ranges = [] if mwe_ranges is None else mwe_ranges
        self.metadata = [] if metadata is None else metadata

    def fastcopy(self) -> "DepGraph":
        """
        copy edges only not word nor tags nor mwe_ranges
        """
        return DepGraph(self.get_all_edges())

    def get_all_edges(self) -> List[Edge]:
        """
        Returns the list of edges found in this graph
        """
        return [edge for siblinghood in self.gov2dep.values() for edge in siblinghood]

    def get_all_labels(self) -> List[str]:
        """
        Returns the list of dependency labels found on the arcs
        """
        return [edge.label for edge in self.get_all_edges()]

    def get_arc(self, gov: int, dep: int) -> Optional[Edge]:
        """
        Returns the arc between gov and dep if it exists or None otherwise
        Args:
            gov (int): node idx
            dep (int): node idx
        Returns:
            A triple (gov,label,dep) or None.
        """
        for edge in self.gov2dep.get(gov, []):
            if edge.dep == dep:
                return edge
        return None

    def oracle_governors(self) -> List[int]:
        """
        Returns a list where each element list[i] is the index of
        the position of the governor of the word at position i.
        """
        N = len(self)
        govs = {edge.dep: edge.gov for edge in self.get_all_edges()}
        govs[0] = 0
        return [govs[idx] for idx in range(N)]

    def oracle_labels(self) -> List[str]:
        """
        Returns a list where each element list[i] is the label of
        the position of the governor of the word at position i.
        """
        N = len(self)
        labels = {edge.dep: edge.label for edge in self.get_all_edges()}
        labels[0] = "_"
        return [labels[idx] for idx in range(N)]

    def add_root(self):
        if not self.gov2dep:  # single word sentence
            self.add_arc(Edge(0, "root", 1))
        elif 0 not in self.gov2dep:
            roots = set(self.gov2dep) - self.has_gov
            if len(roots) > 1:
                raise ValueError("Malformed tree: multiple roots")
            elif len(roots) == 0:
                raise ValueError("Malformed tree: no root")
            self.add_arc(Edge(0, "root", roots.pop()))

    def add_arc(self, edge: Edge):
        """
        Adds an arc to the dep graph
        """
        self.gov2dep.setdefault(edge.gov, []).append(edge)
        self.has_gov.add(edge.dep)

    def is_cyclic_add(self, gov: int, dep: int) -> bool:
        """
        Checks if the addition of an arc from gov to dep would create
        a cycle in the dep tree
        """
        return gov in self.span(dep)

    def is_dag_add(self, gov: int, dep: int) -> bool:
        """
        Checks if the addition of an arc from gov to dep would create
        a Dag
        """
        return dep in self.has_gov

    def span(self, gov: int) -> Set[int]:
        """
        Returns the list of nodes in the yield of this node
        the set of j such that (i -*> j).
        """
        agenda = [gov]
        closure = set([gov])
        while agenda:
            node = agenda.pop()
            if node in self.gov2dep:
                succ = [edge.dep for edge in self.gov2dep[node]]
            else:
                succ = []
            agenda.extend([node for node in succ if node not in closure])
            closure.update(succ)
        return closure

    def _gap_degree(self, node: int) -> int:
        """
        Returns the gap degree of a node
        Args :
            node (int): a dep tree node
        """
        nspan = list(self.span(node))
        nspan.sort()
        gd = 0
        for idx in range(len(nspan)):
            if idx > 0:
                if nspan[idx] - nspan[idx - 1] > 1:
                    gd += 1
        return gd

    def gap_degree(self) -> int:
        """
        Returns the gap degree of a tree (suboptimal)
        """
        return max(self._gap_degree(node) for node in self.gov2dep)

    def is_projective(self) -> bool:
        """
        Returns true if this tree is projective
        """
        return self.gap_degree() == 0

    @classmethod
    def read_tree(cls, istream: TextIO) -> Optional["DepGraph"]:
        """
        Reads a conll tree from input stream
        """
        conll = []
        metadata = []
        line = istream.readline()
        while line and line.isspace():
            line = istream.readline()
        while line and line.startswith("#"):
            metadata.append(line.strip())
            line = istream.readline()
        while line and not line.isspace():
            conll.append(line.strip().split("\t"))
            line = istream.readline()
        if not conll:
            return None
        words = []
        mwe_ranges = []
        postags = []
        edges = []
        for dataline in conll:
            if len(dataline) < 10:  # pads the dataline
                dataline.extend(["-"] * (10 - len(dataline)))
                dataline[6] = "0"

            if "-" in dataline[0]:
                mwe_start, mwe_end = dataline[0].split("-")
                mwe_ranges.append(MWERange(int(mwe_start), int(mwe_end), dataline[1]))
                continue
            else:
                words.append(dataline[1])
                if dataline[3] not in ["-", "_"]:
                    postags.append(dataline[3])
                if dataline[6] != "0":  # do not add root immediately
                    # shift indexes !
                    edges.append(Edge(int(dataline[6]), dataline[7], int(dataline[0])))
        return cls(
            edges,
            words,
            pos_tags=postags,
            with_root=True,
            mwe_ranges=mwe_ranges,
            metadata=metadata,
        )

    def __str__(self):
        """
        Conll string for the dep tree
        """
        lines = self.metadata
        revdeps = {edge.dep: (edge.label, edge.gov) for edge in self.get_all_edges()}
        for node_idx, form in enumerate(self.words[1:], start=1):
            dataline = ["_"] * 10
            dataline[0] = str(node_idx)
            dataline[1] = form
            if self.pos_tags:
                dataline[3] = self.pos_tags[node_idx]
            deprel, head = revdeps.get(node_idx, ("root", 0))
            dataline[6] = str(head)
            dataline[7] = deprel
            mwe_list = [mwe for mwe in self.mwe_ranges if mwe.start == node_idx]
            for mwe in mwe_list:
                lines.append(mwe.to_conll())
            lines.append("\t".join(dataline))
        return "\n".join(lines)

    def __len__(self):
        return len(self.words)


class DependencyBatch(NamedTuple):
    trees: Sequence[DepGraph]
    chars: Sequence[torch.Tensor]
    subwords: Sequence[torch.Tensor]
    encoded_words: Union[torch.Tensor, BertLexerBatch]
    tags: torch.Tensor
    heads: torch.Tensor
    labels: torch.Tensor
    sent_lengths: torch.Tensor


class DependencyDataset:
    """
    A representation of the DepBank for efficient processing.
    This is a sorted dataset.
    """

    PAD_IDX: Final[int] = 0
    PAD_TOKEN: Final[str] = "<pad>"
    UNK_WORD: Final[str] = "<unk>"
    # Labels that are -100 are ignored in torch crossentropy
    LABEL_PADDING: Final[int] = -100

    @staticmethod
    def read_conll(
        filename: Union[str, pathlib.Path], max_tree_length: Optional[int] = None
    ) -> List[DepGraph]:
        with open(filename) as istream:
            treelist = []
            tree = DepGraph.read_tree(istream)
            while tree:
                if max_tree_length is None or len(tree.words) <= max_tree_length:
                    treelist.append(tree)
                else:
                    print(
                        f"Dropped tree with length {len(tree.words)} > {max_tree_length}",
                    )
                tree = DepGraph.read_tree(istream)
        return treelist

    def __init__(
        self,
        treelist: List[DepGraph],
        lexer: lexers.Lexer,
        char_dataset: lexers.CharDataSet,
        ft_dataset: lexers.FastTextDataSet,
        use_labels: Optional[List[str]] = None,
        use_tags: Optional[List[str]] = None,
    ):
        self.lexer = lexer
        self.char_dataset = char_dataset
        self.ft_dataset = ft_dataset
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
        self.encoded_words: List[Union[List[int], BertLexerSentence]] = []
        self.heads: List[List[int]] = []
        self.labels: List[List[int]] = []
        self.tags: List[List[int]] = []
        self.encode()

    def encode(self):
        # NOTE: we mask the ROOT token features with -100 that will be ignored by crossentropy, it's
        # not very satisfying though, maybe hardcode it in (lab|tag)toi ?
        self.encoded_words, self.heads, self.labels, self.tags = [], [], [], []

        for tree in self.treelist:
            encoded_words = self.lexer.tokenize(tree.words)
            if tree.pos_tags:
                deptag_idxes = [
                    self.tagtoi.get(tag, self.tagtoi[self.UNK_WORD])
                    for tag in tree.pos_tags
                ]
            else:
                deptag_idxes = [self.tagtoi[self.UNK_WORD] for _ in tree.words]
            deptag_idxes[0] = -100
            self.tags.append(deptag_idxes)
            self.encoded_words.append(encoded_words)
            heads = tree.oracle_governors()
            heads[0] = -100
            self.heads.append(heads)
            labels = [self.labtoi.get(lab, 0) for lab in tree.oracle_labels()]
            labels[0] = -100
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
            trees = [self.treelist[j] for j in batch_indices]

            chars = tuple(self.char_dataset.batch_chars([t.words for t in trees]))
            encoded_words = self.lexer.pad_batch([self.encoded_words[j] for j in batch_indices])  # type: ignore
            heads = self.pad(
                [self.heads[j] for j in batch_indices], padding_value=self.LABEL_PADDING
            )
            labels = self.pad(
                [self.labels[j] for j in batch_indices],
                padding_value=self.LABEL_PADDING,
            )
            sent_lengths = torch.tensor([len(t) for t in trees])
            subwords = tuple(self.ft_dataset.batch_sentences([t.words for t in trees]))
            tags = self.pad(
                [self.tags[j] for j in batch_indices], padding_value=self.LABEL_PADDING
            )

            yield DependencyBatch(
                chars=chars,
                encoded_words=encoded_words,
                heads=heads,
                labels=labels,
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
    labels = set(
        [lbl for tree in treelist for (_gov, lbl, _dep) in tree.get_all_edges()]
    )
    return [DependencyDataset.PAD_TOKEN, *sorted(labels)]
