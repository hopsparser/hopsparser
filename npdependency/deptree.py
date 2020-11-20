from npdependency.lexers import BertLexerBatch, BertLexerSentence
import pathlib
from random import shuffle
from typing import Iterable, List, NamedTuple, Sequence, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from npdependency import lexers


class DepGraph:

    ROOT_TOKEN = "<root>"

    def __init__(
        self, edges, wordlist=None, pos_tags=None, with_root=False, mwe_range=None
    ):

        self.gov2dep = {}
        self.has_gov = set()  # set of nodes with a governor

        for (gov, label, dep) in edges:
            self.add_arc(gov, label, dep)

        if with_root:
            self.add_root()

        if wordlist is None:
            wordlist = []
        self.words = [DepGraph.ROOT_TOKEN] + wordlist
        self.pos_tags = [DepGraph.ROOT_TOKEN] + pos_tags if pos_tags else None
        self.mwe_ranges = [] if mwe_range is None else mwe_range

    def fastcopy(self):
        """
        copy edges only not word nor tags nor mwe_ranges
        """
        edgelist = list(self.gov2dep.values())
        flatlist = [edge for sublist in edgelist for edge in sublist]
        return DepGraph(flatlist)

    def get_all_edges(self):
        """
        Returns the list of edges found in this graph
        """
        return [edge for gov in self.gov2dep for edge in self.gov2dep[gov]]

    def get_all_labels(self):
        """
        Returns the list of dependency labels found on the arcs
        """
        all_labels = []
        for gov in self.gov2dep:
            all_labels.extend([label for (gov, label, dep) in self.gov2dep[gov]])
        return all_labels

    def get_arc(self, gov, dep):
        """
        Returns the arc between gov and dep if it exists or None otherwise
        Args:
            gov (int): node idx
            dep (int): node idx
        Returns:
            A triple (gov,label,dep) or None.
        """
        if gov in self.gov2dep:
            for (_gov, deplabel, _dep) in self.gov2dep[gov]:
                if _dep == dep:
                    return (_gov, deplabel, _dep)
        return None

    def add_root(self):

        if self.gov2dep and 0 not in self.gov2dep:
            root = list(set(self.gov2dep) - self.has_gov)
            if len(root) == 1:
                self.add_arc(0, "root", root[0])
            else:
                # print(self)
                assert False  # no single root... problem.
        elif not self.gov2dep:  # single word sentence
            self.add_arc(0, "root", 1)

    def add_arc(self, gov, label, dep):
        """
        Adds an arc to the dep graph
        """
        if gov in self.gov2dep:
            self.gov2dep[gov].append((gov, label, dep))
        else:
            self.gov2dep[gov] = [(gov, label, dep)]

        self.has_gov.add(dep)

    def is_cyclic_add(self, gov, dep):
        """
        Checks if the addition of an arc from gov to dep would create
        a cycle in the dep tree
        """
        return gov in self.span(dep)

    def is_dag_add(self, gov, dep):
        """
        Checks if the addition of an arc from gov to dep would create
        a Dag
        """
        return dep in self.has_gov

    def span(self, gov):
        """
        Returns the list of nodes in the yield of this node
        the set of j such that (i -*> j).
        """
        agenda = [gov]
        closure = set([gov])
        while agenda:
            node = agenda.pop()
            succ = (
                [dep for (gov, label, dep) in self.gov2dep[node]]
                if node in self.gov2dep
                else []
            )
            agenda.extend([node for node in succ if node not in closure])
            closure.update(succ)
        return closure

    def _gap_degree(self, node):
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

    def gap_degree(self):
        """
        Returns the gap degree of a tree (suboptimal)
        """
        return max(self._gap_degree(node) for node in self.gov2dep)

    def is_projective(self):
        """
        Returns true if this tree is projective
        """
        return self.gap_degree() == 0

    @staticmethod
    def read_tree(istream):
        """
        Reads a conll tree from input stream
        """
        conll = []
        line = istream.readline()
        while istream and line.isspace():
            line = istream.readline()
        while istream and not line.strip() == "":
            if line[0] != "#":
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
                dataline[6] = 0

            if "-" in dataline[0]:
                mwe_ranges.append(dataline[0].split("-") + [dataline[1]])
                continue
            else:
                words.append(dataline[1])
                if dataline[3] not in ["-", "_"]:
                    postags.append(dataline[3])
                if dataline[6] != "0":  # do not add root immediately
                    edges.append(
                        (int(dataline[6]), dataline[7], int(dataline[0]))
                    )  # shift indexes !
        return DepGraph(
            edges, words, pos_tags=postags, with_root=True, mwe_range=mwe_ranges
        )

    def __str__(self):
        """
        Conll string for the dep tree
        """
        lines = []
        revdeps = [
            (dep, (label, gov))
            for node in self.gov2dep
            for (gov, label, dep) in self.gov2dep[node]
        ]
        revdeps = dict(revdeps)
        for node in range(1, len(self.words)):
            L = ["_"] * 10
            L[0] = str(node)
            L[1] = self.words[node]
            if self.pos_tags:
                L[3] = self.pos_tags[node]
            label, head = revdeps[node] if node in revdeps else ("root", 0)
            L[6] = str(head)
            L[7] = label
            mwe_list = [
                (left, right, word)
                for (left, right, word) in self.mwe_ranges
                if left == L[0]
            ]
            for mwe in mwe_list:
                MWE = ["_"] * 10
                MWE[0] = "-".join(mwe[:2])
                MWE[1] = mwe[2]
                lines.append("\t".join(MWE))
            lines.append("\t".join(L))
        return "\n".join(lines)

    def __len__(self):
        return len(self.words)


class DependencyBatch(NamedTuple):
    words: List[List[str]]
    mwe: List[List[str]]
    chars: Sequence[torch.Tensor]
    subwords: Sequence[torch.Tensor]
    cats: List[List[str]]
    encoded_words: Union[torch.Tensor, BertLexerBatch]
    tags: torch.Tensor
    heads: torch.Tensor
    labels: torch.Tensor


class DependencyDataset:
    """
    A representation of the DepBank for efficient processing.
    This is a sorted dataset.
    """

    PAD_IDX = 0
    PAD_TOKEN = "<pad>"
    UNK_WORD = "<unk>"

    @staticmethod
    def read_conll(filename: Union[str, pathlib.Path]) -> List[DepGraph]:
        with open(filename) as istream:
            treelist = []
            tree = DepGraph.read_tree(istream)
            while tree:
                if len(tree.words) <= 150:
                    treelist.append(tree)
                else:
                    print(
                        f"Dropped tree with length {len(tree.words)} > 150",
                    )
                tree = DepGraph.read_tree(istream)
        return treelist

    def __init__(
        self,
        treelist: List[DepGraph],
        lexer: lexers.Lexer,
        char_dataset: lexers.CharDataSet,
        ft_dataset: lexers.FastTextDataSet,
        use_labels=None,
        use_tags=None,
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
        self.words: List[List[str]] = []
        self.mwe_ranges: List[List[str]] = []
        self.cats: List[List[str]] = []
        self.encode()

    def encode(self):
        self.encoded_words, self.heads, self.labels, self.tags = [], [], [], []
        self.words, self.mwe_ranges, self.cats = [], [], []

        for tree in self.treelist:
            encoded_words = self.lexer.tokenize(tree.words)
            if tree.pos_tags:
                deptag_idxes = [
                    self.tagtoi.get(tag, self.tagtoi[DependencyDataset.UNK_WORD])
                    for tag in tree.pos_tags
                ]
            else:
                deptag_idxes = [
                    self.tagtoi[DependencyDataset.UNK_WORD] for tag in tree.words
                ]
            self.words.append(tree.words)
            self.cats.append(tree.pos_tags)
            self.tags.append(deptag_idxes)
            self.encoded_words.append(encoded_words)
            self.heads.append(self.oracle_governors(tree))
            # the get defaulting to 0 is a hack for labels not found in training set
            self.labels.append(
                [self.labtoi.get(lab, 0) for lab in self.oracle_labels(tree)]
            )
            self.mwe_ranges.append(tree.mwe_ranges)

    def save_vocab(self, filename):
        with open(filename, "w") as out:
            print(" ".join(self.itolab), file=out)
            print(" ".join(self.itotag), file=out)

    def shuffle_data(self):
        N = len(self.encoded_words)
        order = list(range(N))
        shuffle(order)
        self.encoded_words = [self.encoded_words[i] for i in order]
        self.tags = [self.tags[i] for i in order]
        self.heads = [self.heads[i] for i in order]
        self.labels = [self.labels[i] for i in order]
        self.words = [self.words[i] for i in order]
        self.cats = [self.cats[i] for i in order]
        self.mwe_ranges = [self.mwe_ranges[i] for i in order]

    def order_data(self):
        N = len(self.encoded_words)
        order = list(range(N))
        lengths = map(len, self.encoded_words)
        order = [idx for idx, L in sorted(zip(order, lengths), key=lambda x: x[1])]
        self.encoded_words = [self.encoded_words[idx] for idx in order]
        self.tags = [self.tags[idx] for idx in order]
        self.heads = [self.heads[idx] for idx in order]
        self.labels = [self.labels[idx] for idx in order]
        self.words = [self.words[idx] for idx in order]
        self.mwe_ranges = [self.mwe_ranges[idx] for idx in order]
        self.cats = [self.cats[idx] for idx in order]

    def make_batches(
        self,
        batch_size: int,
        shuffle_batches: bool = False,
        shuffle_data: bool = True,
        order_by_length: bool = False,
    ) -> Iterable[DependencyBatch]:
        if shuffle_data:
            self.shuffle_data()
        # shuffling and ordering is relevant : it change the way ties are resolved and thus batch
        # construction
        if order_by_length:
            self.order_data()

        N = len(self.encoded_words)
        batch_order = list(range(0, N, batch_size))
        if shuffle_batches:
            shuffle(batch_order)
        for i in batch_order:
            encoded_words = self.lexer.pad_batch(self.encoded_words[i : i + batch_size])  # type: ignore
            tags = self.pad(self.tags[i : i + batch_size])
            heads = self.pad(self.heads[i : i + batch_size])
            labels = self.pad(self.labels[i : i + batch_size])
            words = self.words[i : i + batch_size]
            mwe = self.mwe_ranges[i : i + batch_size]
            cats = self.cats[i : i + batch_size]
            chars = tuple(self.char_dataset.batch_chars(self.words[i : i + batch_size]))
            subwords = tuple(
                self.ft_dataset.batch_sentences(self.words[i : i + batch_size])
            )
            yield DependencyBatch(
                words=words,
                mwe=mwe,
                chars=chars,
                subwords=subwords,
                cats=cats,
                encoded_words=encoded_words,
                tags=tags,
                heads=heads,
                labels=labels,
            )

    def pad(self, batch: List[List[int]]) -> torch.Tensor:
        tensorized_seqs = [torch.tensor(sent, dtype=torch.long) for sent in batch]
        return pad_sequence(
            tensorized_seqs,
            padding_value=self.PAD_IDX,
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

    @staticmethod
    def oracle_labels(depgraph: DepGraph) -> List[str]:
        """
        Returns a list where each element list[i] is the label of
        the position of the governor of the word at position i.
        Returns:
        a tensor of size N.
        """
        N = len(depgraph)
        edges = depgraph.get_all_edges()
        rev_labels = dict([(dep, label) for (gov, label, dep) in edges])
        return [rev_labels.get(idx, DependencyDataset.PAD_TOKEN) for idx in range(N)]

    @staticmethod
    def oracle_governors(depgraph: DepGraph) -> List[int]:
        """
        Returns a list where each element list[i] is the index of
        the position of the governor of the word at position i.
        Returns:
        a tensor of size N.
        """
        N = len(depgraph)
        edges = depgraph.get_all_edges()
        rev_edges = dict([(dep, gov) for (gov, label, dep) in edges])
        return [rev_edges.get(idx, 0) for idx in range(N)]


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
