import itertools
from dataclasses import dataclass
from typing import Iterable, List, NamedTuple, Optional, Type, TypeVar

from loguru import logger


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
        A list where each element list[i] is the form of the word at position i.
        """
        return [self.ROOT_TOKEN, *(n.form for n in self.nodes)]

    @property
    def pos_tags(self) -> List[str]:
        """A list where each element list[i] is the upos of the word at position i."""
        return [self.ROOT_TOKEN, *(n.upos for n in self.nodes)]

    @property
    def heads(self) -> List[int]:
        """A list where each element list[i] is the index of the position of the governor of the word at position i."""
        return [0, *(n.head for n in self.nodes)]

    @property
    def deprels(self) -> List[str]:
        """A list where each element list[i] is the dependency label of of the word at position i."""
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
        """Read a conll tree from an input stream"""
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

    def to_conllu(self) -> str:
        """CoNLL-U string for the dep tree"""
        lines = self.metadata
        for n in self.nodes:
            mwe_list = [mwe for mwe in self.mwe_ranges if mwe.start == n.identifier]
            for mwe in mwe_list:
                lines.append(mwe.to_conll())
            lines.append(n.to_conll())
        return "\n".join(lines)

    def __str__(self):
        return self.to_conllu()

    def __len__(self):
        return len(self.words)

    @classmethod
    def read_conll(
        cls: Type[_T_DEPGRAPH],
        lines: Iterable[str],
        max_tree_length: Optional[int] = None,
    ) -> Iterable[_T_DEPGRAPH]:
        current_tree_lines: List[str] = []
        # Add a dummy empty line to flush the last tree even if the CoNLL-U mandatory empty last
        # line is absent
        for line in itertools.chain(lines, [""]):
            if not line or line.isspace():
                if current_tree_lines:
                    if (
                        max_tree_length is None
                        or len(current_tree_lines) <= max_tree_length
                    ):
                        yield cls.from_conllu(current_tree_lines)
                    else:
                        logger.info(
                            f"Dropped tree with length {len(current_tree_lines)} > {max_tree_length}",
                        )
                    current_tree_lines = []
            else:
                current_tree_lines.append(line)

    @classmethod
    def from_words(
        cls: Type[_T_DEPGRAPH],
        words: Iterable[str],
    ) -> _T_DEPGRAPH:
        return cls(
            nodes=[
                DepNode(
                    identifier=i,
                    form=w,
                    lemma="_",
                    upos="_",
                    xpos="_",
                    feats="_",
                    head=0,
                    deprel="_",
                    deps="_",
                    misc="_",
                )
                for i, w in enumerate(words, start=1)
            ],
        )
