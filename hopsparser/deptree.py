import collections.abc
import itertools
import re
from dataclasses import dataclass
from typing import (
    NamedTuple,
    cast,
)
from collections.abc import Iterable, Mapping, Sequence

from loguru import logger
from typing import Self


# For use with strip, this is not a list but a concat string, list from
# <https://docs.python.org/3/library/stdtypes.html#str.splitlines>. Interestingly the file, group
# and record separator don't have the Unicode BK Line Break property
# <https://www.unicode.org/reports/tr14>. But they get folded in because they have the bidi property
# <https://github.com/python/cpython/blob/e5237541a098e32a70fc621dee08721a72be7eb8/Tools/unicode/makeunicodedata.py#L440>.
# See <https://github.com/python/cpython/issues/66428#issuecomment-2791216135> for extra fun.

EOL_CHARS = (
    "\t"
    "\n"
    "\r"
    "\N{LINE TABULATION}"
    "\N{FORM FEED}"
    "\N{FILE SEPARATOR}"
    "\N{GROUP SEPARATOR}"
    "\N{RECORD SEPARATOR}"
    "\N{NEXT LINE}"
    "\N{LINE SEPARATOR}"
    "\N{PARAGRAPH SEPARATOR}"
)


class Misc(collections.abc.Sequence[str]):
    def __init__(self, elements: Sequence[str] | None = None):
        if elements is None:
            elements = []
        self._lst = list(elements)
        self.mapping: Mapping[str, str] = dict()
        self._parse()

    def _parse(self):
        mapping = dict()
        for e in self._lst:
            if m := re.match("(?P<key>.+?)=(?P<value>.*)", e):
                mapping[m.group("key")] = m.group("value")
        self.mapping = mapping

    def replace(self, mapping: Mapping[str, str]) -> Self:
        new_elements = []
        modified = set()
        for e in self._lst:
            if m := re.match("(?P<key>.+?)=(?P<value>.*)", e):
                k = m.group("key")
                if (new_value := mapping.get(k)) is not None:
                    if k in modified:
                        logger.warning(
                            f"Extra annotation {k} has multiple labels, replacing only the first one."
                        )
                        new_elements.append(e)
                    else:
                        new_elements.append(f"{k}={new_value}")
                        modified.add(k)
                else:
                    new_elements.append(e)
        # Add new annotations
        for k, v in mapping.items():
            if k not in modified:
                new_elements.append(f"{k}={v}")
        return type(self)(new_elements)

    def __getitem__(self, index):
        return self._lst[index]

    def __len__(self) -> int:
        return len(self._lst)

    def __str__(self) -> str:
        return f"Misc({self._lst}, {self.mapping})"

    def __repr__(self) -> str:
        return f"Misc({self._lst})"

    def to_conllu(self) -> str:
        if not self._lst:
            return "_"
        return "|".join(self._lst)

    @classmethod
    def from_string(cls, s: str) -> Self:
        if s == "_":
            return cls([])
        return cls(s.split("|"))


class MWERange(NamedTuple):
    start: int
    end: int
    form: str
    misc: str

    def to_conll(self) -> str:
        return f"{self.start}-{self.end}\t{self.form}\t_\t_\t_\t_\t_\t_\t_\t{self.misc}"


class EmptyNode(NamedTuple):
    after_node: int
    identifier: str
    form: str
    lemma: str | None
    upos: str | None
    xpos: str | None
    feats: str | None
    deps: str | None
    misc: Misc

    def to_conll(self) -> str:
        row = [
            self.identifier,
            self.form,
            *(
                str(c) if c is not None else "_"
                for c in (
                    self.lemma,
                    self.upos,
                    self.xpos,
                    self.feats,
                    None,
                    None,
                    self.deps,
                )
            ),
            self.misc.to_conllu(),
        ]
        return "\t".join(row)


@dataclass(eq=False)
class DepNode:
    identifier: int
    form: str
    lemma: str | None
    upos: str | None
    xpos: str | None
    feats: str | None
    head: int | None
    deprel: str | None
    deps: str | None
    misc: Misc

    def to_conll(self) -> str:
        row = [
            str(self.identifier),
            self.form,
            *(
                str(c) if c is not None else "_"
                for c in (
                    self.lemma,
                    self.upos,
                    self.xpos,
                    self.feats,
                    self.head,
                    self.deprel,
                    self.deps,
                )
            ),
            self.misc.to_conllu(),
        ]
        return "\t".join(row)


class DepGraph:
    ROOT_TOKEN = "<root>"  # noqa: S105

    def __init__(
        self,
        nodes: Iterable[DepNode],
        empty_nodes: Iterable[EmptyNode] | None = None,
        mwe_ranges: Iterable[MWERange] | None = None,
        metadata: Iterable[str] | None = None,
    ):
        self.nodes = list(nodes)

        govs = {n.identifier: n.head for n in self.nodes}
        # Only do checks on completly annotated trees
        if None not in govs.values():
            if 0 not in govs.values():
                raise ValueError("Malformed tree: no root")
            if (
                unreachable_heads := set(govs.values())
                .difference(govs.keys())
                .difference((0, None))
            ):
                raise ValueError(f"Malformed tree: unreachable heads: {unreachable_heads}")

        self.mwe_ranges = [] if mwe_ranges is None else list(mwe_ranges)
        self.empty_nodes = [] if empty_nodes is None else list(empty_nodes)
        self.metadata = [] if metadata is None else list(metadata)

    @property
    def words(self) -> list[str]:
        """
        A list where each element list[i] is the form of the word at position i.
        """
        return [self.ROOT_TOKEN, *(n.form for n in self.nodes)]

    @property
    def pos_tags(self) -> list[str | None]:
        """A list where each element list[i] is the upos of the word at position i."""
        return [None, *(n.upos for n in self.nodes)]

    @property
    def heads(self) -> list[int | None]:
        """A list where each element list[i] is the index of the position of the governor of the
        word at position i."""
        return [None, *(n.head for n in self.nodes)]

    @property
    def deprels(self) -> list[str | None]:
        """A list where each element list[i] is the dependency label of the word at position i."""
        return [None, *(n.deprel for n in self.nodes)]

    def replace(
        self,
        heads: Mapping[int, int] | None = None,
        deprels: Mapping[int, str] | None = None,
        pos_tags: Mapping[int, str] | None = None,
        misc: Mapping[int, Mapping[str, str]] | None = None,
    ) -> "DepGraph":
        """Return a new `DepGraph`, identical to `self` except for its dependencies, pos tags and
        misc annotations (if specified). All parameters should be dicts mapping node identifiers
        to the new value of the corresponding feature.

        If no argument is provided, this returns a shallow copy of `self`.
        """
        if heads is None:
            heads = dict()

        if deprels is None:
            deprels = dict()

        if pos_tags is None:
            pos_tags = dict()

        if misc is None:
            misc = dict()

        new_nodes = [
            DepNode(
                identifier=node.identifier,
                form=node.form,
                lemma=node.lemma,
                upos=pos_tags.get(node.identifier, node.upos),
                xpos=node.xpos,
                feats=node.feats,
                head=heads.get(node.identifier, node.head),
                deprel=deprels.get(node.identifier, node.deprel),
                deps=node.deps,
                misc=node.misc.replace(misc.get(node.identifier, dict())),
            )
            for node in self.nodes
        ]
        return type(self)(
            empty_nodes=self.empty_nodes[:],
            nodes=new_nodes,
            metadata=self.metadata[:],
            mwe_ranges=self.mwe_ranges[:],
        )

    @classmethod
    def from_conllu(cls, istream: Iterable[str]) -> Self:
        """Read a conll tree from an input stream"""

        metadata: list[str] = []
        mwe_ranges = []
        empty_nodes = []
        nodes = []
        # FIXME: this is clunky, maybe write a better parser that does validation?
        for line in istream:
            if line.startswith("#"):
                # FIXME: this also appends comments in the middle of the tree as metadata
                metadata.append(line.rstrip(EOL_CHARS))
                continue

            row = line.rstrip(EOL_CHARS).split("\t")

            processed_row: list[str | None]
            if "-" in row[0]:
                mwe_start, mwe_end = row[0].split("-")
                mwe_ranges.append(MWERange(int(mwe_start), int(mwe_end), row[1], row[9]))
                continue
            if len(row) < 2:
                raise ValueError("Too few columns to build a DepNode")
            elif len(row) < 10:
                processed_row = [*row, *("_" for _ in range(10 - len(row)))]
            else:
                processed_row = list(row)
            processed_row[2:9] = [c if c != "_" else None for c in processed_row[2:9]]

            if "." in row[0]:
                if processed_row[6] is not None:
                    raise ValueError("Empty tokens can't have a head")
                if processed_row[7] is not None:
                    raise ValueError("Empty tokens can't have a deprel")
                empty_nodes.append(
                    EmptyNode(
                        after_node=int(cast(str, processed_row[0]).split(".", maxsplit=1)[0]),
                        identifier=cast(str, processed_row[0]),
                        form=cast(str, processed_row[1]),
                        lemma=processed_row[2],
                        upos=processed_row[3],
                        xpos=processed_row[4],
                        feats=processed_row[5],
                        deps=processed_row[8],
                        misc=Misc.from_string(cast(str, processed_row[9])),
                    )
                )
                continue
            node = DepNode(
                identifier=int(cast(str, processed_row[0])),
                form=cast(str, processed_row[1]),
                lemma=processed_row[2],
                upos=processed_row[3],
                xpos=processed_row[4],
                feats=processed_row[5],
                head=int(processed_row[6]) if processed_row[6] is not None else None,
                deprel=processed_row[7],
                deps=processed_row[8],
                misc=Misc.from_string(cast(str, processed_row[9])),
            )
            if node.head is None and node.deprel is not None:
                logger.warning(f"Node with empty head and nonempty deprel: {node}")
            nodes.append(node)
        return cls(
            empty_nodes=empty_nodes,
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
            empty_nodes_list = sorted(
                (
                    empty_node
                    for empty_node in self.empty_nodes
                    if empty_node.after_node == n.identifier
                ),
                key=lambda x: int(x.identifier.rsplit(".", maxsplit=1)[1]),
            )
            for empty_node in empty_nodes_list:
                lines.append(empty_node.to_conll())
        return "\n".join(lines)

    def __str__(self):
        return self.to_conllu()

    def __len__(self):
        return len(self.words)

    @classmethod
    def read_conll(
        cls,
        lines: Iterable[str],
        max_tree_length: int | None = None,
    ) -> Iterable[Self]:
        current_tree_lines: list[str] = []
        # Add a dummy empty line to flush the last tree even if the CoNLL-U mandatory empty last
        # line is absent
        for line in itertools.chain(lines, [""]):
            if not line or line.isspace():
                if current_tree_lines:
                    if max_tree_length is None or len(current_tree_lines) <= max_tree_length:
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
        cls,
        words: Iterable[str],
    ) -> Self:
        return cls(
            nodes=[
                DepNode(
                    identifier=i,
                    form=w,
                    lemma=None,
                    upos=None,
                    xpos=None,
                    feats=None,
                    head=None,
                    deprel=None,
                    deps=None,
                    misc=Misc(),
                )
                for i, w in enumerate(words, start=1)
            ],
        )
