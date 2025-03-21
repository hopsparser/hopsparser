# Compatible with Python 3.10+, can be used either as a module or a standalone executable.
#
# Copyright 2017, 2018 Institute of Formal and Applied Linguistics (UFAL), Faculty of Mathematics
# and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of
# the MPL was not distributed with this file, You can obtain one at <http://mozilla.org/MPL/2.0/>.
#
# V2 author: L. Grobol <lgrobol@tuta.com> V2 Changelog:
# - [2025] Version 2.0: Refactoring, optimisations, typing, formatting, removal of internal unit
#       tests (now part of HOPS's test suite).
#
# V1 authors: Milan Straka, Martin Popel <surname@ufal.mff.cuni.cz>
#
# V1 Changelog:
# - [12 Apr 2018] Version 0.9: Initial release.
# - [19 Apr 2018] Version 1.0: Fix bug in MLAS (duplicate entries in functional_children). Add
#                              --counts option.
# - [02 May 2018] Version 1.1: When removing spaces to match gold and system characters, consider
#                              all Unicode characters of category Zs instead of just ASCII space.
# - [25 Jun 2018] Version 1.2: Use python3 in the she-bang (instead of python). In Python2, make the
#                              whole computation use `unicode` strings.

# Command line usage
# ------------------
# conll18_ud_eval.py [-v] gold_conllu_file system_conllu_file
#
# - if no -v is given, only the official CoNLL18 UD Shared Task evaluation metrics are printed
# - if -v is given, more metrics are printed (as precision, recall, F1 score, and in case the metric
#   is computed on aligned words also accuracy on these):
#   - Tokens: how well do the gold tokens match system tokens
#   - Sentences: how well do the gold sentences match system sentences
#   - Words: how well can the gold words be aligned to system words
#   - UPOS: using aligned words, how well does UPOS match
#   - XPOS: using aligned words, how well does XPOS match
#   - UFeats: using aligned words, how well does universal FEATS match
#   - AllTags: using aligned words, how well does UPOS+XPOS+FEATS match
#   - Lemmas: using aligned words, how well does LEMMA match
#   - UAS: using aligned words, how well does HEAD match
#   - LAS: using aligned words, how well does HEAD+DEPREL(ignoring subtypes) match
#   - CLAS: using aligned words with content DEPREL, how well does HEAD+DEPREL(ignoring subtypes)
#       match
#   - MLAS: using aligned words with content DEPREL, how well does HEAD+DEPREL(ignoring
#       subtypes)+UPOS+UFEATS+FunctionalChildren(DEPREL+UPOS+UFEATS) match
#   - BLEX: using aligned words with content DEPREL, how well does HEAD+DEPREL(ignoring
#       subtypes)+LEMMAS match
# - if -c is given, raw counts of correct/gold_total/system_total/aligned words are printed instead
#   of precision/recall/F1/AlignedAccuracy for all metrics.

# API usage
# ---------
# - load_conllu(file)
#   - loads CoNLL-U file from given file object to an internal representation
#   - the file object should return str in both Python 2 and Python 3
#   - raises UDError exception if the given file cannot be loaded
# - evaluate(gold_ud, system_ud)
#   - evaluate the given gold and system CoNLL-U files (loaded with load_conllu)
#   - raises UDError if the concatenated tokens of gold and system file do not match
#   - returns a dictionary with the metrics described above, each metric having three fields:
#     precision, recall and f1

# Description of token matching
# -----------------------------
# In order to match tokens of gold file and system file, we consider the text resulting from
# concatenation of gold tokens and text resulting from concatenation of system tokens. These texts
# should match -- if they do not, the evaluation fails.
#
# If the texts do match, every token is represented as a range in this original text, and tokens are
# equal only if their range is the same.

# Description of word matching
# ----------------------------
# When matching words of gold file and system file, we first match the tokens. The words which are
# also tokens are matched as tokens, but words in multi-word tokens have to be handled differently.
#
# To handle multi-word tokens, we start by finding "multi-word spans". Multi-word span is a span in
# the original text such that
# - it contains at least one multi-word token
# - all multi-word tokens in the span (considering both gold and system ones) are completely inside
#   the span (i.e., they do not "stick out")
# - the multi-word span is as small as possible
#
# For every multi-word span, we align the gold and system words completely inside this span using
# Longest Common Subsequence on their FORMs. The words not intersecting (even partially) any
# multi-word span are then aligned as tokens.

import argparse
from contextlib import suppress
from functools import cached_property
from itertools import takewhile
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, NamedTuple, Sequence, TypeVar


T = TypeVar("T")

# CoNLL-U column names
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

# Content and functional relations
CONTENT_DEPRELS = {
    "nsubj",
    "obj",
    "iobj",
    "csubj",
    "ccomp",
    "xcomp",
    "obl",
    "vocative",
    "expl",
    "dislocated",
    "advcl",
    "advmod",
    "discourse",
    "nmod",
    "appos",
    "nummod",
    "acl",
    "amod",
    "conj",
    "fixed",
    "flat",
    "compound",
    "list",
    "parataxis",
    "orphan",
    "goeswith",
    "reparandum",
    "root",
    "dep",
}

FUNCTIONAL_DEPRELS = {"aux", "cop", "mark", "det", "clf", "case", "cc"}

UNIVERSAL_FEATURES = {
    "PronType",
    "NumType",
    "Poss",
    "Reflex",
    "Foreign",
    "Abbr",
    "Gender",
    "Animacy",
    "Number",
    "Case",
    "Definite",
    "Degree",
    "VerbForm",
    "Mood",
    "Tense",
    "Aspect",
    "Voice",
    "Evident",
    "Polarity",
    "Person",
    "Polite",
}


# UD Error is used when raising exceptions in this module
class UDError(Exception):
    pass


# This could be `slice` maybe
class Span(NamedTuple):
    start: int
    end: int

    @cached_property
    def slice(self) -> slice:
        return slice(self.start, self.end)


@dataclass(eq=False)
class UDWord:
    # 10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
    columns: Sequence[str]
    # `is_multiword==True` means that this word is part of a multi-word token. In that case,
    # `self.span` marks the span of the whole multi-word token.
    is_multiword: bool
    # Span of this word (or MWT, see below) within ud_representation.characters.
    span: Span
    # Reference to the `UDWord` instance representing the HEAD (or `None `if root).
    parent: "UDWord | None" = None
    # List of references to `UDWord` instances representing functional-deprel children.
    functional_children: "list[UDWord]" = field(init=False, default_factory=list)

    @cached_property
    def is_content_deprel(self) -> bool:
        return self.columns[DEPREL] in CONTENT_DEPRELS

    @cached_property
    def is_functional_deprel(self) -> bool:
        return self.columns[DEPREL] in FUNCTIONAL_DEPRELS

    def __repr__(self) -> str:
        return f"UDWord(columns={self.columns}, is_multiword={self.is_multiword}, span={self.span})"


# Internal representation classes
@dataclass(eq=False)
class UDRepresentation:
    # Characters of all the tokens in the whole file.
    # Whitespace between tokens is not included.
    characters: list[str] = field(default_factory=list)
    # List of UDSpan instances with start&end indices into `characters`.
    sentences: list[Span] = field(default_factory=list)
    # List of UDSpan instances with start&end indices into `characters`.
    tokens: list[Span] = field(default_factory=list)
    # List of UDWord instances.
    words: list[UDWord] = field(default_factory=list)
    multi_words: list[Sequence[str]] = field(default_factory=list)


@dataclass(frozen=True)
class Score:
    correct: int
    gold_total: int
    system_total: int
    aligned_total: int | None = None

    @cached_property
    def precision(self) -> float:
        if self.system_total == 0:
            return 0.0
        return self.correct / self.system_total

    @cached_property
    def recall(self) -> float:
        if self.gold_total == 0:
            return 0.0
        return self.correct / self.gold_total

    @cached_property
    def f1(self) -> float:
        if self.system_total + self.gold_total == 0:
            return 0.0
        return 2 * self.correct / (self.system_total + self.gold_total)

    @cached_property
    def aligned_accuracy(self) -> float | None:
        if self.aligned_total is None:
            return None
        elif self.aligned_total == 0:
            return 0.0
        else:
            return self.correct / self.aligned_total


class AlignmentWord(NamedTuple):
    gold_word: UDWord
    system_word: UDWord


@dataclass
class Alignment:
    gold_words: Sequence[UDWord]
    system_words: Sequence[UDWord]
    matched_words: list[AlignmentWord]
    matched_words_map: dict[UDWord, UDWord] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.matched_words_map = {a.system_word: a.gold_word for a in self.matched_words}


# Would probably be even more efficient with numpy arrays
def detect_cycle(heads: Sequence[int]) -> list[int] | None:
    """This assumes that `heads[0]` is set to `0`"""
    on_stack = [True] * len(heads)
    on_stack[0] = False
    pointer = 1
    while True:
        while not on_stack[pointer]:
            pointer += 1
            if pointer == len(heads):
                return None
        current = [False] * len(heads)
        parent = pointer
        while on_stack[parent]:
            on_stack[parent] = False
            current[parent] = True
            parent = heads[parent]
        if current[parent]:
            # Found a cycle!
            cycle_start = parent
            cycle_pointer = parent
            cycle = [parent]
            while (cycle_pointer := heads[cycle_pointer]) != cycle_start:
                cycle.append(cycle_pointer)
            return cycle


def process_sentence_(sentence: Sequence[UDWord]):
    heads: list[int] = [int(word.columns[HEAD]) for word in sentence]
    # +1 because words are 1-indiced
    if incorrect_heads := [h for h in heads if h < 0 or h > len(sentence) + 1]:
        raise UDError(f"In {sentence}: HEADS '{incorrect_heads}' point outside of the sentence")
    n_roots = sum(1 for h in heads if h == 0)
    if n_roots == 0:
        raise UDError(f"Unrooted sentence: {sentence}")
    elif n_roots > 1:
        raise UDError(f"There are multiple roots in sentence {sentence}")

    if (cycle := detect_cycle([0, *heads])) is not None:
        raise UDError(f"There is a cycle in sentence {sentence}: {cycle}")
    for word, head in zip(sentence, heads, strict=True):
        word.parent = sentence[head - 1]
        if word.parent and word.is_functional_deprel:
            word.parent.functional_children.append(word)


def read_line(line: str, expected_id: str) -> Sequence[str]:
    """Split a word line into colums, normalise and validate them.

    Normalisations:

    - Remove space (Unicode Zs characters) from FORM
    - Remove non-universal features
    - Remove deprel subtypes
    """
    columns = line.split("\t")

    if len(columns) != 10:
        raise UDError(f"The CoNLL-U line does not contain 10 tab-separated columns: '{line}'")

    # TODO: validate against nested multi-words
    if "." not in columns[ID] and columns[ID].split("-")[0] != expected_id:
        raise UDError(
            f"Incorrect ID '{columns[ID]}' for '{columns[FORM]}', expected '{expected_id}'"
        )

    # Delete spaces from FORM, so gold.characters == system.characters
    # even if one of them tokenizes the space. Use any Unicode character
    # with category Zs.
    # TODO: that doesn't work for the first line of mwe (although they aren't really words anyway so)
    columns[FORM] = "".join(c for c in columns[FORM] if unicodedata.category(c) != "Zs")

    if not columns[FORM]:
        raise UDError("There is an empty FORM in the CoNLL-U file")

    columns[FEATS] = "|".join(
        sorted(
            feat
            for feat in columns[FEATS].split("|")
            if feat.split("=", 1)[0] in UNIVERSAL_FEATURES
        )
    )
    # Let's ignore language-specific deprel subtypes.
    # TODO: OR MAYBE DON'T??????
    columns[DEPREL] = columns[DEPREL].split(":")[0]
    return tuple(columns)


# Load given CoNLL-U file into internal representation
def load_conllu(file: Iterable[str]) -> UDRepresentation:
    ud = UDRepresentation()

    # Load the CoNLL-U file
    char_index, sentence_start_word, sentence_start_char = 0, None, None
    lines_itr = iter(file)
    for line in lines_itr:
        line = line.rstrip()

        # Handle sentence start boundaries
        if sentence_start_word is None:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            sentence_start_char = char_index
            sentence_start_word = len(ud.words)

        if not line:
            process_sentence_(ud.words[sentence_start_word:])
            # End the sentence
            assert sentence_start_char is not None
            ud.sentences.append(Span(sentence_start_char, char_index))
            sentence_start_char = None
            sentence_start_word = None
            continue

        # Read next token/word
        columns = read_line(line, expected_id=str(len(ud.words) - sentence_start_word + 1))

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Save token
        ud.characters.extend(columns[FORM])
        current_token_char_span = Span(char_index, char_index + len(columns[FORM]))
        ud.tokens.append(current_token_char_span)
        char_index += len(columns[FORM])

        # TODO: improve parsing here
        if m := re.match(r"(?P<start>\d+)(-(?P<end>\d+))?", columns[ID]):
            # Multi-word tokens
            if m.group("end"):
                start = int(m.group("start"))
                end = int(m.group("end"))
                ud.multi_words.append(columns)
                for _ in range(start, end + 1):
                    word_line = next(lines_itr).rstrip()
                    word_columns = read_line(
                        word_line, expected_id=str(len(ud.words) - sentence_start_word + 1)
                    )
                    # TODO: deal with empty words here
                    ud.words.append(
                        UDWord(
                            span=current_token_char_span, columns=word_columns, is_multiword=True
                        )
                    )
            # Basic tokens/words
            else:
                ud.words.append(
                    UDWord(span=current_token_char_span, columns=columns, is_multiword=False)
                )
        else:
            raise UDError(f"Cannot parse token ID '{columns[ID]}'")

    # FIXME: remove this, we are not a validator anyway
    if sentence_start_word is not None:
        raise UDError("The CoNLL-U file does not end with an empty line")

    return ud


def spans_score(gold_spans: Sequence[Span], system_spans: Sequence[Span]) -> Score:
    """Compute an accuracy score for the intersection of sorted spans sequences that might have
    duplicates."""
    # We could make this one operate on iterables by keeping track of the lengths but that's useless
    # for us here.
    correct = 0
    gold_itr = iter(gold_spans)
    system_itr = iter(system_spans)
    g = next(gold_itr)
    s = next(system_itr)
    with suppress(StopIteration):
        # This could be slightly optimized because we know that consecutive spans are of either `(n,
        # k), (k+1, ℓ)` or `(n, k), (n, k)` forms. But we don't need that extra complexity
        if g.start < s.start:
            g = next(gold_itr)
        elif s.start < g.start:
            s = next(system_itr)
        else:
            # At this point we know that `g.start == s.start`
            if g.end == s.end:
                correct += 1
            g = next(gold_itr)
            s = next(system_itr)

    return Score(
        correct=correct,
        gold_total=len(gold_spans),
        system_total=len(system_spans),
    )


def alignment_score(
    alignment: Alignment,
    key_fn: Callable[[UDWord, Callable[[UDWord | None], Any]], Any] | None = None,
    filter_fn: Callable[[UDWord], bool] | None = None,
) -> Score:
    if filter_fn is not None:
        gold = sum(1 for gold in alignment.gold_words if filter_fn(gold))
        system = sum(1 for system in alignment.system_words if filter_fn(system))
        aligned = sum(1 for word in alignment.matched_words if filter_fn(word.gold_word))
    else:
        gold = len(alignment.gold_words)
        system = len(alignment.system_words)
        aligned = len(alignment.matched_words)

    if key_fn is None:
        # Return score for whole aligned words
        return Score(correct=aligned, gold_total=gold, system_total=system)

    def gold_aligned_gold(word: UDWord | None) -> UDWord | None:
        return word

    def gold_aligned_system(word: UDWord | None) -> UDWord | Literal["NotAligned"] | None:
        return alignment.matched_words_map.get(word, "NotAligned") if word is not None else None

    correct = 0
    for words in alignment.matched_words:
        if filter_fn is None or filter_fn(words.gold_word):
            if key_fn(words.gold_word, gold_aligned_gold) == key_fn(
                words.system_word, gold_aligned_system
            ):
                correct += 1

    return Score(
        aligned_total=aligned,
        correct=correct,
        gold_total=gold,
        system_total=system,
    )


def _couple_eq(t: tuple[T, T]) -> bool:
    return t[0] == t[1]


def lcs_align(
    l1: Sequence[T], l2: Sequence[T], *, key: Callable[[T], Any] | None = None
) -> list[tuple[T, T]]:
    """Return matched elements for a longest common subsequence of `l1` and `l2`. `key` is a
    custom key function."""
    # FIXME: we could use Hirschberg or Hunt–Szymanski instead

    if key is None:
        l1_keys = l1
        l2_keys = l2
    else:
        # Precompute the keys
        l1_keys = [key(e) for e in l1]
        l2_keys = [key(e) for e in l2]

    # Fast-track exact sequence equality: this will save a lot of time for identical sequences,
    # which should be the majority of cases if the tokenizers are good (and it won't hurt if they're
    # not).

    # This could be a loop. It would be slightly slower and more legible.
    start_matches = [
        (l1[i], l2[i])
        for i, _ in enumerate(takewhile(_couple_eq, zip(l1_keys, l2_keys, strict=False)))
    ]
    if len(start_matches) == min(len(l1_keys), len(l2_keys)):
        return start_matches

    # Note that we'll need to adjust the indices at the end to take this shift into account.
    shift = len(start_matches)
    l1_keys = l1_keys[shift:]
    l2_keys = l2_keys[shift:]
    # We could check exact matches from the end too but that's annoying to write and if we don't
    # have an exact match anyway, it's probably not worth the trouble.

    # Also we know that the first items here don't match but i don't think it leads to any
    # significant optimisation.
    lcs_matrix = [[0] * (len(l2_keys) + 1) for _ in range(len(l1_keys) + 1)]
    # Indices are shifted by 1 because the LCS matrix is zero-padded (this way we don't need
    # separate computations for the 0-th row and column).
    for i, e1 in enumerate(l1_keys, start=1):
        for j, e2 in enumerate(l2_keys, start=1):
            if e1 == e2:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i - 1][j], lcs_matrix[i][j - 1])

    # Backtrack
    matches: list[tuple[int, int]] = []
    i, j = (len(l1_keys), len(l2_keys))
    while lcs_matrix[i][j] != 0:
        # Avoid references to l1 and l2 so we don't need any indice fiddling here and it
        # micro-optimizes better.
        if lcs_matrix[i][j] > lcs_matrix[i - 1][j - 1]:
            # The indices in lcs_matrix are shifted by one because of the zero-padding
            matches.append((shift + i - 1, shift + j - 1))
            i -= 1
            j -= 1
        elif lcs_matrix[i][j] < lcs_matrix[i - 1][j]:
            i -= 1
        else:
            j -= 1

    return [*start_matches, *((l1[i], l2[j]) for i, j in reversed(matches))]


def word_align_key(w: UDWord) -> str:
    return w.columns[FORM].lower()


# This could easily handle any number of sequences
def get_multiword_spans(
    gold_words: Sequence[UDWord], system_words: Sequence[UDWord]
) -> Sequence[Span]:
    multiwords_spans = sorted(w.span for w in (*gold_words, *system_words) if w.is_multiword)
    if not multiwords_spans:
        return []

    res = []
    current_span = multiwords_spans[0]
    for s in multiwords_spans[1:]:
        if s.start < current_span.end:
            if current_span.end < s.end:
                current_span = Span(current_span.start, s.end)
        else:
            res.append(current_span)
            current_span = s
    res.append(current_span)
    return res


def align_words(gold_words: Sequence[UDWord], system_words: Sequence[UDWord]) -> Alignment:
    if not gold_words or not system_words:
        return Alignment([], [], [])

    alignment = []

    multiword_spans = get_multiword_spans(gold_words, system_words)

    gold_itr = iter(gold_words)
    system_itr = iter(system_words)
    gold_w = next(gold_itr)
    system_w = next(system_itr)

    for next_multiword_span in multiword_spans:
        # The asserts are mostly to keep the typecheckers happy since they can't really reson on
        # iterable shenanigans
        assert gold_w is not None  # noqa: S101
        assert system_w is not None  # noqa: S101
        # These words are not in the span, align them normally
        while (
            gold_w.span.start < next_multiword_span.start
            or system_w.span.start < next_multiword_span.start
        ):
            if gold_w.span == system_w.span:
                alignment.append((gold_w, system_w))
                gold_w = next(gold_itr)
                system_w = next(system_itr)
            elif gold_w.span.start <= system_w.span.start:
                gold_w = next(gold_itr)
            else:
                system_w = next(system_itr)

        # Get the words to align
        gold_multiword_span: list[UDWord] = []
        while gold_w is not None and gold_w.span.end <= next_multiword_span.end:
            gold_multiword_span.append(gold_w)
            gold_w = next(gold_itr, None)

        system_multiword_span: list[UDWord] = []
        while system_w is not None and system_w.span.end <= next_multiword_span.end:
            system_multiword_span.append(system_w)
            system_w = next(system_itr, None)

        # Align
        alignment.extend(
            lcs_align(
                gold_multiword_span,
                system_multiword_span,
                key=word_align_key,
            )
        )

    # Remaining words after we consumed all the spans
    while gold_w is not None and system_w is not None:
        if gold_w.span == system_w.span:
            alignment.append((gold_w, system_w))
            gold_w = next(gold_itr, None)
            system_w = next(system_itr, None)
        elif gold_w.span.start <= system_w.span.start:
            gold_w = next(gold_itr, None)
        else:
            system_w = next(system_itr, None)

    return Alignment(
        gold_words=gold_words,
        system_words=system_words,
        matched_words=[AlignmentWord(g, s) for g, s in alignment],
    )


# Evaluate the gold and system treebanks (loaded using load_conllu).
def evaluate(gold_ud: UDRepresentation, system_ud: UDRepresentation) -> dict[str, Score]:
    # Check that the underlying character sequences do match.
    if len(gold_ud.characters) != len(system_ud.characters):
        raise UDError(
            "The concatenation of tokens in gold file and in system file differ:"
            f" gold is {len(gold_ud.characters)} characters"
            f" and system is {len(system_ud.characters)} characters"
        )
    elif (
        first_diff := next(
            (
                i
                for i, (g, s) in enumerate(
                    zip(gold_ud.characters, system_ud.characters, strict=True)
                )
                if g != s
            ),
            None,
        )
    ) is not None:
        raise UDError(
            "The concatenation of tokens in gold file and in system file differ!\n"
            + "First 20 differing characters in gold file: '{}' and system file: '{}'".format(
                "".join(gold_ud.characters[first_diff : first_diff + 20]),
                "".join(system_ud.characters[first_diff : first_diff + 20]),
            )
        )

    # Align words
    alignment = align_words(gold_ud.words, system_ud.words)

    # Compute the F1-scores
    return {
        "Tokens": spans_score(gold_ud.tokens, system_ud.tokens),
        "Sentences": spans_score(gold_ud.sentences, system_ud.sentences),
        "Words": alignment_score(alignment),
        "UPOS": alignment_score(alignment, lambda w, _: w.columns[UPOS]),
        "XPOS": alignment_score(alignment, lambda w, _: w.columns[XPOS]),
        "UFeats": alignment_score(alignment, lambda w, _: w.columns[FEATS]),
        "AllTags": alignment_score(
            alignment, lambda w, _: (w.columns[UPOS], w.columns[XPOS], w.columns[FEATS])
        ),
        "Lemmas": alignment_score(
            alignment,
            lambda w, ga: w.columns[LEMMA] if ga(w).columns[LEMMA] != "_" else "_",
        ),
        # FIXME: this collapses all roots together
        "UAS": alignment_score(alignment, lambda w, ga: ga(w.parent)),
        "LAS": alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL])),
        "CLAS": alignment_score(
            alignment,
            lambda w, ga: (ga(w.parent), w.columns[DEPREL]),
            filter_fn=lambda w: w.is_content_deprel,
        ),
        "MLAS": alignment_score(
            alignment,
            lambda w, ga: (
                ga(w.parent),
                w.columns[DEPREL],
                w.columns[UPOS],
                w.columns[FEATS],
                [
                    (ga(c), c.columns[DEPREL], c.columns[UPOS], c.columns[FEATS])
                    for c in w.functional_children
                ],
            ),
            filter_fn=lambda w: w.is_content_deprel,
        ),
        "BLEX": alignment_score(
            alignment,
            lambda w, ga: (
                ga(w.parent),
                w.columns[DEPREL],
                w.columns[LEMMA] if ga(w).columns[LEMMA] != "_" else "_",
            ),
            filter_fn=lambda w: w.is_content_deprel,
        ),
    }


def load_conllu_file(path):
    with open(path, encoding="utf8") as in_stream:
        return load_conllu(in_stream)


def evaluate_wrapper(args):
    # Load CoNLL-U files
    gold_ud = load_conllu_file(args.gold_file)
    system_ud = load_conllu_file(args.system_file)
    return evaluate(gold_ud, system_ud)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", type=str, help="Name of the CoNLL-U file with the gold data.")
    parser.add_argument(
        "system_file",
        type=str,
        help="Name of the CoNLL-U file with the predicted data.",
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_true", help="Print all metrics."
    )
    parser.add_argument(
        "--counts",
        "-c",
        default=False,
        action="store_true",
        help="Print raw counts of correct/gold/system/aligned words instead of prec/rec/F1 for all metrics.",
    )
    args = parser.parse_args()

    # Evaluate
    evaluation = evaluate_wrapper(args)

    # Print the evaluation
    if not args.verbose and not args.counts:
        print("LAS F1 Score: {:.2f}".format(100 * evaluation["LAS"].f1))
        print("MLAS Score: {:.2f}".format(100 * evaluation["MLAS"].f1))
        print("BLEX Score: {:.2f}".format(100 * evaluation["BLEX"].f1))
    else:
        if args.counts:
            print("Metric     | Correct   |      Gold | Predicted | Aligned")
        else:
            print("Metric     | Precision |    Recall |  F1 Score | AligndAcc")
        print("-----------+-----------+-----------+-----------+-----------")
        for metric in [
            "Tokens",
            "Sentences",
            "Words",
            "UPOS",
            "XPOS",
            "UFeats",
            "AllTags",
            "Lemmas",
            "UAS",
            "LAS",
            "CLAS",
            "MLAS",
            "BLEX",
        ]:
            m = evaluation[metric]
            if args.counts:
                print(
                    "{:11}|{:10} |{:10} |{:10} |{:10}".format(
                        metric,
                        m.correct,
                        m.gold_total,
                        m.system_total,
                        m.aligned_total
                        or (evaluation[metric].correct if metric == "Words" else ""),
                    )
                )
            else:
                print(
                    "{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                        metric,
                        100 * m.precision,
                        100 * m.recall,
                        100 * m.f1,
                        (
                            "{:10.2f}".format(100 * m.aligned_accuracy)
                            if m.aligned_accuracy is not None
                            else ""
                        ),
                    )
                )


if __name__ == "__main__":
    main()
