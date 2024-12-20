import pathlib
from typing import Callable, cast

from spacy.language import Language
from spacy.tokens import Doc

from hopsparser.parser import BiAffineParser


@Language.factory("hopsparser", assigns=["token.pos", "token.head", "token.dep"])
def hopsparser(
    nlp: Language, name: str, model_path: str | pathlib.Path
) -> Callable[[Doc], Doc]:
    parser = BiAffineParser.load(model_path)

    def process_doc(doc: Doc) -> Doc:
        """Tag and parse a spacy document in place.

        This changes the `pos`, `head` and `dep` properties of tokens.
        """
        parsed = parser.parse(([token.text for token in sent] for sent in doc.sents), raw=True)
        for p, s in zip(parsed, doc.sents):
            for node, token in zip(p.nodes, s):
                token.pos_ = cast(str, node.upos)
                if node.head == 0:
                    token.head = token
                    token.dep_ = "ROOT"
                else:
                    token.head = s[cast(int, node.head) - 1]
                    token.dep_ = cast(str, node.deprel)
        return doc

    return process_doc
