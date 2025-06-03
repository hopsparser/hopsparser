HOPS, an honest parser of sentences
===================================

[![Latest PyPI version](https://img.shields.io/pypi/v/hopsparser.svg)](https://pypi.org/project/hopsparser)
[![Continuous integration](https://github.com/hopsparser/hopsparser/actions/workflows/ci.yml/badge.svg)](https://github.com/hopsparser/hopsparser/actions/workflows/ci.yml)

> It ain't much but it's honest work.

This is a graph-based dependency parser inspired by [Dozat and Manning
(2017)](https://nlp.stanford.edu/pubs/dozat2017deep.pdf)'s biaffine graph parser. Contrary to Dozat,
the parser performs its own tagging and can use several lexers such as FastText, BERT and others. It
has been originally designed within the [FlauBERT](https://github.com/getalp/Flaubert) initiative.

The parser comes with pretrained models ready for parsing French, but it might be trained for other
languages without difficulties.

See the [documentation](http://hopsparser.readthedocs.io) for more information.

## Citation

If you use this parser for your scientific publication, or if you find the resources in this
repository useful, please cite the following paper

```biblatex
@inproceedings{grobol:hal-03223424,
    title = {{Analyse en dépendances du français avec des plongements contextualisés}},
    author = {Grobol, Loïc and Crabbé, Benoît},
    url = {https://hal.archives-ouvertes.fr/hal-03223424},
    booktitle = {{Actes de la 28ème Conférence sur le Traitement Automatique des Langues Naturelles}},
    eventtitle = {{TALN-RÉCITAL 2021}},
    venue = {Lille, France},
    pdf = {https://hal.archives-ouvertes.fr/hal-03223424/file/HOPS_final.pdf},
    hal_id = {hal-03223424},
    hal_version = {v1},
}
```

## Development

If you want a development install (so you can modify the code locally and directly run it), you can
install it in editable mode with the tests extras after cloning the repository

```sh
git clone https://github.com/hopsparser/hopsparser
cd hopsparser
pip install -e ".[spacy,tests,traintools]"
```

In that case, you can run the smoke tests with `tox` to ensure that everything works on your end.

Note that using the editable mode requires `pip >= 21.3.1`.

## Licence

This software is released under the MIT Licence, with some files released under compatible free
licences, see [LICENCE.md](LICENCE.md) for the details.
