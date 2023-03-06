HOPS, an honest parser of sentences
===================================

[![Latest PyPI version](https://img.shields.io/pypi/v/hopsparser.svg)](https://pypi.org/project/hopsparser)
[![Build Status](https://github.com/hopsparser/npdependency/actions/workflows/ci.yml/badge.svg)](https://github.com/hopsparser/hopsparser/actions?query=workflow%3ACI)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/hopsparser/badge/?version=stable)](https://hopsparser.readthedocs.io/en/latest/?badge=stable)

> It ain't much but it's honest work.

This is a graph-based dependency parser inspired by [Dozat and Manning
(2017)](https://nlp.stanford.edu/pubs/dozat2017deep.pdf)'s biaffine graph parser. Contrary to Dozat,
the parser performs its own tagging and can use several lexers such as FastText, BERT and others. It
has been originally designed within the [FlauBERT](https://github.com/getalp/Flaubert) initiative.

The parser comes with pretrained models ready for parsing French, but it might be trained for other
languages without difficulties.

## Getting Started

Check out [*Geting started*](get_started.md).

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