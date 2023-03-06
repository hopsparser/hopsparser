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