HOPS, an honest parser of sentences
===================================

[![Latest PyPI version](https://img.shields.io/pypi/v/hopsparser.svg)](https://pypi.org/project/hopsparser)
[![Build Status](https://github.com/bencrabbe/npdependency/actions/workflows/ci.yml/badge.svg)](https://github.com/bencrabbe/npdependency/actions?query=workflow%3ACI)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is a graph-based dependency parser inspired by the paper of [(Dozat
2017)](https://nlp.stanford.edu/pubs/dozat2017deep.pdf). Contrary to Dozat, the parser performs its
own tagging and can use several lexers such as FastText, BERT and others. It has been originally
designed within the [FlauBERT](https://github.com/getalp/Flaubert) initiative.

I advise to have a GPU with at least 12GB graphical memory. With smaller GPUs, using a BERT
preprocessor will become difficult. The parser comes with pretrained models ready for parsing
French, but it might be trained for other languages without difficulties.

## Installation

The parser is meant to work with python >= 3.8. Install with pip, which should take care of all the
dependencies and install the console entry points

```sh
pip install hopsparser
```

If you want a development install (so you can modify the code locally and directly run it), you can
install it in editable mode after cloning the repository

```sh
git clone https://github.com/bencrabbe/npdependency
cd npdependency
pip install -e .
```

In that case, you can run the smoketests with `tox` to ensure that everything works on your end.

## Parsing task

The parsing task (or prediction task) assumes you have an already trained model in the directory
MODEL. You can parse a file INPUT_FILE in truncated CONLL-U format with the command:

```sh
hopsparser parse MODEL INPUT_FILE OUTPUT_FILE
```

This results in a parsed file called `OUTPUT_FILE`. Both INPUT_FILE and OUTPUT_FILE can be set to
`-` to use the standard i/o streams, which can be convenient if you want to use the parser in a
pipe.

The `INPUT_FILE` argument is supposed to be the path to a file in the
[CONLL-U](https://universaldependencies.org/format.html) format, possibly with missing columns. For
instance:

```conllu
1	Flaubert
2	a
3	écrit
4	Madame
5	Bovary
6	.
```

That is we require word indexation and word forms only. Empty words are currently not supported.
Multi-word tokens are not taken into account by the parsing models but are preserved in the outputs.

Alternatively, you may add the `--raw` flag to the command above, in which case the parser expects a
pre-tokenized raw text file with one sentence per line and individual tokens separated by blanks.

Depending on the model, the parser will be more or less fast and more or less accurate. We can
however expect the parser to process several hundred sentences per second with a decent GPU. The GPU
actually used for performing computations can be specified using the `--device` command line option.

## Running in server mode

See [the server mode documentation](https://github.com/bencrabbe/npdependency/blob/master/docs/server.md).

## Pretrained models

We provide some pretrained models, see the list in [models.md](https://github.com/bencrabbe/npdependency/blob/master/docs/models.md).

The reader may notice a difference with the results published in [(Le et al
2020)](https://arxiv.org/abs/1912.05372). The difference comes from a better usage of fasttext and
from the fact that this parser also predicts part of speech tags while the version described in [(Le
et al 2020)](https://arxiv.org/abs/1912.05372) required predicted tags as part of its input. These
changes make the parser easier to use in "real life" projects.

## Training

Instead of using a pretrained model, one can train his own model. Training a model with BERT
definitely requires a GPU. Unless you have a GPU with a very large amount of onboard memory, I
advise to use very small batch sizes (2, 4, 8, 16, 32, 64) for training. Otherwise you are likely to
run out of memory.

Training can be performed with the following steps:

1. Create a directory OUT for storing your new model
2. Copy a config YAML file from the [examples](https://github.com/bencrabbe/npdependency/tree/master/examples) directory
3. Edit it according to your needs
4. Run the command:

```sh
hopsparser train CONFIG.yaml TRAIN.conllu OUTPUT_DIR --dev-file DEV.conllu --test-file TEST.conllu 
```

After some time (minutes, hours…) you are done and the model is ready to run (go back to the parsing
section). There are other options, see `hopsparser train --help`.

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


## Licence

This software is released under the MIT Licence, with some files released under compatible free
licences, see [LICENCE.md](LICENCE.md) for the details.
