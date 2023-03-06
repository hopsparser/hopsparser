Getting started
===============

## Installation

The parser is meant to work with python >= 3.8. Install with pip, which should take care of all the
dependencies and install the console entry points

```sh
pip install hopsparser
```

## Inference

This assumes you have an already trained model in the directory `MODEL`. You can parse a file
`INPUT_FILE` using the following command

```sh
hopsparser parse MODEL INPUT_FILE OUTPUT_FILE
```

This results in a parsed file at `OUTPUT_FILE`. Both `INPUT_FILE` and OUTPUT_FILE` can be set to
`-` to use the standard i/o streams, which can be convenient if you want to use the parser in a
pipe.

The `INPUT_FILE` argument is supposed to be the path to a file in the
[CONLL-U](https://universaldependencies.org/format.html) format, possibly with missing columns. For
instance:

```text
1	Flaubert
2	a
3	écrit
4	Madame
5	Bovary
6	.
```

That is, we require word indexation and word forms only. 
Multi-word tokens are not taken into account by the parsing models but are preserved in the outputs.

HOPS modifies the columns `UPOS`, `HEAD` and `DEPREL`, all the other columns and tree metadata are
preserved.

Alternatively, you may add the `--raw` flag to the command above, in which case the parser expects a
pre-tokenized raw text file with one sentence per line and individual tokens separated by blanks.

Depending on the model, the parser will be more or less fast and more or less accurate. We can
however expect the parser to process several hundred sentences per second with a decent GPU. The GPU
actually used for performing computations can be specified using the `--device` command line option.

### Use as a spaCy component

HOPS is usable as a component for [spaCy pipelines](https://spacy.io/usage/processing-pipelines).

This requires to install the spaCy extra `pip install "hopsparser[spacy]"` and downloading a spaCy
model whose language matches the one of the HOPS model you want to use (if you want it to be
accurate, that is).

```python
import spacy
from hopsparser import spacy_component

nlp = spacy.load("fr_core_news_sm")
nlp.add_pipe("hopsparser", "hopsparser", config={"model_path": path/to/your/model})
doc = nlp(
    "Je reconnais l'existence du kiwi. Le petit chat est content. L'acrobate a mordu la pomme et la poussière.
)
for sent in doc.sents:
    print(sent)
    for token in sent:
        print(token.i, token.text, token.tag_, token.pos_, token.head.i, token.dep_, sep="\t")
    print("------")
```

This only changes the `pos`, `head` and `dep` properties of the tokens of the spaCy `Doc` and
respects the predicted sentence boundaries.

## Running in server mode

See [the server mode documentation](server.md).

## Pretrained models

We provide some pretrained models, see the list in [models.md](models.md).

## Training

Instead of using a pretrained model, one can train their own model. Training a model with BERT
definitely requires a GPU. Unless you have a GPU with a large amount of onboard memory, using small
batch sizes (2, 4, 8, 16, 32, 64) for training is probably a good idea. Otherwise, you are likely to
run out of memory.

Training can be performed with the following steps:

1. Create a directory OUT for storing your new model
2. Copy a config YAML file from the
   [examples](https://github.com/hopsparser/hopsparser/tree/master/examples) directory
3. Edit it according to your needs
4. Run the command:

```sh
hopsparser train CONFIG.yaml TRAIN.conllu OUTPUT_DIR --dev-file DEV.conllu --test-file TEST.conllu 
```

After some time (minutes, hours…) you are done and the model is ready to run (go back to the parsing
section). There are other options, see `hopsparser train --help`.

### Partial annotations

The CoNLL-U files used for training may include missing annotations in either of the `UPOS`, `HEAD`
and `DEPREL` columns, denoted by an underscore. In that case, the missing annotation will simply be
ignored for training. For nodes where the `HEAD` information is missing, `DEPREL` will also be
ignored.