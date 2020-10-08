# Non projective dependency parsing

This is a repository for non projective dependency parsing stuff.

It currently hosts a graph-based dependency parser inspired by the paper of
[(Dozat 2017)](https://nlp.stanford.edu/pubs/dozat2017deep.pdf). Contrary to Dozat, the parser
performs its own tagging and can use several lexers such as FastText, Bert and others. It has been
specifically designed within the [FlauBERT](https://github.com/getalp/Flaubert) initiative.

I advise to have a GPU with at least 12GB graphical memory. With smaller GPUs, using a BERT
preprocessor will become difficult. The parser comes with pretrained models ready for parsing
French, but it might be trained for other languages without difficulties.

## Installation

Install with pip, which should take care of all the dependencies and install the `graph_parser`
console entry point

```sh
pip install git+https://github.com/bencrabbe/npdependency
```

If you want a development install (so you can modify the code locally and directly run it), you can install it in editable mode after cloning the repository

```sh
git clone https://github.com/bencrabbe/npdependency
cd npdependency
pip install -e .
```

Alternatively (but not recommended), you can also clone this repo, install the dependencies listed in `setup.cfg` and call `python -m npdependency.graph_parser3` directly from the root of the repo.

## Parsing task

The parsing task (or prediction task) assumes you have an already trained model in the directory
MODEL. You can parse a file FILE in conll format (with empty annotations, just words) with the
command:

```sh
graph_parser  --pred_file FILE   MODEL/params.yaml
```

This results in a parsed file called `FILE.parsed`. The `MODEL/params.yaml` is the model
hyperparameters file. An example model is stored in the `default` directory. The file
`default/params.yaml` is an example of such parameter file.

We advise to use the `flaubert` model which is stored in the flaubert directory. Depending on the
model, the parser will be more or less fast and more or less accurate. We can however expect the
parser to process several hundred sentences per second with a decent GPU. The parameter file
provides an option for controlling the GPU actually used for performing computations.

## Pretrained models

We provide some pretrained models:

| Model name         | Language   | device  | LAS  | speed   | Comment                                 | Download link                                                                                            |
| ------------------ | ---------- | ------- | ---- | ------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| ftb_default        | French     | GPU/CPU | 85.9 | fast    | French treebank + fasttext              | [download model](http://www.linguist.univ-paris-diderot.fr/~bcrabbe/depmodels/ftb_default.tar.gz)        |
| ftb_flaubert       | French     | GPU     | 88.3 | average | FlaubertBase+French treebank + fasttext | [download model](http://www.linguist.univ-paris-diderot.fr/~bcrabbe/depmodels/ftb_flaubert.tar.gz)       |
| ftb_camembert      | French     | GPU     | 87.9 | average | camembert+French treebank + fasttext    | [download model](http://www.linguist.univ-paris-diderot.fr/~bcrabbe/depmodels/camembert.tar.gz)          |
| ud_fr_gsd_default  | French     | GPU/CPU | 90.2 | fast    | UD French GSD + fasttext                | [download model](http://www.linguist.univ-paris-diderot.fr/~bcrabbe/depmodels/fr_gsd_default.tar.gz)     |
| ud_fr_gsd_flaubert | French     | GPU     | 92.4 | average | FlaubertBase + UD French GSD + fasttext | [download model](http://www.linguist.univ-paris-diderot.fr/~bcrabbe/depmodels/ud_fr_gsd_flaubert.tar.gz) |
| ud_fro_default     | Old French | GPU/CPU | 85.9 | fast    | SRCMF treebank + fasttext               | [download model](http://www.linguist.univ-paris-diderot.fr/~bcrabbe/depmodels/ud_of_default.tar.gz)      |

The reader may notice a difference with the results published in
[(Le et al 2020)](https://arxiv.org/abs/1912.05372). The difference comes from a better usage of
fasttext and from the fact that this parser also predicts part of speech tags while the version
described in [(Le et al 2020)](https://arxiv.org/abs/1912.05372) required predicted tags as part of
its input. These changes make the parser easier to use in "real life" projects.

## Training task

Instead of using a pretrained model, one can train his own model. Training a model with BERT
definitely requires a GPU. Unless you have a GPU with a very large amount of onboard memory, I
advise to use very small batch sizes (2, 4, 8, 16, 32, 64) for training. Otherwise you are likely to
run out of memory.

Training can be performed with the following steps:

1. Create a directory MODEL for storing your new model
2. `cd` to MODEL
3. copy the `params.yaml` file from another model into MODEL
4. Edit the `params.yaml` according to your needs
5. Run the command:

```sh
graph_parser  --train_file TRAINFILE --dev_file DEVFILE  params.yaml
```

after some time (minutes,hours,days...) you are done and the model is ready to run (go back to the
parsing section)
