Utility scripts
===============

This contains utility scripts that are not part of the parser but make its usage easier. They can
have additional dependencies that you can install using `pip install hopsparser[traintools]`

## `zenodo_upload.py`

Upload files to a Zenodo deposit

## `test_models.py`

Check against performance changes for existing models

## `train_models.py`

Training a treebank×configs matrix

`train_models.py` is a utility to train and evaluate models using several configs on several
treebanks. We use it internally to train the models we provide, and it is not much more flexible
than we need it to be for that purpose (but PR to improve that are welcome). It also has very little
error management so if any train run fails it will just hang until you SIGINT or SIGKILL it.

After installing `hopsparser[traintools]`, it can be run with

```console
python scripts/train_models.py {configs_dir} {treebanks_dir} --args "fasttext={fasttext_model}" --devices "{device1},{device2},{…}" --out-dir {out_dir}
```

For each `{config_name}.yaml` file in `{configs_dir}` and each `{treebank_name}` directory in
`{treebanks_dir}` (containing files named `/.*(train|dev|test)\.conllu/`), this will create a
`{out_dir}/{treebank_name}-{config_name}` directory containing the trained model and the parsed
train and test set. It will also create a summary of the performances of the various runs in
`{out_dir}/summary.tsv`.

Treebank-specific configs can be provided by putting them in sub-directories of `{config_dir}`:
config files in `{config_dir}/{prefix}` will only be used for treebanks with names starting with
`{prefix}`. This is useful for instance when working with UD data, where file names start with a
langcode. In that case, config files found in `{config_dir}/en` will be used for
`en_ewt-ud-{train,dev,test}` but not for `fr_gsd-ud-{train,dev,test}`.

You can also specify a number of rand seeds with `--rand-seeds seed1,seed2,…`, in which case the
summary will report descriptive statistics (mean, standard deviation…) for every configuration,
treebank and additional args combination and `{out_dir}/best` will contain the results of the best
runs.

The `--device` flag is used to specify the devices available to train on as comma-separated list.
The script runs in a rudimentary task queue which distributes the train runs among these devices: every
run waits until a device is available, then grab it, trains on it and releases it once it is done.

To make several runs happen concurrently on the same device, just specify it several times e.g.
`--devices "cuda:1,cuda:1"` will maintain two training process on the GPU with index 1. `"cpu"` is
of course an acceptable device that you can also specify several times and mix with GPU devices, so
this doesn't require access to GPUs.

For reference, we train our models using

```console
python scripts/train_models.py {repo_root}/examples/ {resource_dir}/treebanks --devices "cuda:0,cuda:1" --rand_seeds "0,1,2,3" --out-dir {output_dir}/newmodels"
```

For our contemporary French models, the whole procedure takes around 36h/seed on our machine.

Note that when running with the same output dir, the existing runs will be preserved (and not
re-run) and aggregated in the summaries, so it's easy to add more runs after the fact.
