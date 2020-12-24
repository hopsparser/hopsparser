Utility scripts
===============

This contains utility scripts that are not part of the parser but make its usage easier.


## `train_models.py`: training a treebank×configs matrix

`train_models.py` is an utility to train and evaluate models using several configs on several
treebanks. We use it internally to train the models we provide and it is not much more flexible than
we need it to be for that purpose (but PR to improve that are welcome). It also has very little
error management so if any train run fails it will just hang until you SIGINT or SIGKILL it.

After installing npdependency, it can be run with

```console
python scripts/train_models.py {configs_dir} {treebanks_dir} --fasttext {fasttext_model} --devices "{device1},{device2},{…}" --out-dir {out_dir}
```

For each `{config_name}.yaml}` file in `{configs_dir}` and each `{treebank_name}` directory in
`{treebanks_dir}` (containing files named `(train|dev|test).conllu`), this will create a
`{out_dir}/{treebank_name}-{config_name}` directory containing the trained model and the parsed
train and test set. It will also create a summary of the performances of the various runs un
`{out_dir}/summary.tsv`.

The `--device` flag is used to specify the devices available to train on as comma-separated list.
The script runs in a rudimentary task queue which distributes the train runs among these devices: every
run waits until a device is available, then grab it, trains on it and releases it once it is done. 

To make several runs happen concurrently on the same device, just specify it several times e.g.
`--devices "cuda:1,cuda:1"` will maintain two training process on the GPU with index 1. `"cpu"` is
of course an acceptable device that you can also specify several times and mix with GPU devices, so
this doesn't require access to GPUs.
