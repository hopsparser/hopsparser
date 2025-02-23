Utility scripts
===============

This contains utility scripts that are not part of the parser but make its usage easier. They can
have additional dependencies that you can install using `pip install hopsparser[traintools]`

## `get_UD_data.py`

Download Universal Dependencies treebanks and prepare them for model training.

## `zenodo_upload.py`

Upload files to a Zenodo deposit.

## `check_models.py`

Check against performance changes for existing models.

## `train_models.py`

Training a treebank×configs matrix

`train_models.py` is a utility to train and evaluate models using several configs on several
treebanks. We use it internally to train the models we provide, and it is not much more flexible
than we need it to be for that purpose (but PR to improve that are welcome). It also has very little
error management so if any train run fails it might just hang until you SIGKILL it and its children
processes

After installing `hopsparser[traintools]`, it can be run with

```console
python scripts/train_models.py {config_file} {treebanks_dir} --devices "{device1},{device2},{…}" --out-dir {out_dir}
```

There is an example of config file at </exampl.es/train_config-fr.yaml>

You can also specify a number of rand seeds with `--rand-seeds seed1,seed2,…`.

The `--device` flag is used to specify the devices available to train on as comma-separated list.
The script runs in a rudimentary task queue which distributes the train runs among these devices:
every run waits until a device is available, then grab it, trains on it and releases it once it is
done.

To make several runs happen concurrently on the same device, just specify it several times e.g.
`--devices "cuda:1,cuda:1"` will maintain two training process on the GPU with index 1. `"cpu"` is
of course an acceptable device that you can also specify several times and mix with GPU devices, so
this doesn't require access to GPUs.

For reference, we train our UD models using

```console
python scripts/train_models.py {repo_root}/examples/train_config-ud {resource_dir}/treebanks --devices "cuda:0,cuda:1" --rand_seeds "0,1,2,3" --out-dir {output_dir}/newmodels"
```

After downloading the UD treebanks using `get_UD_data.py`.

The whole procedure takes around 36h/seed on our machine.

Note that when running with the same output dir, the existing runs will be preserved (and not
re-run) and aggregated in the summaries, so it's easy to add more runs after the fact.
