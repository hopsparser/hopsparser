# Non projective dependency parsing


This is a repository for non projective dependency parsing stuff.


It currently hosts a graph based parser inspired by the paper of Dozat. 
Contrary to Dozat, the parser performs its own tagging and can use several lexers such as FastText, Bert and others.
The parser assumes you have a linux/unix workstation, and a GPU with at least 12GB graphical memory. With smaller GPUs,
using a BERT lexer will become difficult. The parser comes with two pretrained models ready for parsing French.


Parsing task
-----------
The parsing task (or prediction task) assumes you have an already trained model in the directory MODEL. 
You can parse a file FILE in conll format (with empty annotations, just words) with the command:

```
python graph_parser.py  --pred_file FILE   MODEL/params.yaml
```

where `MODEL/params.yaml` is the model hyperparameters file. 
An example model is stored in the `default` directory. The file `default/params.yaml` is an example of such parameter file.

We advise to use the `flaubert` model which is stored in the flaubert directory.

Training task
------------

Training a model with BERT definitely requires a GPU. Unless you have a GPU with a very large amount of onboard memory, I advise to use 
very small batch sizes (3 or 4) for training. Otherwise you are likely to run out of memory.

Training can be performed with the following steps:

  1. Create a directory MODEL for storing your new model
  2. `cd` to MODEL 
  3. copy the `params.yaml` file from another model into MODEL
  4. Edit the `params.yaml` according to your needs
  5. Run the command:
```
python graph_parser.py  --train_file TRAINFILE --dev_file DEVFILE --pred_file params.yaml
```
after some time (minutes,hours,days...) you are done and the model is ready to run (go to the parsing section)

  








