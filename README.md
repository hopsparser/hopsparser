# Non projective dependency parsing


This is a repository for non projective dependency parsing stuff.


It currently hosts a graph based parser inspired by the paper of Dozat. 
Contrary to Dozat, the parser performs its own tagging and can use several lexers such as FastText, Bert and others.
The parser assumes you have a linux/unix workstation, and a GPU with at least 12GB graphical memory. With smaller GPUs,
using a BERT lexer will become difficult.


Parsing task
-----------
The parsing task (or prediction task) assumes you have an already trained model in the directory MODEL. 
You can parse a file FILE in conll format (with empty annotations, just words) with the command:

```
python graph_parser.py  --pred_file FILE   MODEL/params.yaml
```

where `params.yaml` is the model hyperparameters file. 
An example model is stored in the `default` directory. The file `default/params.yaml` is an example of such parameter file.

Training task
------------
  








