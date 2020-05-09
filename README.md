# Non projective dependency parsing


This is a repository for non projective dependency parsing stuff.


It currently hosts a graph based parser inspired by the paper of Dozat. 
Contrary to Dozat, the parser performs its own tagging and can use several lexers such as FastText, Bert and others.
The parser assumes you have a linux/unix workstation, and a GPU with at least 12GB graphical memory. With smaller GPUs,
using a BERT lexer will become difficult.


Parsing task
-----------
The parsing task (or prediction task) assumes you have an already trained model in the directory MODEL. 
You can parse a file in CONLL format (with empty annotations, just words)





