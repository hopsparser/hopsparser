# Pretrained models

Sorted by language and corpus.

**Usage note**: The -camembert and -flaubert models use the eponymous embeddings and as such put a
relatively heavy load on hardware. We recommend using them on GPUs with at least 10 GiB memory. Otherwise,
running them on CPUs is still possible, albeit slow.

## French

### FTB-UD

| Model name                   | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |             Download             |
| :--------------------------- | :--------: | :-------: | :---------: | :--------: | :------------------------------: |
| UD_French-FTB-2.9-camembert  |   98.43    |   88.24   |    98.57    |   88.50    | [link][UD_French-FTB-camembert]  |
| UD_French-FTB-2.9-flaubert   |   98.46    |   88.46   |    98.55    |   88.60    |  [link][UD_French-FTB-flaubert]  |

[UD_French-FTB-camembert]:
    https://zenodo.org/record/7703346/files/UD_French-FTB-camembert.tar.xz?download=1
[UD_French-FTB-flaubert]:
    https://zenodo.org/record/7703346/files/UD_French-FTB-flaubert.tar.xz?download=1

### GSD-UD

| Model name                   | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |             Download             |
| :--------------------------- | :--------: | :-------: | :---------: | :--------: | :------------------------------: |
| UD_French-GSD-2.9-camembert  |   98.68    |   95.64   |    98.32    |   94.13   | [link][UD_French-GSD-camembert]  |
| UD_French-GSD-2.9-flaubert   |   98.73    |   95.72   |    98.59    |   94.26    |  [link][UD_French-GSD-flaubert]  |
| UD_French-GSD-2.9-nobert-all |   98.14    |   92.70   |    97.89    |   90.48    | [link][UD_French-GSD-nobert-all] |

[UD_French-GSD-camembert]:
    https://zenodo.org/record/7703346/files/UD_French-GSD-camembert.tar.xz?download=1
[UD_French-GSD-flaubert]:
    https://zenodo.org/record/7703346/files/UD_French-GSD-flaubert.tar.xz?download=1
[UD_French-GSD-nobert-all]:
    https://zenodo.org/record/7703346/files/UD_French-GSD-nobert-all.tar.xz?download=1

### Sequoia-UD

| Model name                       | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |               Download               |
| :------------------------------- | :--------: | :-------: | :---------: | :--------: | :----------------------------------: |
| UD_French-Sequoia-2.9-camembert  |   99.07    |   93.43   |    99.15    |   93.90    | [link][UD_French-Sequoia-camembert]  |
| UD_French-Sequoia-2.9-flaubert   |   99.13    |   94.42   |    99.31    |   94.78    |  [link][UD_French-Sequoia-flaubert]  |
| UD_French-Sequoia-2.9-nobert-all |   97.69    |   87.27   |    97.90    |   87.40    | [link][UD_French-Sequoia-nobert-all] |

[UD_French-Sequoia-camembert]:
    https://zenodo.org/record/7703346/files/UD_French-Sequoia-camembert.tar.xz?download=1
[UD_French-Sequoia-flaubert]:
    https://zenodo.org/record/7703346/files/UD_French-Sequoia-flaubert.tar.xz?download=1
[UD_French-Sequoia-nobert-all]:
    https://zenodo.org/record/7703346/files/UD_French-Sequoia-nobert-all.tar.xz?download=1

### French-spoken-UD

| Model name                         | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |                Download                |
| :--------------------------------- | :--------: | :-------: | :---------: | :--------: | :------------------------------------: |
| UD_French-spoken-2.9-camembert  |   98.03    |   84.07   |    96.85    |   80.33    | [link][UD_French-spoken-camembert]  |
| UD_French-spoken-2.9-flaubert   |   98.20    |   84.54   |    97.05    |   80.59    |  [link][UD_French-spoken-flaubert]  |
| UD_French-Rhapsodie-2.9-nobert-all |   96.89    |   80.11   |    96.01    |   75.12    | [link][UD_French-spoken-nobert-all] |

[UD_French-spoken-camembert]:
    https://zenodo.org/record/7703346/files/UD_all_spoken_French-camembert.tar.xz?download=1
[UD_French-spoken-flaubert]:
    https://zenodo.spokenrg/record/7703346/files/UD_all_spoken_French-flaubert.tar.xz?download=1
[UD_French-spoken-nobert-all]:
    https://zenodo.org/record/7703346/files/UD_all_spoken_French-nobert-all.tar.xz?download=1

## Old French

### SRCMF-UD

Due to changes in the parser in the meantime, the performances of these models differ from those
presented in Grobol et al. (2022).

| Model name                                          | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |                        Download                         |
| :-------------------------------------------------- | :--------: | :-------: | :---------: | :--------: | :-----------------------------------------------------: |
| UD_Old_French-SRCMF-2.9-bertrade_base               |   97.29    |   88.35   |    97.33    |   88.97    |        [link][UD_Old_French-SRCMF-bertrade_base]        |
| UD_Old_French-SRCMF-2.9-camembert_base+mlm-fro      |   97.61    |   90.37   |    97.66    |   91.19    |   [link][UD_Old_French-SRCMF-camembert_base+mlm-fro]    |
| UD_Old_French-SRCMF-2.9-flaubert_base_cased+mlm-fro |   97.65    |   90.91   |    97.69    |   91.00    | [link][UD_Old_French-SRCMF-flaubert_base_cased+mlm-fro] |

If you use these models, please cite

```bibtex
@inproceedings{grobol2022BERTradeUsingContextual,
  title = {{{BERTrade}}: {{Using Contextual Embeddings}} to {{Parse Old French}}},
  booktitle = {Proceedings of the {{Thirteenth Language Resources}} and {{Evaluation Conference}}},
  author = {Grobol, Loïc and Regnault, Mathilde and Ortiz Suárez, Pedro Javier and Sagot, Benoît and Romary, Laurent and Crabbé, Benoit},
  date = {2022-06},
  pages = {1104--1113},
  publisher = {{European Language Resource Association}},
  url = {https://aclanthology.org/2022.lrec-1.119},
  eventtitle = {{{LREC}} 2022},
  langid = {english},
  venue = {Marseille, France}
}

```

[UD_Old_French-SRCMF-bertrade_base]:
    https://zenodo.org/record/7708976/files/UD_Old_French-SRCMF-2.9-bertrade_base-8192-32e_only.tar.xz?download=1
[UD_Old_French-SRCMF-camembert_base+mlm-fro]:
    https://zenodo.org/record/7708976/files/UD_Old_French-SRCMF-2.9-camembert_base%2Bmlm-fro.tar.xz?download=1
[UD_Old_French-SRCMF-flaubert_base_cased+mlm-fro]:
    https://zenodo.org/record/7708976/files/UD_Old_French-SRCMF-2.9-flaubert_base_cased%2Bmlm-fro.tar.xz?download=1
