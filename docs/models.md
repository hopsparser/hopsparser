# Pretrained models

Sorted by language and corpus.

**Usage note**: The -camembert and -flaubert models use the eponymous embeddings and as such put a
relatively heavy load on hardware. We recommend using them on GPUs with at least 10 GiB memory. Otherwise,
running them on CPUs is still possible, albeit slow.

## French

### FTB-UD

| Model name                  | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |             Download             |
| :-------------------------- | :--------: | :-------: | :---------: | :--------: | :------------------------------: |
| UD_French-FTB-2.9-camembert |   98.46    |   88.33   |    98.57    |   88.51    | [link][UD_French-FTB-camembert]  |
| UD_French-FTB-2.9-flaubert  |   98.46    |   88.55   |    98.59    |   88.59    |  [link][UD_French-FTB-flaubert]  |
| UD_French-FTB-2.9-mdeberta  |   98.45    |   88.19   |    98.55    |   88.14    |  [link][UD_French-FTB-mdeberta]  |
| UD_French-FTB-2.9-nobert    |   97.86    |   84.59   |    97.88    |   83.91    | [link][UD_French-FTB-nobert-all] |

[UD_French-FTB-camembert]:
    https://zenodo.org/record/6329736/files/UD_French-FTB-camembert.tar.xz?download=1
[UD_French-FTB-flaubert]:
    https://zenodo.org/record/6329736/files/UD_French-FTB-flaubert.tar.xz?download=1
[UD_French-FTB-mdeberta]:
    https://zenodo.org/record/6329736/files/UD_French-FTB-mdeberta.tar.xz?download=1
[UD_French-FTB-nobert-all]:
    https://zenodo.org/record/6329736/files/UD_French-FTB-nobert-all.tar.xz?download=1


### GSD-UD

| Model name                  | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |             Download             |
| :-------------------------- | :--------: | :-------: | :---------: | :--------: | :------------------------------: |
| UD_French-GSD-2.9-camembert |   98.71    |   95.82   |    98.37    |   93.84    | [link][UD_French-GSD-camembert]  |
| UD_French-GSD-2.9-flaubert  |   98.65    |   95.72   |    98.61    |   94.19    |  [link][UD_French-GSD-flaubert]  |
| UD_French-GSD-2.9-mdeberta  |   98.61    |   95.39   |    98.41    |   93.84    |  [link][UD_French-GSD-mdeberta]  |
| UD_French-GSD-2.9-nobert    |   97.72    |   91.71   |    97.26    |   88.43    | [link][UD_French-GSD-nobert-all] |

[UD_French-GSD-camembert]:
    https://zenodo.org/record/6329736/files/UD_French-GSD-camembert.tar.xz?download=1
[UD_French-GSD-flaubert]:
    https://zenodo.org/record/6329736/files/UD_French-GSD-flaubert.tar.xz?download=1
[UD_French-GSD-mdeberta]:
    https://zenodo.org/record/6329736/files/UD_French-GSD-mdeberta.tar.xz?download=1
[UD_French-GSD-nobert-all]:
    https://zenodo.org/record/6329736/files/UD_French-GSD-nobert-all.tar.xz?download=1

### Sequoia-UD

| Model name                      | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |               Download               |
| :------------------------------ | :--------: | :-------: | :---------: | :--------: | :----------------------------------: |
| UD_French-Sequoia-2.9-camembert |   98.99    |   93.79   |    99.14    |   93.90    | [link][UD_French-Sequoia-camembert]  |
| UD_French-Sequoia-2.9-flaubert  |   99.17    |   94.46   |    99.34    |   94.42    |  [link][UD_French-Sequoia-flaubert]  |
| UD_French-Sequoia-2.9-mdeberta  |   99.11    |   93.75   |    99.11    |   93.59    |  [link][UD_French-Sequoia-mdeberta]  |
| UD_French-Sequoia-2.9-nobert    |   96.54    |   84.61   |    96.85    |   83.97    | [link][UD_French-Sequoia-nobert-all] |

[UD_French-Sequoia-camembert]:
    https://zenodo.org/record/6329736/files/UD_French-Sequoia-camembert.tar.xz?download=1
[UD_French-Sequoia-flaubert]:
    https://zenodo.org/record/6329736/files/UD_French-Sequoia-flaubert.tar.xz?download=1
[UD_French-Sequoia-mdeberta]:
    https://zenodo.org/record/6329736/files/UD_French-Sequoia-mdeberta.tar.xz?download=1
[UD_French-Sequoia-nobert-all]:
    https://zenodo.org/record/6329736/files/UD_French-Sequoia-nobert-all.tar.xz?download=1

### French-Rhapsodie-UD

| Model name                        | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |                Download                |
| :-------------------------------- | :--------: | :-------: | :---------: | :--------: | :------------------------------------: |
| UD_French-Rhapsodie-2.9-camembert |   96.18    |   83.25   |    94.89    |   81.91    | [link][UD_French-Rhapsodie-camembert]  |
| UD_French-Rhapsodie-2.9-flaubert  |   97.98    |   84.51   |    97.56    |   83.90    |  [link][UD_French-Rhapsodie-flaubert]  |
| UD_French-Rhapsodie-2.9-mdeberta  |   98.05    |   84.29   |    97.70    |   83.04    |  [link][UD_French-Rhapsodie-mdeberta]  |
| UD_French-Rhapsodie-2.9-nobert    |   95.32    |   75.57   |    94.33    |   73.66    | [link][UD_French-Rhapsodie-nobert-all] |

[UD_French-Rhapsodie-camembert]:
    https://zenodo.org/record/6329736/files/UD_French-Rhapsodie-camembert.tar.xz?download=1
[UD_French-Rhapsodie-flaubert]:
    https://zenodo.org/record/6329736/files/UD_French-Rhapsodie-flaubert.tar.xz?download=1
[UD_French-Rhapsodie-mdeberta]:
    https://zenodo.org/record/6329736/files/UD_French-Rhapsodie-mdeberta.tar.xz?download=1
[UD_French-Rhapsodie-nobert-all]:
    https://zenodo.org/record/6329736/files/UD_French-Rhapsodie-nobert-all.tar.xz?download=1

## Old French

### SRCMF-UD

⚠ These models are released as previews and have not yet been as extensively tested

| Model name                           | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |                 Download                 |
| :----------------------------------- | :--------: | :-------: | :---------: | :--------: | :--------------------------------------: |
| UD_Old_French-SRCMF-2.9-flaubert+fro |   97.78    |   91.11   |    97.69    |   91.22    | [link][UD_Old_French-SRCMF-flaubert+fro] |

[UD_Old_French-SRCMF-flaubert+fro]:
    https://sharedocs.huma-num.fr/wl/?id=XHczUD5wUPDhhdRlm0cXn5Pm8s6djBFF&fmode=download