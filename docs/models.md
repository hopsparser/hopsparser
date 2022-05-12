# Pretrained models

Sorted by language and corpus.

**Usage note**: The -camembert and -flaubert models use the eponymous embeddings and as such put a
relatively heavy load on hardware. We recommend using them on GPUs with at least 10 GiB memory. Otherwise,
running them on CPUs is still possible, albeit slow.

## French

### FTB-UD

| Model name                   | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |             Download             |
| :--------------------------- | :--------: | :-------: | :---------: | :--------: | :------------------------------: |
| UD_French-FTB-2.9-camembert  |   98.50    |   88.37   |    98.57    |   88.50    | [link][UD_French-FTB-camembert]  |
| UD_French-FTB-2.9-flaubert   |   98.47    |   88.58   |    98.57    |   88.55    |  [link][UD_French-FTB-flaubert]  |
| UD_French-FTB-2.9-nobert-all |   98.11    |   85.35   |    98.17    |   84.79    | [link][UD_French-FTB-nobert-all] |

[UD_French-FTB-camembert]:
    https://zenodo.org/record/6525682/files/UD_French-FTB-camembert.tar.xz?download=1
[UD_French-FTB-flaubert]:
    https://zenodo.org/record/6525682/files/UD_French-FTB-flaubert.tar.xz?download=1
[UD_French-FTB-nobert-all]:
    https://zenodo.org/record/6525682/files/UD_French-FTB-nobert-all.tar.xz?download=1


### GSD-UD

| Model name                   | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |             Download             |
| :--------------------------- | :--------: | :-------: | :---------: | :--------: | :------------------------------: |
| UD_French-GSD-2.9-camembert  |   98.69    |   95.77   |    98.43    |   93.92    | [link][UD_French-GSD-camembert]  |
| UD_French-GSD-2.9-flaubert   |   98.64    |   95.72   |    98.51    |   94.19    |  [link][UD_French-GSD-flaubert]  |
| UD_French-GSD-2.9-nobert-all |   98.28    |   92.81   |    97.75    |   89.99    | [link][UD_French-GSD-nobert-all] |

[UD_French-GSD-camembert]:
    https://zenodo.org/record/6525682/files/UD_French-GSD-camembert.tar.xz?download=1
[UD_French-GSD-flaubert]:
    https://zenodo.org/record/6525682/files/UD_French-GSD-flaubert.tar.xz?download=1
[UD_French-GSD-nobert-all]:
    https://zenodo.org/record/6525682/files/UD_French-GSD-nobert-all.tar.xz?download=1

### Sequoia-UD

| Model name                       | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |               Download               |
| :------------------------------- | :--------: | :-------: | :---------: | :--------: | :----------------------------------: |
| UD_French-Sequoia-2.9-camembert  |   98.93    |   93.75   |    99.05    |   93.68    | [link][UD_French-Sequoia-camembert]  |
| UD_French-Sequoia-2.9-flaubert   |   99.19    |   94.43   |    99.33    |   94.28    |  [link][UD_French-Sequoia-flaubert]  |
| UD_French-Sequoia-2.9-nobert-all |   97.71    |   87.19   |    97.95    |   86.93    | [link][UD_French-Sequoia-nobert-all] |

[UD_French-Sequoia-camembert]:
    https://zenodo.org/record/6525682/files/UD_French-Sequoia-camembert.tar.xz?download=1
[UD_French-Sequoia-flaubert]:
    https://zenodo.org/record/6525682/files/UD_French-Sequoia-flaubert.tar.xz?download=1
[UD_French-Sequoia-nobert-all]:
    https://zenodo.org/record/6525682/files/UD_French-Sequoia-nobert-all.tar.xz?download=1

### French-Rhapsodie-UD

| Model name                         | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |                Download                |
| :--------------------------------- | :--------: | :-------: | :---------: | :--------: | :------------------------------------: |
| UD_French-Rhapsodie-2.9-camembert  |   97.20    |   83.02   |    96.76    |   82.41    | [link][UD_French-Rhapsodie-camembert]  |
| UD_French-Rhapsodie-2.9-flaubert   |   97.75    |   84.69   |    97.39    |   83.51    |  [link][UD_French-Rhapsodie-flaubert]  |
| UD_French-Rhapsodie-2.9-nobert-all |   96.39    |   78.46   |    95.32    |   76.55    | [link][UD_French-Rhapsodie-nobert-all] |

[UD_French-Rhapsodie-camembert]:
    https://zenodo.org/record/6525682/files/UD_French-Rhapsodie-camembert.tar.xz?download=1
[UD_French-Rhapsodie-flaubert]:
    https://zenodo.org/record/6525682/files/UD_French-Rhapsodie-flaubert.tar.xz?download=1
[UD_French-Rhapsodie-nobert-all]:
    https://zenodo.org/record/6525682/files/UD_French-Rhapsodie-nobert-all.tar.xz?download=1

## Old French

### SRCMF-UD

Due to changes in the parser in the meantime, the performances of these models differ from those
presented in Grobol et al. (2022).

| Model name                                          | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) |                        Download                         |
| :-------------------------------------------------- | :--------: | :-------: | :---------: | :--------: | :-----------------------------------------------------: |
| UD_Old_French-SRCMF-2.9-bertrade_base               |   97.30    |   88.52   |    97.23    |   88.88    |        [link](UD_Old_French-SRCMF-bertrade_base)        |
| UD_Old_French-SRCMF-2.9-bertrade_petit              |   96.88    |   87.23   |    96.93    |   87.87    |       [link](UD_Old_French-SRCMF-bertrade_petit)        |
| UD_Old_French-SRCMF-2.9-camembert_base+mlm-fro      |   97.78    |   90.42   |    97.63    |   91.30    |   [link](UD_Old_French-SRCMF-camembert_base+mlm-fro)    |
| UD_Old_French-SRCMF-2.9-flaubert_base_cased+mlm-fro |   97.71    |   91.12   |    97.62    |   91.07    | [link](UD_Old_French-SRCMF-flaubert_base_cased+mlm-fro) |

[UD_Old_French-SRCMF-bertrade_base]:
    https://zenodo.org/record/6542539/files/UD_Old_French-SRCMF-2.9-bertrade_base.tar.xz?download=1
[UD_Old_French-SRCMF-bertrade_petit]:
    https://zenodo.org/record/6542539/files/UD_Old_French-SRCMF-2.9-bertrade_petit.tar.xz?download=1
[UD_Old_French-SRCMF-camembert_base+mlm-fro]:
    https://zenodo.org/record/6542539/files/UD_Old_French-SRCMF-2.9-camembert_base%2Bmlm-fro.tar.xz?download=1
[UD_Old_French-SRCMF-flaubert_base_cased+mlm-fro]:
    https://zenodo.org/record/6542539/files/UD_Old_French-SRCMF-2.9-flaubert_base_cased%2Bmlm-fro.tar.xz?download=1
