# Pretrained models

Sorted by language and corpus.

**Usage note**: The -camembert and -flaubert models use the eponymous embeddings and as such put a
relatively heavy load on hardware. We recommend using them on GPUs with at least 10 GiB memory. Otherwise,
running them on CPUs is still possible, albeit slow.

## French

### FTB-UD

| Model name                  | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                            |
| --------------------------- | ---------- | --------- | ----------- | ---------- | ----------------------------------- |
| UD_French-FTB-2.7-camembert | 98.42      | 88.23     | 98.55       | 88.38      | [link][UD_French-FTB-2.7-camembert] |
| UD_French-FTB-2.7-flaubert  | 98.51      | 88.26     | 98.58       | 88.55      | [link][UD_French-FTB-2.7-flaubert]  |
| UD_French-FTB-2.7-nobert    | 97.81      | 84.69     | 97.85       | 83.93      | [link][UD_French-FTB-2.7-nobert]    |

[UD_French-FTB-2.7-camembert]: https://sharedocs.huma-num.fr/wl/?id=d3IzkD0j011kN8yDaPkhA7P21XSGR9zE

[UD_French-FTB-2.7-flaubert]: https://sharedocs.huma-num.fr/wl/?id=LPI6M69bWEIBPhlEUdVOf8MDjotH62km

[UD_French-FTB-2.7-nobert]: https://sharedocs.huma-num.fr/wl/?id=NR27gyIoicaxxWlcj4lnvu0QpDGP84xp

### FTB-SPMRL

| Model name          | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                    |
| ------------------- | ---------- | --------- | ----------- | ---------- | --------------------------- |
| ftb_spmrl-camembert | 98.83      | 89.02     | 98.69       | 89.31      | [link][ftb_spmrl-camembert] |
| ftb_spmrl-flaubert  | 98.87      | 89.47     | 98.78       | 89.64      | [link][ftb_spmrl-flaubert]  |
| ftb_spmrl-nobert    | 98.2       | 85.2      | 98.13       | 84.49      | [link][ftb_spmrl-nobert]    |

[ftb_spmrl-camembert]: https://sharedocs.huma-num.fr/wl/?id=YmSZmlZovSRwvOCmVxlrIBTc6lDhrNeo

[ftb_spmrl-flaubert]: https://sharedocs.huma-num.fr/wl/?id=fyvPzaP2Xo8PU31gsAN0a0qLnSmjrgq3

[ftb_spmrl-nobert]: https://sharedocs.huma-num.fr/wl/?id=EM0Jm8jzPUONRyE9H4EREQ30689OxIov

### GSD-UD

| Model name                  | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                            |
| --------------------------- | ---------- | --------- | ----------- | ---------- | ----------------------------------- |
| UD_French-GSD-2.7-camembert | 98.57      | 95.55     | 98.48       | 94.04      | [link][UD_French-GSD-2.7-camembert] |
| UD_French-GSD-2.7-flaubert  | 98.57      | 95.49     | 98.53       | 94.24      | [link][UD_French-GSD-2.7-flaubert]  |
| UD_French-GSD-2.7-nobert    | 97.77      | 91.67     | 97.5        | 88.93      | [link][UD_French-GSD-2.7-nobert]    |

[UD_French-GSD-2.7-camembert]: https://sharedocs.huma-num.fr/wl/?id=1EjeZn3JCDDO6ulIgUffY5Rq3qvP7j1I

[UD_French-GSD-2.7-flaubert]: https://sharedocs.huma-num.fr/wl/?id=y0e4VHIwkLe3hggccKbMMf5Sa2TvpXxd

[UD_French-GSD-2.7-nobert]: https://sharedocs.huma-num.fr/wl/?id=nLN86K0ZwH7j3vD8K3zlqZwgKfv8fo4i

### Sequoia-UD

| Model name                      | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                                |
| ------------------------------- | ---------- | --------- | ----------- | ---------- | --------------------------------------- |
| UD_French-Sequoia-2.7-camembert | 99.01      | 93.6      | 99.23       | 93.85      | [link][UD_French-Sequoia-2.7-camembert] |
| UD_French-Sequoia-2.7-flaubert  | 99.18      | 94.13     | 99.33       | 94.71      | [link][UD_French-Sequoia-2.7-flaubert]  |
| UD_French-Sequoia-2.7-nobert    | 96.89      | 85.26     | 97.32       | 85.29      | [link][UD_French-Sequoia-2.7-nobert]    |

[UD_French-Sequoia-2.7-camembert]: https://sharedocs.huma-num.fr/wl/?id=xrGkUv2Mq4ddW4KVdKBehsq0yn3UzpWU

[UD_French-Sequoia-2.7-flaubert]: https://sharedocs.huma-num.fr/wl/?id=L2WEbxUjQhLGxJ0ABglPuK02JWUhejkV

[UD_French-Sequoia-2.7-nobert]: https://sharedocs.huma-num.fr/wl/?id=Q0hVQbtn7jK3BGZqwQbTA4HNpOrWkl5g

### French-Spoken-UD

| Model name                     | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                               |
| ------------------------------ | ---------- | --------- | ----------- | ---------- | -------------------------------------- |
| UD_French-Spoken-2.7-camembert | 96.49      | 80.82     | 96.35       | 78.63      | [link][UD_French-Spoken-2.7-camembert] |
| UD_French-Spoken-2.7-flaubert  | 97.05      | 79.84     | 96.67       | 78.72      | [link][UD_French-Spoken-2.7-flaubert]  |
| UD_French-Spoken-2.7-nobert    | 93.72      | 72.71     | 92.75       | 71.02      | [link][UD_French-Spoken-2.7-nobert]    |

[UD_French-Spoken-2.7-camembert]: https://sharedocs.huma-num.fr/wl/?id=9XuMjPlMPPzxekIid1wJTq53XUPhmkei

[UD_French-Spoken-2.7-flaubert]: https://sharedocs.huma-num.fr/wl/?id=5DfSBR6kAHrJjfVQhUbad3Y6oAvifd9n

[UD_French-Spoken-2.7-nobert]: https://sharedocs.huma-num.fr/wl/?id=5oZ0Oio6R1yO4cqsi9Dn9RsGLkVONt3m

## Old French

### SRCMF-UD

⚠ These models are released as previews and have not yet been as extensively tested

| Model name                       | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                                 |
| -------------------------------- | ---------- | --------- | ----------- | ---------- | ---------------------------------------- |
| UD_Old_French-SRCMF-flaubert+fro | 97.21      | 90.14     | 97.40       | 90.56      | [link][UD_Old_French-SRCMF-flaubert+fro] |

[UD_Old_French-SRCMF-flaubert+fro]: https://sharedocs.huma-num.fr/wl/?id=U3LQ1dGJmzJfIchpTMGyzU7TDmALQy9E
