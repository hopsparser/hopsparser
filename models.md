# Pretrained models

Sorted by language and corpus.

**Usage note**: The -camembert and -flaubert models use the eponymous embeddings and as such put a
relatively heavy load on hardware. We recommend using them on GPUs with at least 10 GiB memory. Otherwise,
running them on CPUs is still possible, albeit slow.

## French

### FTB-UD

| Model name                  | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                            |
| --------------------------- | ---------- | --------- | ----------- | ---------- | ----------------------------------- |
| UD_French-FTB-2.7-camembert | 98.46      |88.25      |98.56      |88.44      | [link][UD_French-FTB-2.7-camembert] |
| UD_French-FTB-2.7-flaubert  | 98.51      |88.27      |98.61      |88.46      | [link][UD_French-FTB-2.7-flaubert]  |
| UD_French-FTB-2.7-nobert    | 97.86      |84.28      |97.89      |83.91      | [link][UD_French-FTB-2.7-nobert]    |

[UD_French-FTB-2.7-camembert]: https://sharedocs.huma-num.fr/wl/?id=qu7IhWYrISIcQDrHUzcf774JkCXfyBI1

[UD_French-FTB-2.7-flaubert]: https://sharedocs.huma-num.fr/wl/?id=6b2tnCQf9HFTEZsdnTmnEvAbloPubjNV

[UD_French-FTB-2.7-nobert]: https://sharedocs.huma-num.fr/wl/?id=lCtafL0B6z53Rxc6MXbJM2d4kpZq0p0e

### FTB-SPMRL

| Model name          | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                    |
| ------------------- | ---------- | --------- | ----------- | ---------- | --------------------------- |
| ftb_spmrl-camembert | 98.88      |89.05      |98.74      |89.35      | [link][ftb_spmrl-camembert] |
| ftb_spmrl-flaubert  | 98.84      |89.22      |98.71      |89.51      | [link][ftb_spmrl-flaubert]  |
| ftb_spmrl-nobert    | 98.25      |85.37      |98.13      |84.57      | [link][ftb_spmrl-nobert]    |

[ftb_spmrl-camembert]: https://sharedocs.huma-num.fr/wl/?id=CkLcU78r5JZ6hDZ5OBuyUJM9ffh6ksgn

[ftb_spmrl-flaubert]: https://sharedocs.huma-num.fr/wl/?id=Hqv26oIWT4cSqSET5OTo82rxxfCTrTln

[ftb_spmrl-nobert]: https://sharedocs.huma-num.fr/wl/?id=8620wtCFnVcvmuP6i2njblOd4dXdGtmV

### GSD-UD

| Model name                  | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                            |
| --------------------------- | ---------- | --------- | ----------- | ---------- | ----------------------------------- |
| UD_French-GSD-2.7-camembert | 98.52      |95.53      |98.22      |94.06      | [link][UD_French-GSD-2.7-camembert] |
| UD_French-GSD-2.7-flaubert  | 98.51      |88.27      |98.61      |88.46      | [link][UD_French-GSD-2.7-flaubert]  |
| UD_French-GSD-2.7-nobert    | 97.79      |91.75      |97.28      |88.95      | [link][UD_French-GSD-2.7-nobert]    |

[UD_French-GSD-2.7-camembert]: https://sharedocs.huma-num.fr/wl/?id=3Ax0VXpnsmUuzqTHPnunBnUVW6AgS1rC

[UD_French-GSD-2.7-flaubert]: https://sharedocs.huma-num.fr/wl/?id=5u7UgVA9cN3GHI6VmyUTvmQI6iDyyU8S

[UD_French-GSD-2.7-nobert]: https://sharedocs.huma-num.fr/wl/?id=xTQ6Bt1EiKakjLsn9xUUe7UGDcXjeu19

### Sequoia-UD

| Model name                      | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                                |
| ------------------------------- | ---------- | --------- | ----------- | ---------- | --------------------------------------- |
| UD_French-Sequoia-2.7-camembert | 99.02      |93.26      |99.07      |93.36      | [link][UD_French-Sequoia-2.7-camembert] |
| UD_French-Sequoia-2.7-flaubert  | 99.16      |94.15      |99.33      |94.31      | [link][UD_French-Sequoia-2.7-flaubert]  |
| UD_French-Sequoia-2.7-nobert    | 96.97      |85.41      |97.26      |85.63      | [link][UD_French-Sequoia-2.7-nobert]    |

[UD_French-Sequoia-2.7-camembert]: https://sharedocs.huma-num.fr/wl/?id=GW5ue77TNS99lQAZJ8fb5ujtj1rEmBfj

[UD_French-Sequoia-2.7-flaubert]: https://sharedocs.huma-num.fr/wl/?id=z6NYjiGPVVzOEfTJQ5pQFRYfvpbzVcQq

[UD_French-Sequoia-2.7-nobert]: https://sharedocs.huma-num.fr/wl/?id=TLhIy5ShxzEOPBUd8YaftFSI99E1qxQk

### French-Spoken-UD

| Model name                     | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                               |
| ------------------------------ | ---------- | --------- | ----------- | ---------- | -------------------------------------- |
| UD_French-Spoken-2.7-camembert | 96.97      |80.38      |96.83      |79.72      | [link][UD_French-Spoken-2.7-camembert] |
| UD_French-Spoken-2.7-flaubert  | 97.06      |80.68      |96.69      |79.15      | [link][UD_French-Spoken-2.7-flaubert]  |
| UD_French-Spoken-2.7-nobert    | 93.94      |72.61      |93.42      |71.66      | [link][UD_French-Spoken-2.7-nobert]    |

[UD_French-Spoken-2.7-camembert]: https://sharedocs.huma-num.fr/wl/?id=MiCoXaMelAQEzxZGzzKTrSGmCIGfNwFd

[UD_French-Spoken-2.7-flaubert]: https://sharedocs.huma-num.fr/wl/?id=x6BswC571NYGO2760Imz4ShtgURajIua

[UD_French-Spoken-2.7-nobert]: https://sharedocs.huma-num.fr/wl/?id=2g7oP1qGb1gxH6M2fjEnRi0N6UNVBh6H

## Old French

### SRCMF-UD

⚠ These models are released as previews and have not yet been as extensively tested

| Model name                       | UPOS (dev) | LAS (dev) | UPOS (test) | LAS (test) | Download                                 |
| -------------------------------- | ---------- | --------- | ----------- | ---------- | ---------------------------------------- |
| UD_Old_French-SRCMF-flaubert+fro | 97.15      | 90.24     | 97.22       | 90.39      | [link][UD_Old_French-SRCMF-flaubert+fro] |

[UD_Old_French-SRCMF-flaubert+fro]: https://sharedocs.huma-num.fr/wl/?id=ssFXOn4ms2ZYx36Xe0FHfaXU1YKoXIA1