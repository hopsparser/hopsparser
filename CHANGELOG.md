Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to
[Semantic Versioning](http://semver.org/).

## [Unreleased]

## [0.8.0] - 2025-06-04

### Fixed

- Emit a warning if an extra label only has one observed value.
- The obsolete `lr/schedule` config parameter will now correctly cause config loading to fail
  instead of silently using exponential.

### Changed

- Minimum torch version bumped to `2.0`
- The `train_models` script has been considerably changed, most notably it can now deal properly
  with treebanks with multiple dev and test sets, and models can be evaluated on treebanks they
  weren't trained on. See the scripts' readme for more details.
- **BREAKING** Default LR schedule changed to constant.
- Fully stop supporting Python 3.9 and 3.10
- **BREAKING** FastText if now supported by [FastText lite](https://pypi.org/project/fasttextlt),
  which is more sustainable in the long term, but means that we don't support training FastText
  models on the fly anymore. Users interested in training FastText models are advised to do it in
  e.g. Gensim, which allows full control over its hyperparameters (our choice was never optimal
  anyway). We also stop saving separate FastText models during training, considerably reducing the
  model size.
- **BREAKING** The evaluation command now uses a slightly different algorithm to match
  segmentations. This should not make any difference for HOPS evaluation (since it doesn't segment),
  but if you're somehow using it as an evaluation provider, your results might change in
  pathological cases. Scores on gold tokenizations should be the same.

## [0.7.1] - 2023-11-16

[Unreleased]: https://github.com/hopsparser/hopsparser/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/hopsparser/hopsparser/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/hopsparser/hopsparser/compare/v0.7.0...v0.7.1

### Fixed

- Update to pydantic 2 to fix breaking changes and pin it.

## [0.7.0] - 2023-03-09

[0.7.0]: https://github.com/hopsparser/hopsparser/compare/v0.6.0...v0.7.0

### Added

- HOPS now provides custom spaCy component to use in spaCy pipelines.
- Options for using weighted multitask losses, including the adaptative strategy used in Candito
  ([2022](https://aclanthology.org/2022.findings-acl.190)).
- HOPS will learn and predict token-level labels encoded in the MISC column (as `key=value`) if you
  give it the name of the key in the `extra_annotations` part of the config. See the example
  mDeBERTa-polyglot
  [config](https://github.com/hopsparser/hopsparser/blob/main/examples/mdeberta-polyglot.yaml) for
  an example of such a config

## [0.6.0] — 2022-07-28

[0.6.0]: https://github.com/hopsparser/hopsparser/compare/v0.5.0...v0.6.0

### Added

- `hopsparser evaluate` now accepts an optional output argument, allowing to write directly to a
  file if needed.
- [A new script](test_models.py) to help catch performances regressions on released models.

### Changed

- We now accept partially annotated CoNLL-U files as input for training: any learnable cell (UPOS,
  HEAD, DEPREL) for which the value is `_` will not contribute to the loss.

## [0.5.0] — 2022-05-13

[0.5.0]: https://github.com/hopsparser/hopsparser/compare/v0.4.2...v0.5.0

The performances of the contemporary models in this release are improved, most notably for models
not using BERT.

### Added

- The `scripts/zenodo_upload.py` script, a helper for uploading files to a Zenodo deposit.

### Changed

- The CharRNN lexer now represent words with last hidden (instead of cell) state of the LSTM and do
  not run on padding anymore.
- Minimal Pytorch version is now `1.9.0`
- Minimal Transformers version is now `4.19.0`
- Use `torch.inference_mode` instead of `toch.no_grad` over all the parser methods.
- BERT lexer batches no longer have an obsolete, always zero `word_indices` attribute
- `DependencyDataset` does not have lexicon attributes (`ito(lab|tag` and their inverse) since we
  don't need these anymore.
- The `train_model` script now skips incomplete runs with a warning.
- The `train_model` script has nicer logging, including progress bars to help keep track of the
  experiments.

### Fixed

- The first word in the word embeddings lexer vocabulary is not used as padding anymore and has a
  real embedding.
- BERT embeddings are now correctly computed with an attention mask to ignore padding.
- The root token embedding coming from BERT lexers is now an average of non-padding words'
  embeddings
- FastText embeddings are now computed by averaging over non-padding subwords' embeddings.
- In server mode, models are now correctly in eval mode and processing is done
  in `torch.inference_mode`.

## [0.4.2] — 2022-04-08

[0.4.2]: https://github.com/hopsparser/hopsparser/compare/v0.4.1...v0.4.2

### Fixed

- Model cross-device loading (e.g. loading on CPU a model trained on GPU) works now ([#65](https://github.com/hopsparser/hopsparser/issues/65))

## [0.4.1] — 2022-03-24

[0.4.1]: https://github.com/hopsparser/hopsparser/compare/v0.4.0...v0.4.1

### Changed

- Remove the dependency on `click_pathlib` ([#63](https://github.com/hopsparser/hopsparser/pull/63))

### Fixed

- Compatibility with setuptools 61 parsing of PEP 621 specs

## [0.4.0] — 2022-03-23

[0.4.0]: https://github.com/hopsparser/hopsparser/compare/v0.3.2...v0.4.0
