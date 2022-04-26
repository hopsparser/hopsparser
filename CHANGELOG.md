Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to
[Semantic Versioning](http://semver.org/).

## [Unreleased]

[Unreleased]: https://github.com/hopsparser/hopsparser/compare/v0.4.2...HEAD

### Changed

- The CharRNN lexer now represent words with last hidden (instead of cell) state of the LSTM and do
  not run on padding anymore.
- Minimal Pytorch version is now `1.9.0`
- Use `torch.inference_mode` instead of `toch.no_grad` over all the parser methods.
- BERT lexer batches no longer have an obsolete, always zero `word_indices` attribute
- `DependencyDataset` does not have lexicon attributes (`ito(lab|tag` and their inverse) since we
  don't need these anymore.
- The `train_model` script now skips incomplete runs with a warning.

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

