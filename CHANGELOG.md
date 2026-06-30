# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-06-30

### Added
- Added CLI tests (`test_cli.py`) achieving near 100% test coverage for `cli.py`.
- Added Map Validation tests (`test_validate.py`) to verify FSC and CCC computations.
- Added tests for `compute_ccc()` zero-variance edge cases in `test_core.py`.
- Added documentation for the `synth-cryo-em-validate` tool in `usage.md`.
- Added `__init__.py` to make the codebase a standard Python package, exposing main API functions and `__version__`.
- Added robust type hints (e.g. `-> tuple[np.ndarray, np.ndarray]`) across `core.py`.

### Fixed
- Fixed a bug in `validate.py` that caused a `ValueError` crash when evaluating very small or flat signal density maps (enforced a minimum step size in the loop).
- Fixed `mkdocs` build warnings (`griffe` missing type annotations) by providing correct function signatures.

### Changed
- Standardized all Python docstrings in `core.py`, `cli.py`, and `validate.py` to use Google-style docstrings.
- Updated `api.md` so that the validation module is properly rendered in MKDocs.

## [0.1.2] - 2026-06-07

### Changed
- Maintenance release: General updates, dependency pins, and CI/CD workflow improvements.

## [0.1.0] - Initial Release
- Initial release of the `synth-cryo-em` utility.
