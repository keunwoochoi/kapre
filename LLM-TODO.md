# Kapre Test Fixing Progress

## Summary of work done:

- Set up a virtual environment and installed all project dependencies.
- Updated dependencies (`numpy`, `librosa`, `tensorflow`) to newer versions and handled compatibility issues.
- Systematically ran and debugged test files one by one.
- **`tests/test_backend.py`**: All tests are now passing.
  - Fixed `NameError` for `librosa`.
  - Replaced deprecated `np.int`.
  - Fixed `TypeError` in `librosa.fft_frequencies`.
  - Used `np.testing.assert_allclose` for float comparisons.
- **`tests/test_signal.py`**: All tests are now passing.
  - Fixed `NameError` for `backend`.
  - Fixed `TypeError` in `librosa` function calls (`frame`, `melspectrogram`).
  - Addressed Keras model serialization issues by:
    - Adding `@register_keras_serializable` to custom layers.
    - Fixing `get_config` and `__init__` methods to handle `data_format` correctly.
    - Replacing deprecated `save_format` argument in `model.save()`.
- **`tests/test_augmentation.py`**: All tests are now passing.
  - Fixed Keras serialization issues for `SpecAugment` and `ChannelSwap` layers.
  - Updated test logic to use `tf.keras.Input` layer for modern API compatibility, resolving `AttributeError`.

## Current Task in Progress:

- All tests are passing. The next step is to update the dependencies.

## Next Steps:

- Run and fix tests for any remaining test files.
- Once all tests are passing, update `setup.py` with the new tested dependency versions.
