# Provenance

Written by us: SAE architecture (the SAE class), training loop, activation caching script, evaluation and metrics code.

Adapted from library: model loading and activation extraction uses TransformerLens's HookedTransformer and run_with_cache API (standard usage, not modified).

Reused as-is: none. No external SAE training code was copied.

Results reported by original authors: not directly compared numerically at this stage. Qualitative comparison planned for the final report.

Results from our own team: all metrics in README.md (MSE, L0, variance explained) are from our own training run on approximately 500,000 subsampled activations.
