# SAE Reproduction - Sparse Autoencoders Find Highly Interpretable Features in Language Models

Reproduction of Cunningham et al. (2024, ICLR) - training a sparse autoencoder on GPT-2 small's residual stream activations.

## Setup
- Base model: GPT-2 small (gpt2, via TransformerLens v3.9.0, pinned for HookedTransformer API compatibility)
- Layer: residual stream after block 6 (blocks.6.hook_resid_post)
- Dataset: NeelNanda/pile-10k
- Activation vectors used: approximately 500,000 tokens (subsampled, stride=8, capped at 512 tokens per document)
- SAE architecture: single hidden layer, 768 to 6144 (8x expansion) to 768, ReLU activation, tied bias term
- Loss: MSE reconstruction plus L1 sparsity penalty (coefficient 1e-3)
- Training: 5 epochs, batch size 1024, Adam optimizer, learning rate 3e-4

## Key design deviations from the paper and why
- Reduced scale: used approximately 500k activation vectors instead of the paper's full-scale training set, due to Colab free-tier GPU and storage limits. Full-scale caching of around 10 million vectors would require about 30GB in float32, infeasible on free-tier compute.
- Activation normalization: raw GPT-2 residual stream activations have large, inconsistent magnitudes. We rescale so the average L2 norm equals sqrt of d_model before training. Without this, training was unstable, with loss spiking non-monotonically. This is standard practice in SAE training but not explicit in the original proposal.
- Learning rate: lowered from an initial 1e-3 to 3e-4 after observing instability on unnormalized data in early testing.

## Results

Final MSE (reconstruction loss): 0.0514
Variance explained: 99.40 percent
Average L0 (active features per token): 199.8 out of 6144, about 3.3 percent
Dictionary size: 6144

Training loss decreased monotonically across all 5 epochs, with no instability after normalization was applied.

Training log:
epoch 0: avg_loss=0.8330, avg_mse=0.3786, avg_l1=454.3962
epoch 1: avg_loss=0.1664, avg_mse=0.0863, avg_l1=80.0925
epoch 2: avg_loss=0.1263, avg_mse=0.0701, avg_l1=56.2251
epoch 3: avg_loss=0.1073, avg_mse=0.0605, avg_l1=46.7688
epoch 4: avg_loss=0.0961, avg_mse=0.0542, avg_l1=41.9829

## How to reproduce
1. Install dependencies: pip install -r requirements.txt
2. Run the notebook cells in order: load GPT-2, cache activations, train SAE, evaluate
3. Checkpoint saved to checkpoints/sae.pt

## Status
Stage 3 (reproduction) complete. Stage 4 (own experiment, hyperparameter sweep on L1 coefficient) not yet started.
