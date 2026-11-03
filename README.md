# Mutagen | Protein Mutation Effect Prediction

Course project by Antonio Pagnotta, Andrea Scalise and Francesco de Martino
for [Advanced Machine Learning](https://corsidilaurea.uniroma1.it/it/archive/2025/33519/attendance/lesson/57c6368b-1013-445e-b969-8964718eb224/2/1/44a53763-f229-4363-ae15-711615933758/b4e2a82f-9697-4caa-9594-7a441228b434?channel=0) in the Data Science program at Sapienza University of Rome (2025/2026).
Francesco de Martino earned **30/30 cum laude (30 e lode)** on the course exam.

Mutagen asks whether hyperbolic representations help distinguish rare, extreme
effects of single amino acid substitutions. The original experiments used
the stability subset of [VenusMutHub](https://huggingface.co/datasets/AI4Protein/VenusMutHub).
The project extracted per-residue embeddings from the AlphaFold 3 Pairformer
before diffusion, then used a Transformer with attention pooling to obtain one
vector per sequence.

The target was divided at -2, -0.5, 0.5 and 2 into five classes from destroyed
to improved. The experiments compared a Euclidean MLP, a Poincare-ball model
trained with Riemannian Adam, and a hierarchical variant inspired by entailment
cones. Class-weighted cross-entropy addressed class imbalance. The repository
also includes an experimental classifier that encodes a mutant relative to
its wild type.

## Workflow

1. Obtain matched wild-type and mutant sequences and retain the assay, measurement
   type, and mutation identifier for each observation.
2. Run a separately installed [AlphaFold 3](https://github.com/google-deepmind/alphafold3)
   with authorized parameters and `--save_embeddings=true`. Retain the single
   embeddings, typically shaped `(sequence length, 384)` for a standard single chain.
3. Prepare local arrays and a table linking each variant to its target.
   [merge_embeddings_with_fitness_labels.ipynb](notebooks/merge_embeddings_with_fitness_labels.ipynb) records the original data
   preparation. It renames files in place and picks the first duplicate match.
   Use a scratch copy and resolve duplicate measurements by assay before training.
4. [train_attention_pooling.ipynb](notebooks/train_attention_pooling.ipynb) records the original attention pooling
   experiment, which reads `reduced_256_embedd.pkl` and writes
   `finalized_pooled_embeddings.pkl`. The downstream script expects the latter
   in the working directory with fields `embedding`, `file_name`, and `fitness`.
   Names use `<protein>_WT.npy` and `<protein>_<mutation>.npy`. Load pickle files
   only when you generated them or trust their source.

Install the downstream dependencies with `python -m pip install -r requirements.txt`.
With the local pooled artifact in place, run `python train_hyperbolic_delta.py`
from the repository root. The other notebooks preserve the exploratory comparisons.
AlphaFold 3 installation and GPU inference are separate from this Python environment.

## Evaluation status

The course report recorded the following exploratory validation results.
The hyperbolic model had the highest macro F1, and the hierarchical model had
the highest reported Spearman correlation.

| Metric | Euclidean | Hyperbolic | Hierarchical hyperbolic |
| --- | ---: | ---: | ---: |
| Macro F1 | 0.724 | 0.790 | 0.772 |
| Spearman rho | 0.891 | 0.897 | 0.929 |

**AlphaFold 3 output notice.** These aggregate findings are derived from
AlphaFold 3 embeddings transformed through attention pooling and downstream
prediction. Use is subject to the
[AlphaFold 3 Output Terms of Use](third_party/alphafold3/OUTPUT_TERMS_OF_USE.md)
and the accompanying [Legally Binding Terms of Use](<third_party/alphafold3/Legally Binding Terms of Use.txt>).
The table transcribes the course report and was not regenerated from this snapshot.

The original experiments used random splits of variants. The archived attention
pooling notebook fits against the full set of targets before downstream splitting,
so those validation metrics are exploratory. A clean evaluation must assign folds
before fitting pooling or the predictor. Report assay-specific targets and keep
stability measurements such as delta-delta-G and delta-Tm separate. For a future
ProteinGym evaluation, use its supervised single-substitution folds and report
per-assay Spearman and MSE on held-out variants.

## AlphaFold 3 attribution and terms

Abramson, J. et al. *Accurate structure prediction of biomolecular interactions
with AlphaFold 3*. Nature (2024). https://doi.org/10.1038/s41586-024-07487-w

AlphaFold 3 is an external dependency. Its upstream source is covered by
[Apache 2.0](third_party/alphafold3/LICENSE). Obtain model parameters directly
from Google under the [Model Parameters Terms of Use](third_party/alphafold3/WEIGHTS_TERMS_OF_USE.md)
and [Prohibited Use Policy](third_party/alphafold3/WEIGHTS_PROHIBITED_USE_POLICY.md).
The upload contains project source and documentation. Keep model parameters,
generated embeddings, downstream checkpoints, and third-party data local.
Any separately published AlphaFold 3 output, including substantially derived
material, is subject to the [Output Terms of Use](third_party/alphafold3/OUTPUT_TERMS_OF_USE.md).
Those terms cover noncommercial use, attribution, notices and disclosure of
transformations such as pooling.
