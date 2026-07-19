# mi-lens

![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![Models](https://img.shields.io/badge/models-TinyLlama%20%7C%20Pythia-6A5ACD)
![Lenses](https://img.shields.io/badge/lenses-Logit%20Lens%20%7C%20Tuned%20Lens%20%7C%20J--lens-0F766E)
![Analysis](https://img.shields.io/badge/analysis-heatmaps%20%7C%20fuzzy--trace%20%7C%20residual%20metrics-2563EB)
![Status](https://img.shields.io/badge/status-research%20prototype-D97706)

Project-owned toolkit for comparing **logit lens**, **tuned lens**, and **J-lens** readouts on the same prompts, layers, and models. The repository vendors the exact `jlens` and `tuned_lens` code used in the experiments, and adds local plotting, widget, and evaluation utilities for side-by-side analysis.

![Lens diff interactive plotting widget: J-lens vs. Tuned lens](assets/jlens_tunedlens.png)

The current examples and analysis notebooks focus on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, but the repository is structured to support broader lens experiments, including Pythia-style models, through the vendored packages and local analysis modules.

## Lens Readouts

Let `h_l` be the residual stream at layer `l`, `N` the model's final normalization, and `U` the unembedding matrix.

- **Logit lens** reads out intermediate predictions by prematurely applying the model's final readout:
  `z_l = U(N(h_l))`
- **Tuned lens** learns a layer-specific translator `T_l` before unembedding:
  `z_l = U(N(h_l + T_l(h_l)))`
- **J-lens** transports `h_l` through a fitted Jacobian map `J_l` before unembedding:
  `z_l = U(N(J_l h_l))`

## Evaluation

We compare lenses along three complementary axes:

- **Verbatim fidelity**: exact-token metrics such as top-1, top-k, gold rank, and gold probability.
- **Surface/gist proxy**: softer overlap metrics based on normalized token forms and top-k character n-gram similarity.
- **Mechanistic faithfulness**: hidden-space agreement with the model's actual final residual, measured with cosine similarity and relative `L2` error.

These metrics are implemented in the local `methods` package and used throughout the notebooks and comparison widgets.

## Install

```bash
pip install -e .
```

This installs the checked-in project package together with the vendored lens packages and local analysis modules, including:

- `jlens`
- `tuned_lens`
- `plotting`
- `methods`

## Repository Layout

- `lenses/jacobian_lens`: vendored Jacobian Lens code.
- `lenses/tuned_logit_lens`: vendored Tuned Lens / Logit Lens code.
- `src/plotting`: local widgets and comparison plots.
- `src/methods`: local evaluation helpers, including fuzzy-trace-style metrics and residual-alignment analyses.
- `notebooks`: exploratory analysis and comparison notebooks.

## Current Focus

The repository is currently centered on:

- pairwise lens comparison widgets for `logit lens`, `J-lens`, and `tuned lens`
- prompt-level and layerwise analysis on TinyLlama
- evaluation pipelines that separate predictive fit from more mechanistic hidden-state agreement

For deeper examples, see the notebooks under `notebooks/`.
