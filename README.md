# mi-lens

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Lenses](https://img.shields.io/badge/lenses-Logit%20%7C%20Tuned%20%7C%20J--lens-0F766E)
![MoE](https://img.shields.io/badge/MoE-FlexOlmo%20%7C%20HF--MoE-2563EB)
![Status](https://img.shields.io/badge/status-research%20toolkit-D97706)

`mi-lens` is a research toolkit for layer-wise analysis of causal language
models and mixture-of-experts models. It combines three related but distinct
lines of analysis:

1. **Lens readouts:** Logit Lens, Tuned Lens, and J-Lens on matched prompts,
   layers, languages, and model variants.
2. **Router and expert analysis:** router geometry, token-level expert
   allocation, probability and margin statistics, activation and co-activation,
   public-versus-domain expert comparisons, and output-facing expert effects.
3. **Sparse routing analysis:** SAE and ITDA predictors of the model's observed
   expert selection, with unigram, bigram, neuron, and PCA controls.

The main study uses FlexOlmo and FlexDanish compositions. The repository also
contains reusable adapters and lens pipelines for other Hugging Face causal
language models.

## Scientific Scope

The analyses answer different questions and should not be conflated:

| Analysis | Primary question |
| --- | --- |
| Logit/Tuned/J-Lens | How do intermediate residual states read out as token predictions? |
| Router statistics | Which experts are selected, with what probability and weight, and on which inputs? |
| Expert contribution/prism analysis | Does a public or domain expert increase or decrease the output-relevant logit contribution? |
| Weight geometry | How similar are router rows and expert MLP weights, including distance from the public expert? |
| SAE/ITDA predictors | Can sparse pre-router features predict the router's observed selected-expert set? |
| FlexMORE-oriented analysis | Which MLP sublayers and low-rank expert deltas are most important or sensitive? |

SAE and ITDA results are therefore predictive evidence about routing, not a
complete causal explanation of why an expert was selected. Router statistics,
expert-output decomposition, residual analysis, and output-facing logit
measurements provide complementary evidence.

## Model and Architecture Support

Support is adapter-based rather than tied to one toy model:

- **Standard Hugging Face causal LMs:** lens capture and comparison work when
  the model exposes ordinary hidden states, a final normalization, and an
  output projection compatible with `AutoModelForCausalLM`.
- **Generic Hugging Face MoE models:** `HFMoEAdapter` uses
  `output_router_logits=True` when the model exposes router logits through the
  Hugging Face output API.
- **FlexOlmo/FlexDanish/FlexMoRE-style models:** `FlexOlmoAdapter` handles the
  local Flex gate outputs and captures pre-router states, router probabilities,
  selected experts, and routed MLP outputs. Flex checkpoints may require the
  project-specific Transformers fork and GPT-2 slow-tokenizer configuration.
- **Configured model families:** the repository currently contains configs for
  Apertus, Ministral, Qwen, and FlexDanish variants. The adapter is the
  compatibility boundary; a new architecture needs an adapter if its hidden
  states, unembedding, or router outputs use a different interface.
- **TinyLlama:** retained as a small smoke-test and layout-validation model. It
  is not the scientific focus of the study.

Architecture support means that the interface is implemented and configurable;
it does not guarantee that every checkpoint can load without its own tokenizer
or custom-Transformers requirements. Run the compatibility check before a
large extraction.

## How A Lens Is Applied

For a residual state `h_l` at layer `l`, a normal logit lens applies the final
normalization and unembedding:

```text
logits_l = unembed(final_norm(h_l))
```

Tuned Lens and J-Lens first transform or transport the intermediate state and
then use the corresponding output readout. The stored summaries include target
probability, surprisal, rank, top-1, entropy, top-1/top-2 margin, top-k token
IDs and decoded tokens, plus comparisons with the final model distribution.

For an MoE block, the ordinary lens is applied to the actual residual stream.
The MoE-specific analysis additionally computes the routed MLP mixture and,
where supported, individual weighted expert outputs. These vectors are
projected into output space only as controlled contribution diagnostics. A
gate or up projection is not treated as an independent additive logit source:
the gated MLP contains a multiplicative interaction. The down-projection and
complete expert outputs are the valid additive residual contributions.

## Installation

Create the project environment on the machine where the analysis will run:

```bash
conda env create -f environment.yml
conda activate mi-lens-env
scripts/install_environment.sh
```

For FlexOlmo/FlexDanish checkpoints that require the local Transformers fork:

```bash
scripts/install_environment.sh \
  --flex-transformers /work/training/transformers
```

The project-specific tokenizer configuration is selected by the model config;
Flex checkpoints normally use `GPT2Tokenizer` with `use_fast: false`. Verify
the loaded Transformers path and version before starting a long run.

The vendored `jlens` and `tuned_lens` sources are tracked in this repository.
FlexEval is useful reference and analysis code, but it is not silently modified
or required as a runtime dependency by `mi-lens`.

## Main Workflows

### Compatibility check

```bash
python scripts/compatibility/run_model_compatibility.py \
  --config configs/compatibility/flexdanish_55b_bf16.json
```

### Fit J-Lens or Tuned Lens

```bash
python scripts/jlens/run_fit_jlens.py \
  --config configs/jlens_fit/qwen_base_da_trainfit.json

python scripts/tuned_lens/run_train_tuned_lens.py \
  --config configs/tuned_lens_train/qwen_base_da_trainfit.json
```

Use the matching capture config for the same model, tokenizer, data role, and
maximum-length policy:

```bash
python scripts/capture/run_capture_logit_lens.py \
  --config configs/capture/qwen_base_da_eval.json

python scripts/capture/run_capture_tuned_lens.py \
  --config configs/capture/qwen_base_da_eval.json

python scripts/capture/run_capture_jlens.py \
  --config configs/capture/qwen_base_da_eval.json
```

### FlexLens

FlexLens is a separate output-facing pass over a composed Flex model. It does
not replace RouterInterp or modify its artifacts:

```bash
python scripts/capture_flexlens.py \
  --config configs/flexlens_flexdanish_8x7b_a4_55b_v2.json
```

It writes compact per-token rows and optional per-prompt tensor artifacts under
the configured project output directory. Full vocabulary logits are not stored
in every row by default.

### Router, sparse, and expert analysis

The full Flex RouterInterp batch is generated from the matrix config. Passing
`--run` runs the generated resumable batch immediately:

```bash
python scripts/routerinterp/build_flex_routerinterp_batch.py \
  --config configs/routerinterp/flex_routerinterp_full.json \
  --run
```

Alternatively, build first and run the generated manifest separately:

```bash
python scripts/routerinterp/build_flex_routerinterp_batch.py \
  --config configs/routerinterp/flex_routerinterp_full.json

python scripts/routerinterp/run_routerinterp_batch.py \
  --config tmp/routerinterp/batch_configs/flex_routerinterp_paper_protocol/batch.json
```

The batch runs the configured five router layers and keeps each model stage in
its own process. This reduces stale model and CUDA state between models. It
captures router inputs, probabilities, selected IDs and weights, activation and
co-activation summaries, sparse predictors, ITDA comparisons, and supporting
geometry diagnostics according to the selected config.

## Data and Reproducibility

Dataset source partitions and analysis roles are distinct. The study has two
roles:

- **fit:** source-order tokens used to fit sparse dictionaries and routing
  probes;
- **held-out analysis:** disjoint source-order data used for router, domain,
  activation, co-activation, and predictor metrics.

Provider names such as `train`, `validation`, `test`, `retain1`, and `retain2`
describe the source dataset; they are not additional project roles.

Router datasets are loaded from their configured Hugging Face or local source
at runtime, without shuffling. Dataset caches, manifests, model intermediates,
router artifacts, and reports are written below the project `tmp/` directory.
The full router capture is streamed into SAE fitting where required; paper-scale
fit activations are not accumulated as one multi-terabyte tensor.

Plain-text and lens-specific exports can still be created with the helpers in
`src/mi_lens/methods/data_prep.py` when a fixed local manifest is useful. Every
export should retain source indices and dataset metadata.

## Repository Layout

- `src/mi_lens/adapters`: model and router compatibility adapters.
- `src/mi_lens/methods`: lens metrics, tokenization audits, data preparation,
  fuzzy-trace comparisons, and FlexLens measurements.
- `src/mi_lens/pipelines`: fit, capture, comparison, RouterInterp, and FlexLens
  orchestration.
- `src/mi_lens/sparse`: SAE, ITDA, feature, and RouterInterp components.
- `src/mi_lens/plotting`: Plotly/PDF reports, router figures, and lens widgets.
- `lenses/jacobian_lens`: vendored J-Lens implementation.
- `lenses/tuned_logit_lens`: vendored Tuned Lens and Logit Lens implementation.
- `configs`: model, fit, capture, router, and compatibility configurations.
- `scripts`: command-line entry points for reproducible runs.
- `notebooks`: exploratory work and result inspection.
- `tmp`: runtime-only caches, manifests, captures, and reports; do not commit
  large model or activation artifacts.

## Interpreting Results

The project reports capability, routing, sparse prediction, and contribution
results separately. A high public-expert selection rate is not automatically
router collapse: for dense `a7`/`a8` compositions all experts may be active by
construction, so normalized weights and output contributions are required.
Likewise, a good SAE F1 score means that sparse features predict observed
selection; it does not prove that those features caused the model's answer.

All numerical summaries should reject non-finite values and validate their
domain, such as probabilities in `[0, 1]`, normalized probability rows,
selection shares in `[0, 1]`, and cosine similarities in `[-1, 1]` up to
tolerance. Incomplete benchmark results remain explicitly missing rather than
being converted to zero.
