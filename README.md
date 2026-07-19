# mi-lens

## Examples using TinyLlama/TinyLlama-1.1B-Chat-v1.0
### Lens diff interactive heatmap widgets
![Lens diff interactive plotting widget: J-Lens vs. Tuned Lens](assets/jlens_tunedlens.png)
![Lens diff interactive plotting widget: Logit Lens vs. Tuned Lens](assets/logitlens_tunedlens.png)

### Lens diff layer-wise
![Gold rank lens diff: J-Lens, Tuned Lens, Logit Lens](assets/gold_rank.png)
![Gold prob lens diff: J-Lens, Tuned Lens, Logit Lens](assets/gold_prob.png)
![Jensen-Shannon divergence vs final lens diff: J-Lens, Tuned Lens, Logit Lens](assets/js_vs_final.png)
![Jaccard vs final lens diff: J-Lens, Tuned Lens, Logit Lens](assets/jaccard_vs_final.png)

Project-owned lens analysis toolkit with vendored `jlens` and `tuned_lens` code,
plus local plotting and evaluation methods for comparing lens readouts.

## Install

```bash
pip install -e .
```

This installs the exact checked-in versions of:
- `jlens`
- `tuned_lens`
- `plotting`
- `methods`
