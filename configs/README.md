# Configs

These JSON configs keep model, data, dtype, and sequence-length choices aligned
across:

- compatibility checks
- J-lens fitting
- tuned-lens training
- logit-lens extraction
- tuned-lens extraction
- J-lens extraction

Run from the project root:

```bash
python scripts/compatibility/run_model_compatibility.py --config configs/compatibility/qwen_base.json
python scripts/jlens/run_fit_jlens.py --config configs/jlens_fit/qwen_base_da_trainfit.json
python scripts/tuned_lens/run_train_tuned_lens.py --config configs/tuned_lens_train/qwen_base_da_trainfit.json
python scripts/capture/run_capture_logit_lens.py --config configs/capture/qwen_base_da_eval.json
python scripts/capture/run_capture_tuned_lens.py --config configs/capture/qwen_base_da_eval.json
python scripts/capture/run_capture_jlens.py --config configs/capture/qwen_base_da_eval.json
```

Conventions:

- `trainfit` configs use `data/train_fit/...`
- `eval` configs use `data/eval/...`
- capture configs are shared by logit / tuned / J-lens extraction
- J-lens fit and tuned-lens train use the same dataset path, dtype, and
  `max_seq_len` for each model/language pair
- `num_steps` is tuned-lens specific, while `skip_first` is J-lens specific
