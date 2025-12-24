## LLM scenario harness (Anthropic)

This repo is a small, config-driven harness to run **scenario × frame** prompts against Anthropic models, log raw attempts to JSONL, apply hybrid labeling (heuristics + judge-blind LLM judge), and export a CSV.

### Quickstart

- **Install** (recommended: use your existing `venv/`):

```bash
python -m pip install -r requirements.txt
```

- **Configure**:
  - `config/experiment.yaml`
  - `scenarios/scenarios.yaml`
  - `frames/frames.yaml`

- **Run**:

```bash
export ANTHROPIC_API_KEY="..."
python -m src.main run --config config/experiment.yaml --suite baseline_anthropic
```

- **Export CSV**:

```bash
python -m src.main export-csv --config config/experiment.yaml
```

### Outputs
- `outputs/<run_id>/attempts_raw.jsonl`: request/response + metadata, append-only
- `outputs/<run_id>/attempts_labels.jsonl`: labeling records, append-only
- `outputs/<run_id>/attempts.csv`: merged view for analysis


