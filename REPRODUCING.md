# Reproducing the paper's results

This file maps each table and figure in *Bergson: An Open Source Library for
Data Attribution* to the configuration files, scripts, and raw records in this
artifact. Every Bergson run writes a self-describing `config.yaml` into its
output directory; re-running `bergson <config.yaml>` replays it exactly (see
`docs/reproducibility.rst`).

## Environment

```bash
git clone ANONYMIZED_REPOSITORY_URL bergson
cd bergson
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -e ".[dev]"                             # torch >= 2.5, transformers, datasets
# optional, for the dattri / Kronfluence comparison benchmarks:
pip install dattri kronfluence
```

All models and datasets are public and downloaded from Hugging Face on first
use (`gpt2`, `EleutherAI/pythia-*`, `EleutherAI/bergson-wikitext-512-chunks`,
`EleutherAI/SmolLM2-135M-10B`, `cais/wmdp`, `NeelNanda/pile-10k`). Multi-GPU
runs use `torchrun`-style `distributed:` settings inside each YAML; reduce
`nproc_per_node` and raise `grad_accum_steps` to run on fewer GPUs.

## Figure 2 — TrackStar pipeline as a YAML file

```bash
bergson examples/pipelines/trackstar_wmdp.yaml
```

The paper's listing shows the same pipeline pointed at `HuggingFaceTB/SmolLM3-3B`
and `allenai/dolmino-mix-1124`; substitute those identifiers for the
`model:`/`dataset:` fields to reproduce it verbatim.

## Table 1 — Linear datamodeling scores (GPT-2 fine-tuned on WikiText)

All four methods are validated against the same family of leave-1%-out
retrained models using `bergson validate`, which is the last step of each
pipeline below. Run the MAGIC pipeline first: it trains the model, computes
per-query MAGIC scores, and saves the retrained models that the other three
pipelines validate against.

| Row | Command |
|---|---|
| MAGIC | `bergson examples/compare_wikitext/magic_paper_bs64_eps1e-6.yaml` |
| SOURCE | `bergson examples/compare_wikitext/source.yaml` |
| EK-FAC | `bergson examples/compare_wikitext/ekfac.yaml` |
| TrackStar | `bergson examples/compare_wikitext/trackstar.yaml` |

`magic_paper_bs64_eps1e-6.yaml` uses the hyperparameters stated in the paper
(batch size 64, Adam root-epsilon 1e-6, `m = 50` queries, `N = 400` retrained
models). The remaining YAMLs in `examples/compare_wikitext/` ship with the
library's current recipe (batch size 256, root-epsilon 1e-8, `N = 100`); the
per-query results of that recipe are tabulated in
`examples/compare_wikitext/README.md`. To validate SOURCE/EK-FAC/TrackStar
against the `N = 400` bank, set `retrained_dir: runs/compare_wikitext/random_paper`
in the `validate:` step of each YAML.
The TrackStar row of Table 1 uses a per-module projection dimension of 1024
(`projection_dim: 1024` in `trackstar.yaml`; the shipped default is 32).

Mean Spearman (LDS) and Pearson correlations over the 50 queries, with
bootstrap confidence intervals over the retrained models, are printed at the
end of each `validate` step and written to `<run_path>/validate*/`.

## Figure 3a — Attribution time vs. number of training tokens (1 GPU)

End-to-end latency of MAGIC, EK-FAC, TrackStar, and gradient cosine similarity
for a single query against 512-token training items from
`EleutherAI/SmolLM2-135M-10B` using Pythia-160M.

```bash
# TrackStar / EK-FAC / cosine (bergson build -> reduce -> score), 10K..100M tokens
bash benchmarks/run_small_models_cli_1gpu.sh        # edit TOKEN_SCALES / MODELS at the top
# MAGIC (fine-tune with batch size 32, then unrolled differentiation)
bash benchmarks/run_small_models_magic_1gpu.sh
# Hessian / factor fitting overhead (EK-FAC vs. Kronfluence)
python -m benchmarks.benchmark_factors --help
```

Each run writes a `benchmark_cli.json` / `benchmark_magic.json` record under
`runs/<benchmark>/<model>/<tokens>-…/`. Aggregate records and plot with

```bash
python -m benchmarks.plot_cli_benchmark --source 1:runs/benchmarks/cli_benchmark_1gpu.csv --output_path figures/
python -m benchmarks.plot_programmatic_benchmark --help
```

Raw records from our runs are committed under `runs/benchmarks/*.csv`,
`runs/bergson_inmem_benchmark/`, `runs/dattri_benchmark/`,
`runs/proj_comparison/`, and `docs/benchmarks/`.

## Figure 3b — Gradient-collection latency vs. GPU count

Gradient collection (`bergson build`) over 10M tokens of
`EleutherAI/SmolLM2-135M-10B` for Pythia-160M, Pythia-1B, and Pythia-12B on
1–16 GPUs. Models ≥ 1B parameters are sharded with FSDP automatically; smaller
models use replicated data parallelism.

```bash
bash benchmarks/run_small_models_cli_1gpu.sh    # 1 GPU
bash benchmarks/run_small_models_cli_2gpu.sh    # 2 GPUs
bash benchmarks/run_small_models_cli_8gpu.sh    # 8 GPUs (70M / 160M / 1B)
bash benchmarks/run_large_models_cli_8gpu.sh    # 8 GPUs (6.9B / 12B)
bash benchmarks/parallel_runs/launch_all_benchmarks.sh   # one model per GPU, in parallel
```

The 16-GPU points are the 8-GPU configuration launched on two nodes with
`nnode: 2` (data parallelism across nodes, FSDP within each node); see
`scripts/run_cli_benchmark_slurm.py` for the SLURM launcher we used. The
theoretical zero-communication curve in the figure is
`t(1 GPU) / num_gpus + startup`, where `startup` is the build-time intercept
measured at 10K tokens. Combine per-GPU-count CSVs with

```bash
python -m benchmarks.plot_cli_benchmark \
  --source 1:runs/benchmarks/cli_benchmark_1gpu.csv \
  --source 8:runs/benchmarks/cli_benchmark_8gpu.csv \
  --output_path figures/
```

### Hardware note

The paper reports timings on NVIDIA A100 80GB GPUs. The records committed in
this artifact were collected on the hardware recorded in each CSV/JSON
(`hardware` column: A40, GH200 120GB, or H100 80GB); absolute times differ from
the paper, scaling behaviour does not.

## Appendix A.1 — Token-level attribution of biosecurity knowledge

Pipeline: fine-tune the Deep Ignorance 7B model with LoRA adapters (rank 32,
α = 64, dropout 0; all MLP and attention modules) on 130M tokens of the WMDP
forget set with Adam (β₁ = 0.9, β₂ = 0.999), FP16, batch size 16, sequence
length 1024, 20 warm-up steps, gradient clipping 1.0; then attribute the
robust-subset MCQA accuracy with per-token MAGIC and re-train with the top-10%
tokens up-weighted ×5.

* Training with a LoRA adapter: `bergson train` (see `examples/train_lora.py`
  and `docs/training.rst`).
* MCQA query formatting: `bergson/templates/mcqa.yaml` (LM-Evaluation-Harness
  style template; used for `cais/wmdp`).
* Per-token MAGIC attribution: `bergson magic` with `per_token: true`
  (`docs/magic.rst`).
* Token re-weighting for the re-training run: the trainer's per-token weighted
  loss, which re-weights items without changing data order (`docs/training.rst`).

The final fine-tuned artifacts are deliberately not released (see the paper's
Broader Impacts section); the configuration above reproduces the procedure on
the public `cais/wmdp` forget set.

## Appendix A.2 — Attributing non-differentiable objectives with GRPO

Provide a `reward` column alongside each rollout and pass `--reward_column
reward` to preprocessing (`docs/data-preprocessing.rst`, "Rewards"). When a
reward column is present, Bergson computes the Dr. GRPO policy-gradient loss
over each prompt group (rows with missing rewards are dropped with
`--skip_nan_rewards`) and the resulting gradients can be attributed with any of
the gradient-based methods (`bergson build` / `bergson score`; see
`docs/gradient-collection.rst`).

## Appendix E — Library comparison (Tables 2 and 3)

Feature and method coverage were verified by running each library's
reference example on one 8×A100 node:

```bash
python -m benchmarks.benchmark_dattri --help          # dattri
python -m benchmarks.kronfluence_benchmark --help     # Kronfluence
bash scripts/run_bergson_vram_8gpu.sh                  # largest-model / VRAM ceiling, Bergson
bash scripts/run_kronfluence_vram_8gpu.sh              # largest-model / VRAM ceiling, Kronfluence
python -m benchmarks.plot_vram_comparison --help
```

VRAM records are committed under `runs/vram_benchmark/`,
`runs/kronfluence_vram_benchmark/`, and `runs/dattri_vram_benchmark/`.

## What is not included

* The exact `config.yaml` files serialized by the A100 scaling runs and by the
  Deep Ignorance fine-tuning run are not part of this snapshot; the
  configurations above are their sources.
* Raw scheduler logs from the benchmark runs were omitted (they contain cluster
  paths); the aggregate CSV/JSON measurements they produced are included.
