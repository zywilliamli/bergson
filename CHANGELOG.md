# CHANGELOG


## v0.19.0 (2026-08-02)

### Features

- Adam SOURCE preconditioner and Eq-43 hybrid
  ([#379](https://github.com/EleutherAI/bergson/pull/379),
  [`5d4b765`](https://github.com/EleutherAI/bergson/commit/5d4b76567d64e730014280f34ecd6c1f2708e5ed))

Evaluate the unrolling eigenfunctions on a diagonal approximation of the preconditioned Hessian
  P^1/2 H P^1/2, built per segment from the checkpoints' saved second moments (Bae et al. 2024, App.
  C). With adam_segment_hybrid, F_segment uses the App. D Eq-43 form: the diagonal supplies only the
  matrix exponential and the EK-FAC factors supply H^-1. Requires optimizer.pt in each checkpoint
  dir, written by the trainer's save_optimizer_state and carried across by export_checkpoints.

### Testing

- Save_optimizer_state must not hang under FSDP
  ([#380](https://github.com/EleutherAI/bergson/pull/380),
  [`054ebc7`](https://github.com/EleutherAI/bergson/commit/054ebc7c25901a0ae75a3a69cc6ba4957a2328ec))

Covers the every-rank collective fix from #376.


## v0.18.0 (2026-08-02)

### Features

- Auto-export trainer DCP checkpoints in the SOURCE pipeline
  ([#378](https://github.com/EleutherAI/bergson/pull/378),
  [`92867b2`](https://github.com/EleutherAI/bergson/commit/92867b27c993b6c1df2f184f15479bdf9671efc9))

* fix: read the training config out of the saved {steps, metadata} document

save_run_config has always wrapped the step list in a document with a metadata block, so
  load_training_config rejected every real run directory; its test hand-rolled the bare list it
  expected instead of calling save_run_config. Have the test write configs the way the writer does,
  and report an unparseable payload as ValueError so callers guessing at a run dir still degrade to
  their fallback.

(cherry picked from commit cf91c5b67892ff4e4c830ba816099d5335033962)

* feat: auto-export trainer DCP checkpoints in the SOURCE pipeline

resolve() converts raw checkpoints/step_<n>.ckpt paths to the HF exported/checkpoint-<n> dirs
  from_pretrained needs, exporting on demand and reusing existing exports, instead of rejecting
  them. load_training_config gains a fallback so a checkpoint whose sibling config.yaml belongs to
  an attribution run infers nothing rather than failing the pipeline.

* refactor: put trainer and export code with their owners

trainer_run.py becomes train_cfg_io.py: it reads the training run's config and artifacts, and the
  old name read as "run the trainer". write_lr_history moves into the magic trainer that calls it,
  ending magic's import of approx_unrolling; ensure_exported and EXPORT_DIRNAME move next to
  export_checkpoints in utils.trainer_export.


## v0.17.1 (2026-08-02)

### Bug Fixes

- Score each segment against its own checkpoints' training gradients
  ([#377](https://github.com/EleutherAI/bergson/pull/377),
  [`4d5de7e`](https://github.com/EleutherAI/bergson/commit/4d5de7e33c66eeb47408dcc00e7a641d9e4141b1))

score_per_segment_and_aggregate evaluated every segment's training gradients at the final
  checkpoint, collapsing SOURCE toward influence functions at convergence. Bae et al. 2024 Sec. 3.4
  defines g_bar_l as the expected training gradient over segment l's checkpoints; scores are linear
  in the training gradient, so averaging per-checkpoint scores within the segment is equivalent and
  reuses the existing scorer.


## v0.17.0 (2026-08-02)

### Bug Fixes

- Call save_second_moments_as_optimizer_pt from every rank
  ([#376](https://github.com/EleutherAI/bergson/pull/376),
  [`4174502`](https://github.com/EleutherAI/bergson/commit/41745028e7966538b889e9c29b60ed9ba7421347))

Gathering FSDP's sharded (DTensor) moments is a collective, so gating the call on global_rank == 0
  hangs non-zero ranks; the function already restricts the file write to rank 0.

### Features

- Interval save mode for the trainer ([#373](https://github.com/EleutherAI/bergson/pull/373),
  [`8eea5ef`](https://github.com/EleutherAI/bergson/commit/8eea5ef2d5bfe91245454f5c8018376631508a9f))

save_mode="interval" writes a checkpoint every save_interval steps, plus the final state when the
  cadence lands on it (the other modes' backward replay indexes the data stream by checkpoint, so
  only interval mode may add a trailing snapshot). save_mode and save_interval move from MagicConfig
  to TrainingConfig so `bergson train` covers trainer-only runs such as producing SOURCE
  checkpoints, and MagicSaveMode becomes SaveMode to match; the MAGIC backward rejects interval
  mode.


## v0.16.1 (2026-08-02)

### Bug Fixes

- Normalize SOURCE segment eigenvalues per document
  ([#375](https://github.com/EleutherAI/bergson/pull/375),
  [`ad939f5`](https://github.com/EleutherAI/bergson/commit/ad939f5afc69808d23c26926a5e4b712c6b9fab7))

Segment eigenvalues were unnormalized sums over documents x checkpoints. EK-FAC is unaffected --
  damping is relative to mean(lambda), so every inversion is homogeneous in the scale -- but
  SOURCE's eigenfunctions are not scale-invariant: sigma enters exp(-lr*K*sigma), and the
  unnormalized scale puts it in the function's dead range (LDS 0.125 vs 0.383). Eq. 1 of Bae et al.
  2024 defines the risk as a per-document mean, and kronfluence divides its lambda matrix by the
  processed-sample count; dividing the pooled segment sum by the pooled document count matches both.


## v0.16.0 (2026-08-02)

### Features

- **validate**: Save subsets.json by default and reuse it if present
  ([#372](https://github.com/EleutherAI/bergson/pull/372),
  [`781583b`](https://github.com/EleutherAI/bergson/commit/781583bf096f7d5e8315ac049df53de8fdfeeda4))

Write subsets.json and reload it when possible so different methods using the same re-train bank
  re-use them

### Refactoring

- Rename save_retrained_models to save_models
  ([#371](https://github.com/EleutherAI/bergson/pull/371),
  [`d6c3940`](https://github.com/EleutherAI/bergson/commit/d6c3940b0c0baf8a329cb3b75ef64f8eaeb63456))

Rename save_retrained_models -> save_models


## v0.15.1 (2026-07-29)

### Bug Fixes

- Import load_from_optimizer lazily to break a spawn-time cycle
  ([#368](https://github.com/EleutherAI/bergson/pull/368),
  [`f824782`](https://github.com/EleutherAI/bergson/commit/f8247824424db8a39216fe6b88228a7740e11536))

Importing it at module scope pulls in bergson.gradients, which re-enters the package __init__ and
  reaches magic.cli -> validate -> magic.trainer. In-process that resolves, but a spawned worker
  unpickling through magic.cli hits magic.trainer half-initialized and fails on TrainerState, taking
  out test_grad_accum_matches_full_batch.


## v0.15.0 (2026-07-29)

### Features

- Seamless Bergson Trainer -> SOURCE attribute
  ([#367](https://github.com/EleutherAI/bergson/pull/367),
  [`914c77a`](https://github.com/EleutherAI/bergson/commit/914c77a0eb208194015abc523b9f360a3d0a0bf2))

feat: run SOURCE directly off a bergson training run


## v0.14.0 (2026-07-29)

### Features

- Magic gradient accumulation ([#357](https://github.com/EleutherAI/bergson/pull/357),
  [`e7d0358`](https://github.com/EleutherAI/bergson/commit/e7d0358cd59b948a299019529f42ad174c89ce59))

feat: Add gradient accumulation (grad_accum_steps) to MAGIC trainer


## v0.13.4 (2026-07-28)

### Performance Improvements

- Re-use re-train bank losses in `bergson validate`
  ([#365](https://github.com/EleutherAI/bergson/pull/365),
  [`52a5a34`](https://github.com/EleutherAI/bergson/commit/52a5a34bf99a850cf7d502fe65708a6d5ebcf0b7))

* perf: cache method-independent bank losses in evaluate_retrained

Evaluating attribution scores against a pre-saved leave-k-out bank re-runs every banked model on the
  query set to get each subset's query-loss diff. Those per-subset losses depend only on the bank
  and the query set, never on the attribution scores, so scoring a second method against the same
  bank repeats the entire (dominant) cost for nothing.

Cache the per-subset losses (and baseline) under the bank keyed by the query set and load settings,
  and reuse them automatically whenever the key matches. The first method scored on a bank pays the
  full cost; every later method skips model evaluation entirely and only re-sums its own scores. A
  metadata guard recomputes when the key no longer matches.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.13.3 (2026-07-27)

### Performance Improvements

- Remove torch.compile from normalizers (fixes import on Python 3.14)
  ([#361](https://github.com/EleutherAI/bergson/pull/361),
  [`315088c`](https://github.com/EleutherAI/bergson/commit/315088c21fd17c9d2663b5172214688030baf67d))

* fix: guard torch.compile so bergson imports on Python 3.14

torch.compile raises RuntimeError at decoration time on interpreters Dynamo doesn't support (Python
  3.14+ with torch <= 2.9). The normalizers in bergson/gradients.py apply it at import time, so
  importing bergson crashed outright on 3.14.

- Add compile_if_supported: falls back to the eager function with a warning when torch.compile is
  unavailable - Add tests covering the fallback path, including a subprocess test that simulates the
  3.14 condition on any interpreter - Add an import-py314 CI job that installs bergson on Python
  3.14 and imports it, catching future import-time regressions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

* ci: run compile-fallback tests on the Python 3.14 job

* perf: remove torch.compile from normalizers

A100 benchmarks (torch 2.7.1, py3.11) show the compiled normalizers are net slower in the common
  case: ~50us of dynamo guard overhead per call dominates the sub-100us kernels at 768-dim shapes
  and on every bias path, and each run pays 0.3-1.5s compile latency per graph (17 graphs observed
  from bin-packing's varying batch sizes). Compiled kernels only win on very large tensors (up to
  ~2x on [4, 50304, 768] bf16), a regime worth a few percent of end-to-end build time at most, and
  only when --optimizer_state is set.

Removing the decorators also makes bergson importable on Python 3.14, where torch.compile raises at
  decoration time (torch <= 2.9). The import-py314 CI job guards against reintroducing import-time
  compiles.

---------

Co-authored-by: Claude Fable 5 <noreply@anthropic.com>


## v0.13.2 (2026-07-24)

### Bug Fixes

- Thread hessian_dtype into the EK-FAC lambda collector; add fp64
  ([#356](https://github.com/EleutherAI/bergson/pull/356),
  [`2671534`](https://github.com/EleutherAI/bergson/commit/26715342c5705782814596dde59450b98b1fb675))

hessian_dtype only reached the covariance collectors — the eigenvalue-correction pass ignored it and
  accumulated in the activations' dtype. Pass it to LambdaCollector in both the standard and
  per-checkpoint (SOURCE) paths, and add an fp64 option for kronfluence-parity precision studies.


## v0.13.1 (2026-07-22)

### Bug Fixes

- Scale random projections by 1/sqrt(p) (Johnson-Lindenstrauss)
  ([#346](https://github.com/EleutherAI/bergson/pull/346),
  [`4b9b255`](https://github.com/EleutherAI/bergson/commit/4b9b255b332d6efb15452873a41afcd57a2ca26e))

* Scale random projections by 1/sqrt(p) (Johnson-Lindenstrauss)

create_projection_matrix normalized each row over its n entries, giving entry variance 1/n. JL
  requires 1/p (p = projected dim) so that E[A^T A] = I and projected inner products are unbiased.
  The consequence is not the overall scale — a constant cancels from any ranking — but that the
  factor p/n depends on the module's SHAPE, so modules are silently reweighted against each other in
  the summed score. Attention (d x d) and MLP (d x 4d) blocks differ by 4x.

Measured E[proj]/true, 3000 draws per shape:

shape p before predicted p^2/(o*i) after 32x32 16 0.24972 0.25000 0.99889 64x64 32 0.25042 0.25000
  1.00167 128x128 32 0.06247 0.06250 0.99949 256x64 32 0.06253 0.06250 1.00045

This matches the TrackStar paper (arXiv 2410.17413 §A.1.2), which specifies the same two-sided
  per-block projection and states "Projection matrix entries are sampled i.i.d. from N(0, 1/d)" —
  variance 1/(number of rows), exactly A /= sqrt(m). It also matches docs/preprocessing.rst's claim
  that projections preserve inner products, which the old scaling did not.

Existing tests could not catch this: they compare the collector against L @ G @ R^T using the SAME
  matrices, so they are invariant to how those are scaled. test_global_projection_linearity uses one
  R sliced into blocks — the paper's construction, not the code's.
  tests/test_projection_inner_products.py now pins projected against unprojected (12 tests, all 12
  fail without this fix), including that the scale does not depend on module shape.

BREAKING: changes all stored index scales. Existing indices built with projection_dim > 0 must be
  regenerated, and any published numbers re-run. runs/test_build_cache.npy is regenerated here; the
  measured ratio is exactly 0.5000, matching theory (sqrt(o*i)/p = 8/16 for tiny-Phi3).

THREE TESTS FAIL ON THIS BRANCH AND ARE LEFT FAILING, PENDING A DECISION:

tests/test_batch_size_invariance.py::test_gradient_scale_invariance[100-100]
  tests/test_batch_size_invariance.py::test_gradient_scale_invariance[50-150] Relative error is
  UNCHANGED by this fix (6.83e-06 on main vs 7.58e-06 here) and already exceeds the assert_close
  default rtol=1.3e-06 on main. It passes on main only because the values are ~150x smaller, so the
  absolute difference stays under atol=1e-05. Restoring the correct scale trips atol. The test is
  magnitude-sensitive rather than wrong about the invariant.

tests/test_gradients.py::test_gradient_collector_proj_norm 1 of 256 elements, greatest relative
  difference 2.46e-04 against the test's explicit 1e-4 tolerance. Probably a near-cancelling element
  landing differently at the new scale — the test already carries a comment about accumulating
  numerical error — but this is NOT proven.

These are regression guards; loosening them is a call for a human, not something to do to make a
  branch green.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

* Make two projection tests independent of gradient magnitude

test_gradient_scale_invariance compared std separately-vs-combined with assert_close's defaults. The
  invariant is scale-free, but the default atol=1e-5 let a relative disagreement pass whenever the
  operands were small enough: on main the relative error is already 6.83e-06 against rtol=1.3e-06,
  and the test passes only because the values are ~150x smaller than the correctly-scaled ones. Use
  rtol=1e-4 with atol=0 so the check tracks the invariant rather than the magnitude.

test_gradient_collector_proj_norm compares A @ normalize(g) @ B.T against the collector, which
  projects activations and output grads separately. The two orderings are mathematically identical
  and differ only in fp32 rounding; at the corrected scale that noise reached 2.46e-04 relative on
  an O(1) element, over the existing 1e-4. Widen to 1e-3 and say what the tolerance is measuring.

* Add projection_scale to read indexes built with the old scaling

Indexes store their projected gradients, so changing the projection scaling makes existing ones
  unreadable at the correct weighting. projection_scale selects the convention: "jl" (default,
  variance 1/projection_dim) or "row_norm".

GradientProcessor.load resolves a missing projection_scale key to "row_norm", so existing indexes
  keep working with no user action, and new ones record "jl" in processor_config.yaml. Threaded
  through IndexConfig, create_processor and EkfacConfig, which builds matching matrices to compress
  the IVHP output.

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci

---------

Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>

Co-authored-by: pre-commit-ci[bot] <66853113+pre-commit-ci[bot]@users.noreply.github.com>


## v0.13.0 (2026-07-19)

### Features

- Compress K-FAC IVHP output to match compressed gradient stores
  ([`157fa1b`](https://github.com/EleutherAI/bergson/commit/157fa1b96b80b24648aabf593d3400f83a407666))

EKFAC's H^-1 must be applied to the full, unprojected gradient (the eigenbasis rotation only makes
  sense at the true parameter dimension), so apply_hessian.py's IVHP path and the query build step
  have always forced projection_dim=0. But the training-side gradient store built by `bergson
  build`/`bergson score` uses index_cfg.projection_dim's per-module Kronecker random projection by
  default, so the hessian_pipeline's scoring step had to force projection_dim=0 there too --
  silently ignoring whatever compression the user asked for and scoring against full dense
  gradients.

EkfacConfig gains projection_dim/projection_type; when set, EkfacApplicator.compute_ivhp_sharded
  compresses each module's H^-1 G output to [p, p] as a post-processing step, using the same
  create_projection_matrix identifiers (f"{name}/left" / f"{name}/right") that bergson build already
  uses on training gradients. The hessian_pipeline now threads
  index_cfg.projection_dim/projection_type into the apply step and no longer overrides the score
  step's projection_dim to 0, so a compressed query and a compressed training-side store are
  directly comparable end to end.

Supersedes EleutherAI/bergson#275, which predates the FactoredPreconditioner refactor and no longer
  applies; ports the same idea (compress the IVHP output rather than the pre-Hessian gradient) onto
  the current apply path.

Co-Authored-By: Girish Gupta <girish@girishgupta.com>

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>


## v0.12.1 (2026-07-18)

### Bug Fixes

- Correctly differentiate cross-rank gradient sync in MAGIC's backward-through-training
  ([`bfc2938`](https://github.com/EleutherAI/bergson/commit/bfc29381bc2436310ef0b554716177ee858711e4))

Trainer.step() synchronizes per-rank parameter gradients with a differentiable all-reduce so the
  resulting graph can be differentiated again during backward-through-training. Rebased onto main,
  which had independently landed a different fix for the same underlying bug (51a822dd, "Update DDP
  MAGIC all reduce") using torch.distributed.nn.functional.all_reduce plus a single `/world_size`
  correction applied once at the end of Trainer.backward().

This commit replaces that mechanism with a custom _ReplicatedAllReduceSum autograd Function,
  differentiable to arbitrary order via a recursive backward. Unlike torch.distributed.nn's
  all_reduce, this one already applies its `/world_size` correction inline (dividing before the
  reduce, once per occurrence), so the leftover unconditional `bwd_state.weight_grads /=
  dist.get_world_size()` at the end of `backward()` — carried over from main's fix, which does need
  it — was double-correcting and silently deflating scores by another 1/world_size. Confirmed via
  tests/test_ddp.py::test_ddp_matches_single_process, which should be exact with no cross-rank
  correction needed for a single step: it failed at exactly a 1/world_size ratio with the line still
  in place, and passes with it removed. Both CPU (test_ddp.py) and real multi-GPU
  (test_distributed_magic.py, FSDP-vs-DDP with and without grad clipping) suites pass after the fix.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>


## v0.12.0 (2026-07-18)

### Features

- Add mix_hessians toggle to TrackStar pipeline
  ([`08e9e23`](https://github.com/EleutherAI/bergson/commit/08e9e23876320b731c662b1466bfdbb4bdd7c617))

Lets TrackStar skip computing and mixing the query autocorrelation hessian (steps 2-3) and
  precondition with the value/train hessian alone. Useful when query and train data are IID and the
  query set is too small to estimate its own hessian reliably.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

### Refactoring

- Use invert_psd_matrix in semantic scoring examples
  ([`d005091`](https://github.com/EleutherAI/bergson/commit/d0050914b1d419c25ea768b98f78ab7e305c931d))

Updates example scripts to call bergson.hessians.inversion.invert_psd_matrix instead of the old
  bergson.utils.math.damped_psd_power, matching the already-landed inversion API. Where a custom
  regularizer was passed in, fold it into H + damping_factor*regularizer before inverting rather
  than passing it as a separate argument.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>


## v0.11.1 (2026-07-18)

### Bug Fixes

- Optimizer-state loading — orientation, HF param groups, PEFT, FSDP
  ([`342a216`](https://github.com/EleutherAI/bergson/commit/342a2160a07af76f7b5c31e6b221a9d7251f9b83))

Four fixes to the optimizer.pt loading used for attribution normalizers, plus matching export
  hardening:

- Orient second moments by module class (LayerAdapter.weight_transposed: HF Conv1D stores [in, out])
  instead of shape matching, which cannot detect transposed storage for SQUARE weights — GPT-2's
  attn.c_proj normalizers were silently transposed. - Map state indices group-aware: HF Trainer
  writes two decay/no-decay param groups whose index order differs from named_parameters(); the old
  positional mapping assigned moments to the wrong modules. Reconstructed with transformers' own
  utilities and shape-verified. - Apply the group-aware mapping to PEFT checkpoints too (HF Trainer
  splits LoRA params across the same groups). - Make save_second_moments_as_optimizer_pt
  collective-safe (FSDP DTensor moments are gathered by every rank; rank 0 writes — previously a
  rank-0 -only call would hang) and record each entry's canonical param_name, since FSDP wrapping
  renames and reorders parameters; readers prefer the recorded name. Optional standard
  step/betas/eps fields make snapshot exports self-describing. Model type widened to nn.Module.

Tests: square-Conv1D orientation, decay-split identity vs a real two-group AdamW, PEFT + two-group
  checkpoint, and the real-checkpoint round-trips now look entries up group-aware with shape
  assertions the old convention fails.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>


## v0.11.0 (2026-07-17)

### Features

- Custom gradient store class
  ([`3ae519c`](https://github.com/EleutherAI/bergson/commit/3ae519c57e45090ea153d15bc0ba24c97b8e0937))

Trim comments referencing the removed structured-dtype layout.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>


## v0.10.2 (2026-07-11)

### Bug Fixes

- Update bergson
  ([`ab4b933`](https://github.com/EleutherAI/bergson/commit/ab4b933083f52c8be144ebd25b2c7cb5c7d0a1f4))

### Documentation

- Add Known limitations section (MoE fused experts, FSDP host-RAM load)
  ([`290140a`](https://github.com/EleutherAI/bergson/commit/290140abbb605f97fddbf83d4839838199783dcd))

Document two issues we are not fixing now, each with the responsible source file:line so maintainers
  can locate them:

- MoE fused-parameter experts and router are bare nn.Parameters on custom modules, outside
  LayerAdapter.supported_modules (bergson/gradients.py:238; module walk in
  bergson/collector/collector.py:154-155), so they are silently skipped. Only attention projections
  and lm_head (~1-2% of params) are tracked. - Under FSDP the load path uses device_map="cpu" per
  rank (bergson/utils/worker_utils.py:166), so every rank replicates a full dequantized model in
  host RAM before sharding, which can OOM host memory independent of GPU VRAM.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_012NMtUnc12k72cAXd4uRMzN

- Compact MoE limitation, drop FSDP host-RAM limitation
  ([`1f573ba`](https://github.com/EleutherAI/bergson/commit/1f573ba0c9ef0ffb9f5e0b998704445b7203b8e3))

Trim the MoE fused-experts note to what users need (affected model families, consequence,
  legacy-layout caveat) and remove the FSDP host-RAM limitation, which is a straightforward fix
  pending coordination.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_0149bxnr7vz4KcR8HpT4ipgn


## v0.10.1 (2026-07-09)

### Bug Fixes

- Load mmap'd FAISS ANN indices on CPU without crashing
  ([`12e54ac`](https://github.com/EleutherAI/bergson/commit/12e54acdf8345405c06a16c5a4d52be7903642a7))

`index_to_device(index, "cpu")` unconditionally called `faiss.index_gpu_to_cpu`, which clones the
  index and raises `RuntimeError: clone not supported ... OnDiskInvertedLists` for any IVF/ANN index
  mmap'd from disk (the shipped `mmap_index=False` default). Only exact `Flat` survived. The CPU
  branch is now a no-op; the GPU->CPU conversion after a GPU build is done explicitly in
  `create_index` only when `device != "cpu"`.

Also expose the FAISS index config on the `query` CLI (nested `QueryConfig.faiss_cfg`) so ANN /
  `mmap_index=True` is actually reachable, and add a regression test for the on-disk ANN CPU load
  path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_012NMtUnc12k72cAXd4uRMzN

### Documentation

- Drop redundant inline comments at index_to_device call sites
  ([`2ebc293`](https://github.com/EleutherAI/bergson/commit/2ebc293fb936f88c4ec7987f354df5fc7685a2d4))

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

- Trim index_to_device docstring
  ([`5ecc0b0`](https://github.com/EleutherAI/bergson/commit/5ecc0b0f6b3531688ee060453ed258bab7f996eb))

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

### Refactoring

- Keep symmetric index_to_device, fix the bug at the __init__ call site
  ([`50345cf`](https://github.com/EleutherAI/bergson/commit/50345cf6d09ce87fac16224ba4242b479d7c264c))

Revert the index_to_gpu rename. index_to_device stays a symmetric CPU<->GPU move; the actual bug was
  only that `FaissIndex.__init__` routed an already-CPU mmap'd shard through it with `"cpu"`,
  hitting the `index_gpu_to_cpu` clone that crashes on OnDiskInvertedLists. The loader now guards
  `device != "cpu"` so an already-CPU shard is never moved, and `create_index` uses the same helper
  for its genuine GPU->CPU conversion (also guarded). No behavioural change on any tested path; the
  crash stays fixed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

- Make index_to_device self-guarding (no-op when already on target)
  ([`fc43de6`](https://github.com/EleutherAI/bergson/commit/fc43de6aa02c345f0aed41459eec48f99b5a6501))

Rather than requiring call sites to know "don't ask for cpu if it's already cpu", `index_to_device`
  now detects GPU residency (`_is_gpu_resident`: a direct GpuIndex, or the IndexShards/IndexReplicas
  container a multi-GPU move returns) and treats a CPU->CPU request as a no-op, returning the index
  unchanged instead of cloning via `index_gpu_to_cpu` (which crashes on mmap'd OnDiskInvertedLists).
  Both call sites (`FaissIndex.__init__`, `create_index`) drop their `device != "cpu"` guards and
  just call the helper.

Adds unit tests for the guard: it no-ops on an in-memory CPU index and on the mmap'd OnDisk index
  that used to crash (asserting the raw `index_gpu_to_cpu` still raises, so the regression case is
  real).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

- Rename index_to_device -> index_to_gpu, drop dead CPU branch
  ([`080bbf0`](https://github.com/EleutherAI/bergson/commit/080bbf0380e5f77ab9b62a1c5cc859a935b32e45))

The CPU destination in `index_to_device` was only ever a no-op after the prior fix:
  `FaissIndex.__init__` reads shards from disk (already CPU) and only ever needs to push *up* to a
  GPU, and the one genuine GPU->CPU move lives in `create_index`. Rename to `index_to_gpu` (no-op on
  "cpu") and guard the loader call with `device != "cpu"` so the direction is explicit and the
  OnDiskInvertedLists clone trap cannot be reintroduced. Also trims a verbose config docstring.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo

### Testing

- Cover the index_to_device op path, not just the no-op
  ([`64484d8`](https://github.com/EleutherAI/bergson/commit/64484d8e2968ad8eb7847ef35b32fa7b2ce50ba2))

Add a test that a GPU-resident index is actually converted (not returned as-is): an `IndexShards`
  container is what a multi-GPU move returns and what `_is_gpu_resident` flags for conversion, so
  bringing it to CPU must yield a new, non-container, still-searchable index. Complements the two
  no-op tests.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EeaHeis3YkmXwc5emPJnXo


## v0.10.0 (2026-06-05)

### Features

- Consolidate per-run YAMLs into one reproducible config.yaml
  ([`d33a6bb`](https://github.com/EleutherAI/bergson/commit/d33a6bb056a617e212f2d811187eefcde36cfdad))


## v0.9.1 (2026-04-10)

### Bug Fixes

- Preconditioner bug in attributor.py
  ([`6232d1e`](https://github.com/EleutherAI/bergson/commit/6232d1e3464ec4fc9056a5c45b5efc7b4c421318))


## v0.9.0 (2026-03-18)

### Bug Fixes

- Release
  ([`dec3df9`](https://github.com/EleutherAI/bergson/commit/dec3df98a0707f0058bf193c27ef4f4e50fab6ac))

### Features

- Add flag to enable TF32
  ([`35ab164`](https://github.com/EleutherAI/bergson/commit/35ab16400afda484ccff717b7a4b48ae6f06811d))


## v0.8.1 (2026-03-18)

### Bug Fixes

- Release bergson without pinned transformers
  ([`ef9dc9a`](https://github.com/EleutherAI/bergson/commit/ef9dc9a6bd4604162fcd9c1ba5bcca18f3936455))


## v0.8.0 (2026-03-08)

### Features

- Set default precision to fp32 in IndexConfig and ScoreConfig
  ([`92d4807`](https://github.com/EleutherAI/bergson/commit/92d4807df7b73cee21c6e375c79454b021998671))

Co-authored-by: Lucia Quirke <luciaquirke@users.noreply.github.com>


## v0.7.2 (2026-03-04)


## v0.7.1 (2026-03-03)

### Bug Fixes

- Always compute mixing coefficient in Trackstar pipeline
  ([`c990375`](https://github.com/EleutherAI/bergson/commit/c990375e69d309f348c489f9bfc9cf9cddc28f6d))

Remove the conditional guard — lambda is always auto-computed from the preconditioner eigenvalues
  since the cost is negligible.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.0 (2026-03-03)

### Bug Fixes

- Standardize trace collector preconditioning
  ([`6a14e53`](https://github.com/EleutherAI/bergson/commit/6a14e534a403c72bae4a340009ab84d385b7928b))

### Features

- Enable trackstar
  ([`2dd26d3`](https://github.com/EleutherAI/bergson/commit/2dd26d31fe4f88d1f2d19537958208b914cec2c8))


## v0.6.2 (2026-03-02)

### Bug Fixes

- Convert PyArrow Column to list in allocate_batches
  ([`7fe4dd3`](https://github.com/EleutherAI/bergson/commit/7fe4dd32181c5bc7ce5684e452bc442862e22e7f))

HuggingFace Dataset column access (ds["length"]) returns a PyArrow Column, not a Python list.
  Iterating over it element-by-element (via sorted(), random indexing) is ~1000x slower than on a
  native list. For 10M items this caused allocate_batches to hang for 13+ hours instead of
  completing in ~17 seconds.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Convert PyArrow columns to list at callsites of allocate_batches
  ([`5d734dc`](https://github.com/EleutherAI/bergson/commit/5d734dc23bb083819890ca17d1b44f377ae35d69))

Move the list conversion out of allocate_batches (which types doc_lengths as list[int]) to the
  callsites that pass HF Dataset columns. Use ds["length"][:] which returns a plain list[int].

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Remove redundant zero-fill loop in MemmapSequenceScoreWriter
  ([`558829f`](https://github.com/EleutherAI/bergson/commit/558829f717f8679d517765d5c3d9beac2f2249b2))

np.memmap w+ mode already creates a zero-filled file, making the per-field written flag
  initialization loop unnecessary. For large datasets (10M+ items) with many query scores, the
  strided writes through the structured dtype caused multi-hour hangs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Use [:] instead of list() for consistency
  ([`c76d131`](https://github.com/EleutherAI/bergson/commit/c76d131c357b6b8e7880da48b4640510ffe5a654))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.6.1 (2026-03-02)

### Bug Fixes

- Unpin transformers by explicitly setting float32 dtype in tests
  ([`0b6c226`](https://github.com/EleutherAI/bergson/commit/0b6c22615b7cce4ca62f71cb93847e3027fa68ba))

Transformers 4.56+ changed from_config() to honor the config's torch_dtype field, causing test
  models (tiny-GPTNeoX, tiny-Phi3) to be created in float16 instead of float32. This caused gradient
  comparison tests to fail from reduced precision, not from any actual change in gradient collection
  logic.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.6.0 (2026-02-17)

### Bug Fixes

- Use _csv._writer type for csv_recorder annotation
  ([`6e6289c`](https://github.com/EleutherAI/bergson/commit/6e6289c266b36304a6d79a35bb6b9fe3c35fa95a))

csv.writer is a function, not a class, so it cannot be used as a type annotation. Import the private
  _writer type from _csv and use it for the Generator yield type. Also fix the None check to use `if
  not path` since QueryConfig.record uses empty string as the sentinel value.

Co-authored-by: Lucia Quirke <luciaquirke@users.noreply.github.com>

### Continuous Integration

- Pin pyright version and fix faiss type error
  ([`b9f54cf`](https://github.com/EleutherAI/bergson/commit/b9f54cf9e7caf3c13af78f1a2d3d766f2055c3da))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Use Python 3.11 for typechecking
  ([`9ef4122`](https://github.com/EleutherAI/bergson/commit/9ef4122903eed2ecf496f803c5d1aba4c62295cb))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Use Python 3.11 for typechecking
  ([`ea50dd8`](https://github.com/EleutherAI/bergson/commit/ea50dd8ed9dc02b0f21ce7621f7d0ff53622ea87))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- Add --record flag to query CLI for saving results to CSV
  ([`59770ff`](https://github.com/EleutherAI/bergson/commit/59770ff88c5dbfffabd6ce0f51e5a56edbae2c0b))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- Replace try/finally CSV block with context manager
  ([`6431320`](https://github.com/EleutherAI/bergson/commit/6431320b7c167191b157b3fc53013818ecdd5135))

Co-authored-by: Lucia Quirke <luciaquirke@users.noreply.github.com>


## v0.5.2 (2026-02-17)

### Bug Fixes

- Pass batches to CollectorComputer in fit_normalizers
  ([`c95d5d4`](https://github.com/EleutherAI/bergson/commit/c95d5d498ad900af8a95902535fdfe740696088f))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Continuous Integration

- Improve Claude workflows (fetch-depth, timeout, max-turns, pip install)
  ([`7a315e5`](https://github.com/EleutherAI/bergson/commit/7a315e58758fac24f76400043eeac559380a2952))

- Run tests and typechecking in parallel
  ([`e690fc0`](https://github.com/EleutherAI/bergson/commit/e690fc0bed99ff5e705e8e82d790e961f3ceba33))


## v0.5.1 (2026-01-30)

### Bug Fixes

- Release
  ([`f0ad2be`](https://github.com/EleutherAI/bergson/commit/f0ad2bee12b0eb16f1c211a891b8bd78e89ea45e))


## v0.5.0 (2026-01-08)

### Features

- Add optimizer-aware gradients
  ([`497edab`](https://github.com/EleutherAI/bergson/commit/497edab8f2ca19d8fcb1d409fbd99452a929584e))


## v0.4.6 (2026-01-06)

### Bug Fixes

- Update build.yml
  ([`ba4cd5a`](https://github.com/EleutherAI/bergson/commit/ba4cd5ad49d36595c5ea063037eb832aa3a1a3b4))


## v0.4.5 (2026-01-06)

### Bug Fixes

- Always use unstructured gradients in score
  ([`595ed92`](https://github.com/EleutherAI/bergson/commit/595ed92deb06278f343a489f782e318916036eb2))


## v0.4.4 (2026-01-05)

### Bug Fixes

- Release bergson
  ([`c9040a6`](https://github.com/EleutherAI/bergson/commit/c9040a6dc12bea49b8f3e4bf8efbe82c92022bca))


## v0.4.3 (2026-01-05)

### Bug Fixes

- Release bergson
  ([`350dafe`](https://github.com/EleutherAI/bergson/commit/350dafe9c419ac3a874848a9d355af52de2407bb))


## v0.4.2 (2025-12-22)

### Bug Fixes

- Unit normalize in float32
  ([`cae8352`](https://github.com/EleutherAI/bergson/commit/cae8352c783cd68516ccab18a6746ba974455043))


## v0.4.1 (2025-12-20)

### Bug Fixes

- Pin transformers to avoid fp error bug
  ([`9feac20`](https://github.com/EleutherAI/bergson/commit/9feac20e237d66825a5d16c385e4174bb02f4705))


## v0.4.0 (2025-12-03)

### Features

- Enable specifying a custom tokenizer
  ([`9781a55`](https://github.com/EleutherAI/bergson/commit/9781a5538491aae3bf53af8247ae2509fe801b59))


## v0.3.0 (2025-12-03)

### Features

- Release bergson
  ([`64b5baf`](https://github.com/EleutherAI/bergson/commit/64b5baf4aa998c4e7573e24dcda939e74185c5f4))


## v0.2.0 (2025-11-13)

### Features

- Add on-the-fly queries
  ([`0ce0ee2`](https://github.com/EleutherAI/bergson/commit/0ce0ee2a0ec151f3fa0e6ee1eef3810408a54128))


## v0.1.1 (2025-10-16)

### Bug Fixes

- Simplify query
  ([`fd37173`](https://github.com/EleutherAI/bergson/commit/fd37173bf7c3d25daa6af065e7f261f2b774ce69))


## v0.1.0 (2025-10-16)

### Features

- Add on-the-fly queries
  ([`294661e`](https://github.com/EleutherAI/bergson/commit/294661e1d7ad7220917562991a1c7582b6181632))


## v0.0.0 (2025-10-07)
