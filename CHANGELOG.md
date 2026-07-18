# CHANGELOG


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
