from enum import Enum
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import parse_hf_uri
from peft import PeftModel, get_peft_model_state_dict
from transformers import PreTrainedModel

from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    LayerAdapter,
    Normalizer,
)


def load_optimizer(optimizer_state: str) -> dict:
    """Load an optimizer state dict from a local path or Hugging Face URI.

    ``optimizer_state`` may be:

    - a local file: loaded directly.
    - a local directory: ``optimizer.pt`` inside it is loaded.
    - a Hugging Face URI ``hf://<repo>[@<revision>][/<path>]``. The path
      is treated as a file when it ends in ``.pt``/``.pth`` and otherwise
      as a directory containing ``optimizer.pt``. An omitted/empty path
      resolves to ``optimizer.pt`` at the repo root.
    """
    if optimizer_state.startswith("hf://"):
        uri = parse_hf_uri(optimizer_state)
        if uri.path_in_repo.endswith((".pt", ".pth")):
            filename = uri.path_in_repo
        elif uri.path_in_repo:
            filename = f"{uri.path_in_repo.rstrip('/')}/optimizer.pt"
        else:
            filename = "optimizer.pt"
        path = Path(
            hf_hub_download(
                repo_id=uri.id,
                filename=filename,
                revision=uri.revision,
                repo_type=uri.type,
            )
        )
    else:
        local = Path(optimizer_state)
        if not local.exists():
            raise FileNotFoundError(
                f"Optimizer state '{optimizer_state}' is not a local path "
                f"and does not start with 'hf://'."
            )
        path = local / "optimizer.pt" if local.is_dir() else local

    return torch.load(path, map_location="cpu", weights_only=False)


class OptimizerStateFormat(Enum):
    """Optimizer state format for a single module - Adafactor-style factored
    optimizer states (e.g. 2D+ modules in Adafactor optimizers), or Adam-style
    unfactored states."""

    UNFACTORED = 1
    FACTORED = 2


def get_optimizer_state_format(param_state) -> OptimizerStateFormat | None:
    if not isinstance(param_state, dict):
        return None

    if "exp_avg_sq" in param_state:
        return OptimizerStateFormat.UNFACTORED

    if "exp_avg_sq_row" in param_state:
        return OptimizerStateFormat.FACTORED

    bnb_state = param_state.get("__bnb_optimizer_quant_state__")
    if isinstance(bnb_state, dict) and "state2" in bnb_state:
        # 8-bit Adam
        return OptimizerStateFormat.UNFACTORED

    return None


def get_unfactored_second_moment(state: dict) -> torch.Tensor:
    """Return the second moment tensor for an unfactored optimizer state.

    Adam and 8-bit Adam always use unfactored tensors.
    Adafactor has multiple factored moment tensors for 2D+ parameters,
    and unfactored tensors for 1D parameters.
    """
    if "exp_avg_sq" in state:
        return state["exp_avg_sq"]
    return state["__bnb_optimizer_quant_state__"]["state2"]


def _base_model_prefix(model) -> str:
    """Return the dotted path of ``model.base_model`` within ``model`` (with a
    trailing dot), or ``""`` if the base model is the model itself.

    Collection builds its collector from ``model.base_model``, so module names
    seen during collection are relative to it (GPT-2: ``h.0.attn.c_attn``, not
    ``transformer.h.0.attn.c_attn``). Normalizer keys must match.
    """
    base = getattr(model, "base_model", model)
    if base is model:
        return ""
    for name, module in model.named_modules():
        if module is base:
            return f"{name}." if name else ""
    return ""


def _orient_weight_second_moment(exp_avg_sq, model, layer_name):
    """Return ``exp_avg_sq`` in the collector's ``[out, in]`` orientation.

    The optimizer stores the second moment in the parameter's own layout, which
    is ``[out, in]`` for ``nn.Linear`` but ``[in, out]`` for HF ``Conv1D`` (GPT-2
    attn/mlp). The collector always feeds ``AdamNormalizer`` a ``[out, in]``
    gradient, so a Conv1D moment must be transposed or the divide broadcasts
    against the wrong axis. Orient by matching the module's out/in sizes.
    """
    try:
        module = model.get_submodule(layer_name)
        o = getattr(module, LayerAdapter.out_attr(module))
        i = getattr(module, LayerAdapter.in_attr(module))
    except (AttributeError, ValueError):
        return exp_avg_sq  # unknown module; leave as-is
    if tuple(exp_avg_sq.shape) == (o, i):
        return exp_avg_sq
    if tuple(exp_avg_sq.shape) == (i, o):
        return exp_avg_sq.T.contiguous()
    return exp_avg_sq


def get_normalizers(
    optimizer_state,
    target_param_index_to_name,
    target_modules,
    adapter_suffix,
    include_bias,
    device,
    base_prefix="",
    model=None,
    eps_root=0.0,
):
    normalizers: dict[str, Normalizer] = {}
    for param_idx, state in optimizer_state["state"].items():
        param_idx = int(param_idx)
        if param_idx not in target_param_index_to_name:
            continue

        param_name = target_param_index_to_name[param_idx]
        if not param_name.endswith(".weight"):
            continue

        layer_name = param_name.removesuffix(".weight")
        # Normalizers are looked up during collection by the module's name
        # *relative to ``model.base_model``* (see bergson/collection.py, which
        # builds the collector from ``model.base_model``). The optimizer state,
        # however, is keyed off the full model's parameter names, so strip the
        # base-model prefix (e.g. "transformer." for GPT-2) or the lookup silently
        # misses and no normalization is applied.
        module_name = layer_name.removeprefix("base_model.").removeprefix(base_prefix)
        module_name = module_name + adapter_suffix

        if target_modules is not None and module_name not in target_modules:
            continue

        optimizer_format = get_optimizer_state_format(state)

        if optimizer_format is None:
            print("Unrecognized format, skipping normalizer for param_idx", param_idx)
            continue

        bias_exp_avg_sq = _get_bias_second_moment(
            layer_name, target_param_index_to_name, optimizer_state, include_bias
        )
        bias_on_device = (
            bias_exp_avg_sq.to(device) if bias_exp_avg_sq is not None else None
        )

        if optimizer_format == OptimizerStateFormat.UNFACTORED:
            exp_avg_sq = get_unfactored_second_moment(state)
            if exp_avg_sq.ndim != 2:
                continue
            if model is not None:
                exp_avg_sq = _orient_weight_second_moment(exp_avg_sq, model, layer_name)
            normalizers[module_name] = AdamNormalizer(
                weight_avg_sq=exp_avg_sq.to(device),
                bias_avg_sq=bias_on_device,
            )
        elif optimizer_format == OptimizerStateFormat.FACTORED:
            row = state["exp_avg_sq_row"]
            col = state.get("exp_avg_sq_col")
            if row.ndim != 1 or col is None:
                continue
            normalizers[module_name] = AdafactorNormalizer(
                row=row.to(device),
                col=col.to(device),
                bias_avg_sq=bias_on_device,
            )

    return normalizers


def load_from_optimizer(
    model: PreTrainedModel | PeftModel,
    optimizer_state: str,
    include_bias: bool = False,
    target_modules: set[str] | None = None,
) -> dict[str, Normalizer]:
    """Load optimizer second moments from a checkpoint and create normalizer
    instances for each target linear layer.

    Auto-detects the optimizer format:

    - Adam/AdamW: ``exp_avg_sq`` -> AdamNormalizer
    - Adafactor: ``exp_avg_sq_row``/``exp_avg_sq_col`` -> AdafactorNormalizer
    - 8-bit Adam (BitsAndBytes): ``state2`` -> AdamNormalizer

    Args:
        model: The model whose parameter names are used to map optimizer
            state indices to layer names.
        optimizer_state: Local path to an optimizer state file or a
            checkpoint directory containing ``optimizer.pt``, or a Hugging
            Face URI ``hf://<repo>[@<revision>][/<path>]`` (see
            :func:`load_optimizer`).
        include_bias: Whether to include bias second moments.
        target_modules: Optional set of module names to include. If ``None``,
            all linear layers are included.

    Returns:
        Dictionary mapping layer names to normalizer instances.
    """
    optimizer_state_dict = load_optimizer(optimizer_state)

    # The optimizer state is keyed by position in the trainable parameter list.
    # For PEFT checkpoints, only include PEFT params.
    adapter_suffix = ""
    base_prefix = ""
    if isinstance(model, PeftModel):
        st = get_peft_model_state_dict(model)
        params_for_index = list(st.items())
        # peft serializes LoRA keys without the active adapter name (e.g.
        # ``...lora_A.weight``), but extract_peft_target_modules and the
        # actual submodule paths include it (``...lora_A.default``). Append
        # the adapter name so module_name lookups match target_modules.
        adapters = list(model.peft_config.keys())
        if len(adapters) == 1:
            adapter_suffix = "." + adapters[0]
    else:
        params_for_index = list(model.named_parameters())
        # Collection runs on ``model.base_model``, so normalizer keys must be
        # relative to it (e.g. drop GPT-2's "transformer." prefix).
        base_prefix = _base_model_prefix(model)

    target_param_index_to_name: dict[int, str] = {}
    for idx, (name, _param) in enumerate(params_for_index):
        target_param_index_to_name[idx] = name

    device = next(model.parameters()).device

    normalizers = get_normalizers(
        optimizer_state_dict,
        target_param_index_to_name,
        target_modules,
        adapter_suffix,
        include_bias,
        device,
        base_prefix,
        model,
    )
    assert normalizers, (
        f"No optimizer second moments found in '{optimizer_state}'. "
        "Ensure the checkpoint was saved from an Adam-family or Adafactor optimizer."
    )

    types = {type(n).__name__ for n in normalizers.values()}
    print(
        f"Loaded {len(normalizers)} normalizers ({', '.join(types)}) "
        f"from '{optimizer_state}'"
    )
    return normalizers


def save_second_moments_as_optimizer_pt(
    model: PreTrainedModel | PeftModel,
    opt_state,
    path: str | Path,
) -> int:
    """Export a torchopt AdamW ``opt_state`` to a PyTorch ``optimizer.pt``.

    Writes ``{"state": {idx: {"exp_avg_sq": nu}}, "param_groups": [...]}`` where
    ``idx`` indexes ``model.named_parameters()`` (deduplicated) exactly as
    :func:`load_from_optimizer` reads it, so the file round-trips into
    attribution normalizers.

    The mapping is done **by name**, not by position, because torchopt stores
    ``nu`` as a flat list in optree's sorted-key order of the params dict passed
    to ``optimizer.init`` -- which is neither insertion order nor
    ``named_parameters()`` order, and differs in length from the deduplicated
    ``named_parameters()`` when weights are tied (e.g. GPT-2 ``lm_head``/``wte``).
    Zipping the wrong orders is the classic module-name mismatch, so we align
    strictly by name and assert per-tensor shape agreement.

    Returns the number of weight second moments written.
    """
    # Must mirror Trainer.initialize: name-keyed, duplicates kept, trainable only.
    params = {
        k: v
        for k, v in model.named_parameters(remove_duplicate=False)
        if v.requires_grad
    }
    # optree flattens a dict by sorted keys; torchopt's nu list follows that.
    ordered_names = sorted(params)

    adam_state = next((s for s in opt_state if hasattr(s, "nu")), None)
    if adam_state is None:
        raise ValueError(
            "opt_state has no second-moment (`nu`) field; "
            "save_optimizer_state supports AdamW optimizers only."
        )
    nu = list(adam_state.nu)
    assert len(nu) == len(ordered_names), (
        f"nu has {len(nu)} entries but the trainable params dict has "
        f"{len(ordered_names)}; ordering assumptions are violated."
    )

    def _to_cpu(t):
        t = t.full_tensor() if hasattr(t, "full_tensor") else t
        return t.detach().to("cpu")

    name_to_nu = {}
    for name, moment in zip(ordered_names, nu):
        if moment is None:  # e.g. Muon leaves 2D params without a second moment
            continue
        assert tuple(moment.shape) == tuple(params[name].shape), (
            f"second-moment shape {tuple(moment.shape)} != param shape "
            f"{tuple(params[name].shape)} for '{name}' -- nu/name misalignment."
        )
        name_to_nu[name] = _to_cpu(moment)

    state: dict[int, dict] = {}
    param_ids: list[int] = []
    for idx, (name, _param) in enumerate(model.named_parameters()):
        if name in name_to_nu:
            state[idx] = {"exp_avg_sq": name_to_nu[name]}
            param_ids.append(idx)

    optimizer_pt = {"state": state, "param_groups": [{"params": param_ids}]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer_pt, path)
    print(f"Saved {len(state)} optimizer second moments to '{path}'")
    return len(state)


def _get_bias_second_moment(
    layer_name: str,
    param_index_to_name: dict[int, str],
    optimizer_state: dict,
    include_bias: bool,
) -> torch.Tensor | None:
    """Look up bias exp_avg_sq for a layer, if present and requested."""
    if not include_bias:
        return None

    bias_name = layer_name + ".bias"
    for idx, name in param_index_to_name.items():
        if name == bias_name:
            bias_state = optimizer_state["state"].get(idx)
            optimizer_format = get_optimizer_state_format(bias_state)
            if optimizer_format == OptimizerStateFormat.UNFACTORED:
                return get_unfactored_second_moment(bias_state)
            return None

    return None
