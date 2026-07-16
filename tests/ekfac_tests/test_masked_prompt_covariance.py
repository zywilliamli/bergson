"""Regression test: KFAC covariance accumulators must include masked/prompt
positions, not just the loss-masked completion.

``g_t`` (the gradient of the loss w.r.t. a module's output at position ``t``)
is generally nonzero at prompt positions too, because completion-token losses
backprop through them via causal attention. The activation/gradient
covariances (A_cov, S_cov) that KFAC/EKFAC use to approximate the Fisher/GGN
should therefore be accumulated over every real (non-padding) position except
the last, exactly like the per-token gradient rows tested in
``tests/test_attribute_tokens.py::test_masked_prompt_token_grads_cover_all_positions``.
"""

import pytest
import torch
import torch.nn.functional as F

from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import collect_hessians
from bergson.utils.utils import get_device
from tests.ekfac_tests.test_utils import load_sharded_covariances


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kfac_covariance_covers_all_positions(tmp_path, model):
    """A_cov/S_cov must match a reference computed over every real position
    (prompt + completion), not just the completion the loss is masked to."""
    from datasets import Dataset

    device = torch.device(get_device(0))
    model = model.float().to(device)

    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Leading 6 positions are prompt (masked), trailing 4 are the completion.
    labels = [-100, -100, -100, -100, -100, -100, 7, 8, 9, 10]
    length = len(input_ids)

    ds = Dataset.from_dict(
        {
            "input_ids": [input_ids],
            "labels": [labels],
            "length": [length],
        }
    )

    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    index_cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        loss_reduction="sum",
        include_bias=False,
    )
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    hessian_cfg = HessianConfig(
        method="kfac", hessian_dtype="fp32", use_dataset_labels=True
    )

    collect_hessians(
        model=model,
        data=ds,
        index_cfg=index_cfg,
        batches=[[0]],
        target_modules=target_modules,
        hessian_cfg=hessian_cfg,
    )

    a_cov = load_sharded_covariances(index_cfg.partial_run_path / "activation_sharded")
    s_cov = load_sharded_covariances(index_cfg.partial_run_path / "gradient_sharded")

    # --- Independent reference: capture g (output grad) and a (input
    # activation) for every target module in a single backward, matching
    # fwd_bwd_hessian_factory's use_dataset_labels=True loss exactly.
    x = torch.tensor([input_ids], device=device)
    y = torch.tensor([labels], device=device)

    cap: dict[str, dict] = {}
    handles = []
    for n in target_modules:
        m = model.base_model.get_submodule(n)
        handles.append(
            m.register_forward_hook(
                lambda _mod, inp, _out, n=n: cap.setdefault(n, {}).update(
                    a=inp[0].detach()
                )
            )
        )
        handles.append(
            m.register_full_backward_hook(
                lambda _mod, _gi, go, n=n: cap.setdefault(n, {}).update(
                    g=go[0].detach()
                )
            )
        )
    model.zero_grad(set_to_none=True)
    logits = model(x).logits[:, :-1]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y[:, 1:].reshape(-1),
        reduction="none",
    ).reshape_as(y[:, 1:])
    losses.sum(1).sum().backward()  # loss_reduction="sum" -> denom=1.0
    for h in handles:
        h.remove()

    max_prompt_frac = 0.0
    for n in target_modules:
        a_full = cap[n]["a"][0].double()  # [S, I]
        g_full = cap[n]["g"][0].double()  # [S, O]

        # Every real position but the last (length - 1 rows), independent of
        # the completion mask.
        a_ref = a_full[: length - 1]
        g_ref = g_full[: length - 1]
        a_cov_ref = a_ref.mT @ a_ref
        s_cov_ref = g_ref.mT @ g_ref

        # Completion-only reference, for the "prompt really contributes" check.
        vmask = torch.zeros(length, dtype=torch.bool, device=device)
        vmask[:-1] = y[0, 1:] != -100
        g_comp = g_full * vmask.unsqueeze(-1)
        s_cov_comp = g_comp.mT @ g_comp

        torch.testing.assert_close(
            a_cov[n].double().cpu(),
            a_cov_ref.cpu(),
            atol=1e-3,
            rtol=1e-3,
            msg=f"module {n}: activation covariance must cover all real "
            f"positions, not just the completion",
        )
        torch.testing.assert_close(
            s_cov[n].double().cpu(),
            s_cov_ref.cpu(),
            atol=1e-3,
            rtol=1e-3,
            msg=f"module {n}: gradient covariance must cover all real "
            f"positions, not just the completion",
        )
        max_prompt_frac = max(
            max_prompt_frac,
            (
                (s_cov_ref.cpu() - s_cov_comp.cpu()).norm()
                / s_cov_ref.cpu().norm().clamp_min(1e-12)
            ).item(),
        )

    # Prompt positions genuinely contribute, so the fix is materially
    # different from the old completion-only behavior.
    assert max_prompt_frac > 0.1, (
        f"expected a substantial prompt-position contribution to the gradient "
        f"covariance, got max fraction {max_prompt_frac:.4f}"
    )
