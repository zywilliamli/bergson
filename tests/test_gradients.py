import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from quelle.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientCollector,
    ProjectionGenerator,
)


def test_phi3():
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(config)

    # It's important that we use a batch size of one so that we can simply use the
    # aggregate gradients from the backward itself and compare against those
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    inputs = dict(input_ids=tokens, labels=tokens)

    with GradientCollector(model) as collector:
        model(**inputs).loss.backward()

    adafactors: dict[str, AdafactorNormalizer] = {}
    adams: dict[str, AdamNormalizer] = {}

    # Go through the motions of what GradientCollector does, but after the fact
    generator = ProjectionGenerator(model.device, collector.seed)
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue

        o, i = layer.out_features, layer.in_features
        A, B = generator.next_projection(collector.p, collector.q, o, i)

        g = layer.weight.grad
        assert g is not None

        proj_grad = A @ g @ B.T
        collected_grad = collector._buffers[name].squeeze(0)
        torch.testing.assert_close(proj_grad, collected_grad)

        # Store normalizers for this layer
        moments = g.square()
        adams[name] = AdamNormalizer(moments)
        adafactors[name] = adams[name].to_adafactor()
