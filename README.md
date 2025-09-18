# Bergson
This library enables you to trace the memory of deep neural nets with gradient-based data attribution techniques. We currently focus on TrackStar, as described in [Scalable Influence and Fact Tracing for Large Language Model Pretraining](https://arxiv.org/abs/2410.17413v3) by Chang et al. (2024), although we plan to add support for other methods inspired by influence functions in the near future.

We view attribution as a counterfactual question: **_If we "unlearned" this training sample, how would the model's behavior change?_** This formulation ties attribution to some notion of what it means to "unlearn" a training sample. Here we focus on a very simple notion of unlearning: taking a gradient _ascent_ step on the loss with respect to the training sample. To mimic the behavior of popular optimizers, we precondition the gradient using Adam or Adafactor-style estimates of the second moments of the gradient.

# Installation

We're not yet on PyPI, but you can `git clone` the repo and install it as a package using pip:

```bash
git clone https://github.com/EleutherAI/bergson.git
cd bergson
pip install .
```

# Usage
The first step is to build an index of gradients for each training sample. You can do this from the command line, using `bergson` as a CLI tool:

```bash
bergson <output_path> --model <model_name> --dataset <dataset_name>
```

This will create a directory at `<output_path>` containing the gradients for each training sample in the specified dataset. The `--model` and `--dataset` arguments should be compatible with the Hugging Face `transformers` library. By default it assumes that the dataset has a `text` column, but you can specify other columns using `--prompt_column` and optionally `--completion_column`. The `--help` flag will show you all available options.

You can also use the library programmatically to build the index. The `collect_gradients` function is just a bit lower level the CLI tool, and allows you to specify the model and dataset directly as arguments. The result is a HuggingFace dataset which contains a handful of new columns, including `gradients`, which contains the gradients for each training sample. You can then use this dataset to compute attributions.

At the lowest level of abstraction, the `GradientCollector` context manager allows you to efficiently collect gradients for _each individual example_ in a batch during a backward pass, simultaneously randomly projecting the gradients to a lower-dimensional space to save memory. If you use Adafactor normalization, which is the default, we will do this in a very compute-efficient way which avoids computing the full gradient for each example before projecting it to the lower dimension. There are two main ways you can use `GradientCollector`:

1. Using a `closure` argument, which enables you to make use of the per-example gradients immediately after they are computed, during the backward pass. If you're computing summary statistics or other per-example metrics, this is the most efficient way to do it.
2. Without a `closure` argument, in which case the gradients are collected and returned as a dictionary mapping module names to batches of gradients. This is the simplest and most flexible approach but is a bit more memory-intensive.

## Training Gradients

Gradient collection during trainig is supported for HuggingFace's Trainer and SFTTrainer. Training gradients are saved in the original order corresponding to their dataset items, and when the `track_order` flag is set the training steps associated with each training item are separately saved.

```python
from bergson import GradientCollectorCallback, prepare_for_gradient_collection

callback = GradientCollectorCallback(
    path="runs/example",
    track_order=True,
    use_optimizer_state=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    callbacks=[callback],
)
trainer = prepare_for_gradient_collection(trainer)
trainer.train()
```

## Attention Head Gradients

By default Bergson collects gradients for named parameter matrices, but gradients for individual attention heads within a named matrix can be collected to. To collect head gradients add a `head_cfgs` dictionary to the training calllback or static index config.

```python
from bergson import HeadConfig, IndexConfig, DataConfig
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("RonenEldan/TinyStories-1M", trust_remote_code=True, use_safetensors=True)

...

collect_gradients(
    model=model,
    data=data,
    processor=processor,
    path="runs/example_with_heads",
    head_cfgs={
        "h.0.attn.attention.out_proj": HeadConfig(num_heads=16, head_size=4, head_dim=2),
    },
)
```

## GRPO

Where a reward signal is available we compute gradients using a weighted advantage estimate based on Dr. GRPO:

```bash
bergson <output_path> --model <model_name> --dataset <dataset_name> --reward_column <reward_column_name>
```

# Development

```bash
pip install -e .[dev]
pytest
```
