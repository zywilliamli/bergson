from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

dataset = load_from_disk("../indices/facts_dataset.hf")
assert isinstance(dataset, Dataset), "Expected a Hugging Face Dataset"

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
tokenizer.chat_template = "{{ fact }}"

peft_cfg = LoraConfig(
    r=1,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
training_args = SFTConfig(
    dataset_text_field="fact",
    output_dir="finetune",
    label_names=["fact"],
)
trainer = SFTTrainer(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_cfg,
)
trainer.train()
