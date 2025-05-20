import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from quelle import estimate_preconditioner, estimate_second_moments
from quelle.data import MemmapDataset

dist.init_process_group("nccl")

rank = dist.get_rank()
torch.cuda.set_device(rank)


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-160m", device_map={"": f"cuda:{rank}"}
)

dataset = MemmapDataset("/mnt/ssd-1/pile_preshuffled/standard/document.bin", 2049)
dataset = dataset.shard(dist.get_world_size(), rank)

if rank == 0:
    print("Estimating second moments...")

out = estimate_second_moments(model, dataset, num_examples=1000)
torch.save(out, "second_moments.pth")

if rank == 0:
    print("Estimating preconditioner...")

# We need a lot of examples for the preconditioner
preconditioner = estimate_preconditioner(model, dataset, out, num_examples=10_000)
torch.save(preconditioner, "preconditioner.pth")
