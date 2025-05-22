import torch.distributed as dist

dist.init_process_group("nccl")
d = dist.get_rank()
print(d)

# artificially wait for 5 seconds

import time

time.sleep(10)


dist.destroy_process_group()
