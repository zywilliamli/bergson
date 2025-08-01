import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.fsdp import fully_shard
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .collection import collect_gradients, fit_normalizers
from .data import IndexConfig, allocate_batches, tokenize
from .gradients import GradientProcessor
from .utils import assert_type, get_layer_list


def worker(rank: int, world_size: int, cfg: IndexConfig, ds: Dataset | IterableDataset):
    torch.cuda.set_device(rank)

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    match cfg.precision:
        case "bf16":
            dtype = torch.bfloat16
        case "fp16":
            dtype = torch.float16
        case "fp32":
            dtype = torch.float32
        case "int4" | "int8":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        case other:
            raise ValueError(f"Unsupported precision: {other}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map={"": f"cuda:{rank}" if not cfg.fsdp else "cpu"},
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=cfg.precision == "int4",
                load_in_8bit=cfg.precision == "int8",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_storage=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if cfg.precision in ("int4", "int8")
            else None
        ),
        torch_dtype=dtype,
        revision=cfg.revision,
    )

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    if cfg.fsdp:
        # Shard each individual transformer layer
        for layer in get_layer_list(model):
            fully_shard(layer)

        # Shard the entire model
        fully_shard(model)

    # Check for PEFT adapters
    try:
        adapters = model.active_adapters()
    except ValueError:
        target_modules = None
    else:
        cfg.normalizer = "adam"
        cfg.reshape_to_square = True

        if rank == 0:
            print("PEFT model detected. Using Adam and reshape_to_square = True")

        target_modules = set()

        for adapter_name in adapters:
            state = model.get_adapter_state_dict(adapter_name)

            for name in state:
                prefix = name.removesuffix(".weight")
                name = prefix + "." + adapter_name

                try:
                    model.get_submodule(name)
                except AttributeError:
                    print(f"Adapter parameter '{name}' not found in the model.")

                target_modules.add(name.removeprefix("model."))

    if os.path.exists(cfg.processor_path):
        if rank == 0:
            print(f"Loading processor from '{cfg.processor_path}'")

        processor = GradientProcessor.load(
            cfg.processor_path,
            map_location=f"cuda:{rank}",
        )
    else:
        if cfg.normalizer != "none":
            # Evenly sample `stats_sample_size` examples to compute statistics
            if isinstance(ds, Dataset):
                if cfg.stats_sample_size is not None and cfg.stats_sample_size < len(
                    ds
                ):
                    stats_ds = ds.shuffle(seed=0).select(range(cfg.stats_sample_size))
                else:
                    stats_ds = ds
            else:
                if cfg.stats_sample_size is not None:
                    stats_iterable_ds = ds.shuffle(seed=0).take(cfg.stats_sample_size)
                    stats_ds = assert_type(
                        Dataset, Dataset.from_generator(lambda: iter(stats_iterable_ds))
                    )
                else:
                    stats_ds = assert_type(
                        Dataset, Dataset.from_generator(lambda: iter(ds))
                    )

            normalizers = fit_normalizers(
                model,
                stats_ds,
                batches=allocate_batches(stats_ds["length"][:], cfg.token_batch_size),
                kind=cfg.normalizer,
                target_modules=target_modules,
            )
        else:
            normalizers = {}

        processor = GradientProcessor(
            normalizers,
            fisher_fourth_root=cfg.fisher_fourth_root,
            projection_dim=cfg.projection_dim or None,
            reshape_to_square=cfg.reshape_to_square,
        )
        if rank == 0:
            processor.save(cfg.run_path)

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"][:], cfg.token_batch_size)
        collect_gradients(
            model,
            ds,
            processor,
            cfg.run_path,
            batches=batches,
            skip_preconditioners=cfg.skip_preconditioners,
            target_modules=target_modules,
        )
    else:
        # Convert each chunk to Dataset then collect their gradients
        buf, chunk_id = [], 0

        def flush():
            nonlocal buf, chunk_id
            if not buf:
                return
            sub_ds = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(sub_ds["length"], cfg.token_batch_size)
            collect_gradients(
                model,
                sub_ds,
                processor,
                os.path.join(cfg.run_path, f"chunk-{chunk_id:05d}"),
                batches=batches,
                skip_preconditioners=cfg.skip_preconditioners,
                target_modules=target_modules,
            )
            buf.clear()
            chunk_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == cfg.streaming_chunk_size:
                flush()
        flush()


def dist_worker(rank: int, world_size: int, cfg: IndexConfig, ds: Dataset):
    try:
        worker(rank, world_size, cfg, ds)
    finally:
        dist.destroy_process_group()


def build_gradient_dataset(cfg: IndexConfig):
    # Do all the data loading and preprocessing on the main process
    data_str = cfg.data.dataset
    if data_str.endswith(".csv"):
        ds = assert_type(Dataset, Dataset.from_csv(data_str))
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = assert_type(Dataset, Dataset.from_json(data_str))
    else:
        try:
            ds = load_dataset(data_str, split="train", streaming=cfg.streaming)

            if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
                raise NotImplementedError(
                    "DatasetDicts and IterableDatasetDicts are not supported."
                )
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(data_str, keep_in_memory=False)
            else:
                raise e

    remove_columns = ds.column_names if cfg.drop_columns else None

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, model_max_length=cfg.token_batch_size, revision=cfg.revision
    )
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer),
        remove_columns=remove_columns,
    )

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, cfg, ds)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = start_processes(
            "build",
            dist_worker,
            args={i: (i, world_size, cfg, ds) for i in range(world_size)},
            envs={
                i: {
                    "LOCAL_RANK": str(i),
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(port),
                }
                for i in range(world_size)
            },
            logs_specs=DefaultLogsSpecs(),
        )
        ctx.wait()
