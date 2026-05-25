import json
from pathlib import Path

import torch
from safetensors.torch import safe_open
from transformers import AutoConfig, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def print_title(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def find_snapshot_dir(model_name: str) -> Path:
    cache_dir = Path("/workspace") / ".cache" / "huggingface" / "hub"
    repo_dir_name = "models--" + model_name.replace("/", "--")
    repo_dir = cache_dir / repo_dir_name

    if not repo_dir.exists():
        raise FileNotFoundError(f"模型缓存目录不存在: {repo_dir}")

    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"snapshots 目录不存在: {snapshots_dir}")

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"没有找到 snapshot: {snapshots_dir}")

    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0]


def print_config(model_path: Path):
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    print_title("CONFIG")

    fields = [
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "tie_word_embeddings",
        "torch_dtype",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    ]

    for name in fields:
        print(f"{name:28s}: {getattr(config, name, None)}")

    return config


def print_tokenizer(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    print_title("TOKENIZER")

    print(f"tokenizer class          : {tokenizer.__class__.__name__}")
    print(f"vocab size               : {len(tokenizer)}")
    print(f"bos_token                : {repr(tokenizer.bos_token)}")
    print(f"bos_token_id             : {tokenizer.bos_token_id}")
    print(f"eos_token                : {repr(tokenizer.eos_token)}")
    print(f"eos_token_id             : {tokenizer.eos_token_id}")
    print(f"pad_token                : {repr(tokenizer.pad_token)}")
    print(f"pad_token_id             : {tokenizer.pad_token_id}")

    text = "你好"
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()

    print()
    print(f"test text                : {text}")
    print(f"input ids                : {ids}")
    print(f"tokens                   : {tokenizer.convert_ids_to_tokens(ids)}")

    return tokenizer


def list_safetensors_files(model_path: Path):
    files = sorted(model_path.glob("*.safetensors"))

    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        files = sorted({model_path / name for name in weight_map.values()})

    if not files:
        raise FileNotFoundError(f"没有找到 safetensors 权重文件: {model_path}")

    return files


def print_weight_files(model_path: Path):
    print_title("WEIGHT FILES")

    files = list_safetensors_files(model_path)

    for f in files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"{f.name:50s} {size_mb:10.2f} MB")

    return files


def collect_weights(files):
    weights = []

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights.append(
                    {
                        "name": key,
                        "shape": tuple(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "file": file.name,
                    }
                )

    weights.sort(key=lambda x: x["name"])
    return weights


def print_all_weights(weights):
    print_title("ALL WEIGHTS")

    for w in weights:
        print(
            f"{w['name']:70s} "
            f"{str(w['shape']):28s} "
            f"{w['dtype']:12s} "
            f"{w['file']}"
        )

    print()
    print(f"total tensors: {len(weights)}")


def print_layer0(weights):
    print_title("LAYER 0 WEIGHTS")

    for w in weights:
        if w["name"].startswith("model.layers.0."):
            print(
                f"{w['name']:70s} "
                f"{str(w['shape']):28s} "
                f"{w['dtype']:12s}"
            )


def print_key_groups(weights):
    print_title("KEY GROUPS")

    groups = {
        "embedding": ["embed_tokens"],
        "attention": ["self_attn"],
        "mlp": ["mlp"],
        "norm": ["norm"],
        "lm_head": ["lm_head"],
    }

    for group_name, keywords in groups.items():
        count = sum(
            any(k in w["name"] for k in keywords)
            for w in weights
        )
        print(f"{group_name:16s}: {count}")


def print_expected_shapes(config):
    print_title("EXPECTED CORE SHAPES")

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None)

    if head_dim is None:
        head_dim = hidden_size // num_heads

    print(f"hidden_size              : {hidden_size}")
    print(f"intermediate_size        : {intermediate_size}")
    print(f"num_attention_heads      : {num_heads}")
    print(f"num_key_value_heads      : {num_kv_heads}")
    print(f"head_dim                 : {head_dim}")

    print()
    print("Typical Qwen2 decoder layer shapes:")
    print(f"q_proj.weight            : ({num_heads * head_dim}, {hidden_size})")
    print(f"k_proj.weight            : ({num_kv_heads * head_dim}, {hidden_size})")
    print(f"v_proj.weight            : ({num_kv_heads * head_dim}, {hidden_size})")
    print(f"o_proj.weight            : ({hidden_size}, {num_heads * head_dim})")
    print(f"gate_proj.weight         : ({intermediate_size}, {hidden_size})")
    print(f"up_proj.weight           : ({intermediate_size}, {hidden_size})")
    print(f"down_proj.weight         : ({hidden_size}, {intermediate_size})")


def main():
    model_path = find_snapshot_dir(MODEL_NAME)

    print_title("MODEL PATH")
    print(model_path)

    config = print_config(model_path)
    print_tokenizer(model_path)

    files = print_weight_files(model_path)
    weights = collect_weights(files)

    print_expected_shapes(config)
    print_key_groups(weights)
    print_layer0(weights)
    print_all_weights(weights)


if __name__ == "__main__":
    main()