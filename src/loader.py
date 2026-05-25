import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import safe_open
from transformers import AutoConfig, AutoTokenizer


class QwenLoader:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        model_path: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        if model_path is None:
            self.model_path = self._find_snapshot_dir(model_name)
        else:
            self.model_path = Path(model_path)

        self.config = None
        self.tokenizer = None
        self.weight_files: List[Path] = []
        self.weight_map: Dict[str, Path] = {}

    def _find_snapshot_dir(self, model_name: str) -> Path:
        cache_dir = Path("/workspace") / ".cache" / "huggingface" / "hub"
        repo_dir = cache_dir / ("models--" + model_name.replace("/", "--"))

        if not repo_dir.exists():
            raise FileNotFoundError(f"找不到模型缓存目录: {repo_dir}")

        snapshots_dir = repo_dir / "snapshots"

        if not snapshots_dir.exists():
            raise FileNotFoundError(f"找不到 snapshots 目录: {snapshots_dir}")

        snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]

        if not snapshots:
            raise FileNotFoundError(f"snapshots 目录为空: {snapshots_dir}")

        snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return snapshots[0]

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        return self.config

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        return self.tokenizer

    def scan_weight_files(self):
        index_file = self.model_path / "model.safetensors.index.json"

        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)

            raw_weight_map = index["weight_map"]

            self.weight_map = {
                name: self.model_path / filename
                for name, filename in raw_weight_map.items()
            }

            self.weight_files = sorted(set(self.weight_map.values()))
            return self.weight_files

        files = sorted(self.model_path.glob("*.safetensors"))

        if not files:
            raise FileNotFoundError(f"找不到 safetensors 权重文件: {self.model_path}")

        self.weight_files = files

        for file in files:
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.weight_map[key] = file

        return self.weight_files

    def load_tensor(
        self,
        name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if not self.weight_map:
            self.scan_weight_files()

        if name not in self.weight_map:
            raise KeyError(f"找不到权重: {name}")

        file = self.weight_map[name]

        if device is None:
            device = self.device

        if dtype is None:
            dtype = self.dtype

        with safe_open(file, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)

        tensor = tensor.to(device=device)

        if tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)

        return tensor

    def list_weight_names(self) -> List[str]:
        if not self.weight_map:
            self.scan_weight_files()

        return sorted(self.weight_map.keys())

    def summary(self):
        if self.config is None:
            self.load_config()

        if not self.weight_map:
            self.scan_weight_files()

        print("=" * 80)
        print("MODEL PATH")
        print("=" * 80)
        print(self.model_path)

        print()
        print("=" * 80)
        print("CONFIG")
        print("=" * 80)
        print(f"model_type              : {self.config.model_type}")
        print(f"vocab_size              : {self.config.vocab_size}")
        print(f"hidden_size             : {self.config.hidden_size}")
        print(f"intermediate_size       : {self.config.intermediate_size}")
        print(f"num_hidden_layers       : {self.config.num_hidden_layers}")
        print(f"num_attention_heads     : {self.config.num_attention_heads}")
        print(f"num_key_value_heads     : {self.config.num_key_value_heads}")
        # RoPE theta not found??
        # print(f"rope_theta              : {self.config.rope_theta}")
        print(f"rms_norm_eps            : {self.config.rms_norm_eps}")

        print()
        print("=" * 80)
        print("WEIGHTS")
        print("=" * 80)
        print(f"weight files            : {len(self.weight_files)}")
        print(f"tensors                 : {len(self.weight_map)}")

        for file in self.weight_files:
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"{file.name:50s} {size_mb:10.2f} MB")


def demo():
    loader = QwenLoader(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        dtype=torch.float16,
    )

    loader.summary()

    print()
    print("=" * 80)
    print("TEST LOAD TENSORS")
    print("=" * 80)

    names = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]

    for name in names:
        tensor = loader.load_tensor(name)
        print(f"{name:55s} {tuple(tensor.shape)} {tensor.dtype} {tensor.device}")


if __name__ == "__main__":
    demo()