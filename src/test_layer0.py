import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import (
    QwenConfig,
    QwenDecoderLayer,
    make_causal_mask,
)


def build_config_from_hf(hf_config) -> QwenConfig:
    """
    把 Hugging Face 模型的 config 转成我们自己定义的 QwenConfig。
    """
    return QwenConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        rms_norm_eps=hf_config.rms_norm_eps,
        # rope_theta=hf_config.rope_theta,
        max_position_embeddings=hf_config.max_position_embeddings,
        attention_bias=getattr(hf_config, "attention_bias", True),
    )


def copy_layer0_weights(my_layer, hf_model):
    """
    把 HF 模型第 0 层的权重复制到我们手写的 QwenDecoderLayer 中。

    注意：
    Qwen2.5 的 q/k/v projection 通常有 bias；
    o_proj 和 MLP projection 通常没有 bias。
    """
    sd = hf_model.state_dict()

    prefix = "model.layers.0."

    with torch.no_grad():
        # RMSNorm
        my_layer.input_layernorm.weight.copy_(
            sd[prefix + "input_layernorm.weight"]
        )
        my_layer.post_attention_layernorm.weight.copy_(
            sd[prefix + "post_attention_layernorm.weight"]
        )

        # Attention projection weights
        my_layer.self_attn.q_proj.weight.copy_(
            sd[prefix + "self_attn.q_proj.weight"]
        )
        my_layer.self_attn.k_proj.weight.copy_(
            sd[prefix + "self_attn.k_proj.weight"]
        )
        my_layer.self_attn.v_proj.weight.copy_(
            sd[prefix + "self_attn.v_proj.weight"]
        )
        my_layer.self_attn.o_proj.weight.copy_(
            sd[prefix + "self_attn.o_proj.weight"]
        )

        # Attention projection bias
        if prefix + "self_attn.q_proj.bias" in sd:
            my_layer.self_attn.q_proj.bias.copy_(
                sd[prefix + "self_attn.q_proj.bias"]
            )
        if prefix + "self_attn.k_proj.bias" in sd:
            my_layer.self_attn.k_proj.bias.copy_(
                sd[prefix + "self_attn.k_proj.bias"]
            )
        if prefix + "self_attn.v_proj.bias" in sd:
            my_layer.self_attn.v_proj.bias.copy_(
                sd[prefix + "self_attn.v_proj.bias"]
            )

        # MLP
        my_layer.mlp.gate_proj.weight.copy_(
            sd[prefix + "mlp.gate_proj.weight"]
        )
        my_layer.mlp.up_proj.weight.copy_(
            sd[prefix + "mlp.up_proj.weight"]
        )
        my_layer.mlp.down_proj.weight.copy_(
            sd[prefix + "mlp.down_proj.weight"]
        )


def print_error(my_output, hf_output):
    """
    打印数值误差。
    """
    diff = (my_output - hf_output).abs()

    print("=" * 80)
    print("Error:")
    print("max_abs_error :", diff.max().item())
    print("mean_abs_error:", diff.mean().item())
    print("=" * 80)

    print("HF first values:")
    print(hf_output[0, 0, :10])

    print("My first values:")
    print(my_output[0, 0, :10])

    print("Diff first values:")
    print(diff[0, 0, :10])
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
        help="本地 Hugging Face 模型目录，例如 ~/.cache/huggingface/hub/xxx",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="你好",
        help="用于测试的输入文本",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = args.device

    # 为了更容易对齐，先用 float32。
    # 如果显存不够，可以改成 torch.float16。
    dtype = torch.float32

    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
        device_map="cuda",
    local_files_only=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        local_files_only=True
    )
    

    hf_model.eval()

    config = build_config_from_hf(hf_model.config)

    print("=" * 80)
    print("Config:")
    print(config)
    print("=" * 80)

    # 构造我们自己的第 0 层
    my_layer0 = QwenDecoderLayer(config).to(device=device, dtype=dtype)
    my_layer0.eval()

    # 复制 HF 第 0 层权重
    copy_layer0_weights(my_layer0, hf_model)

    # 准备输入
    input_ids = tokenizer(
        args.text,
        return_tensors="pt",
    ).input_ids.to(device)

    batch_size, seq_len = input_ids.shape

    print("input_ids:", input_ids)
    print("input_ids.shape:", input_ids.shape)

    position_ids = torch.arange(
        seq_len,
        device=device,
    ).unsqueeze(0).expand(batch_size, seq_len)

    attention_mask = make_causal_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        dtype=dtype,
        device=device,
    )

    with torch.no_grad():
        # 1. 用 HF 完整 model 跑一遍，拿 hidden_states[1]
        # hidden_states[0] 是 embedding 输出
        # hidden_states[1] 是第 0 层输出
        hf_outputs = hf_model.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

        hf_embed_output = hf_outputs.hidden_states[0]
        hf_layer0_output = hf_outputs.hidden_states[1]

        # 2. 我们的 layer0 使用同一个 embedding 输出作为输入
        my_layer0_output = my_layer0(
            hf_embed_output,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    print("hf_embed_output.shape :", hf_embed_output.shape)
    print("hf_layer0_output.shape:", hf_layer0_output.shape)
    print("my_layer0_output.shape:", my_layer0_output.shape)

    print_error(
        my_output=my_layer0_output,
        hf_output=hf_layer0_output,
    )


if __name__ == "__main__":
    main()