import torch

from model import (
    QwenConfig,
    QwenRMSNorm,
    QwenMLP,
    QwenAttention,
    QwenDecoderLayer,
    repeat_kv,
    make_causal_mask,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    config = QwenConfig(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
        attention_bias=True,
    )

    batch_size = 1
    seq_len = 8

    x = torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        device=device,
        dtype=dtype,
    )

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

    print("device:", device)
    print("dtype:", dtype)
    print("=" * 60)

    norm = QwenRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
    ).to(device=device, dtype=dtype)

    y = norm(x)
    print("RMSNorm output:", y.shape)

    mlp = QwenMLP(config).to(device=device, dtype=dtype)
    y = mlp(x)
    print("MLP output:", y.shape)

    kv = torch.randn(
        batch_size,
        config.num_key_value_heads,
        seq_len,
        config.hidden_size // config.num_attention_heads,
        device=device,
        dtype=dtype,
    )
    repeated_kv = repeat_kv(
        kv,
        config.num_attention_heads // config.num_key_value_heads,
    )
    print("repeat_kv output:", repeated_kv.shape)

    attn = QwenAttention(config).to(device=device, dtype=dtype)
    y = attn(
        x,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    print("Attention output:", y.shape)

    layer = QwenDecoderLayer(config).to(device=device, dtype=dtype)
    y = layer(
        x,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    print("DecoderLayer output:", y.shape)

    print("=" * 60)
    print("All module shape tests passed.")


if __name__ == "__main__":
    main()