import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ===== 1. 加载 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

# ===== 2. 加载模型 =====
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
    local_files_only=True
)

model.eval()

# ===== 3. 测试输入 =====
text = "你好"

# tokenizer 编码
inputs = tokenizer(
    text,
    return_tensors="pt"
)

# 移动到 GPU
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# ===== 4. 打印 input_ids =====
print("=" * 50)
print("input_ids:")
print(inputs["input_ids"])

# ===== 5. forward =====
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

# ===== 6. 打印 logits shape =====
print("=" * 50)
print("logits.shape:")
print(logits.shape)

# ===== 7. 取最后一个 token 的 logits =====
last_token_logits = logits[0, -1]

# ===== 8. top10 logits =====
topk = torch.topk(last_token_logits, k=10)

top_indices = topk.indices
top_values = topk.values

print("=" * 50)
print("last token top10 logits:")

for rank in range(10):
    token_id = top_indices[rank].item()
    logit = top_values[rank].item()

    token_text = tokenizer.decode([token_id])

    print(
        f"{rank+1:02d} | "
        f"token_id={token_id:<8} "
        f"logit={logit:>10.4f} "
        f"token={repr(token_text)}"
    )