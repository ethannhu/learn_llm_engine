from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    local_files_only=True
)


inputs = tokenizer("你好，请介绍一下你自己", return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=50
    )

print(tokenizer.decode(out[0]))