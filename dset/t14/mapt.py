import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# ==== 模型和设备 ====
MODEL_PATH = "/root/.cache/modelscope/hub/llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda:0"

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ==== 测试图片 ====
# 可以用自己的路径替换
image_path = "test.jpg"
frame = Image.open(image_path).convert("RGB")

# ==== 提问 ====
prompt = (
    "Answer strictly with 'yes' or 'no'.\n"
    "Is this image showing a person playing basketball?\nAnswer:"
)

conversation = [{
    "role": "user",
    "content": [
        {"type": "image", "url": frame},
        {"type": "text", "text": prompt},
    ],
}]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(DEVICE, torch.float16)

# 找到 "yes" 和 "no" 的 token id
yes_idx = processor.tokenizer.convert_tokens_to_ids("yes")
no_idx = processor.tokenizer.convert_tokens_to_ids("no")

# ==== 生成并取出 logits ====
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        return_dict_in_generate=True,
        output_scores=True
    )

logits = outputs.scores[0]                # shape: [1, vocab_size]
yes_no_logits = logits[0, [yes_idx, no_idx]]
yes_no_probs = torch.nn.functional.softmax(yes_no_logits, dim=-1)

print("yes/no logits:", yes_no_logits)
print("yes/no probs:", yes_no_probs)
print("Predicted yes probability:", yes_no_probs[0].item())
