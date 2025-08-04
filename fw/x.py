import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 模型缓存路径
model_dir = "/root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct"

# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto"
)

# 加载默认处理器
processor = AutoProcessor.from_pretrained(model_dir)

def generate_yes_no_logits(messages):
    """
    生成 `yes` 和 `no` 的 logits。
    """
    # 准备推理输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    if not image_inputs:
        return {"error": "Invalid or missing image input"}
    if not text:
        return {"error": "Invalid or missing text input"}

    # 调整图像大小以适配模型
    image_inputs = [img.resize((256, 256)) for img in image_inputs]

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    print(f"Processed text: {text}")
    print(f"Processed image inputs: {image_inputs}")

    # 推理：生成输出
    try:
        generated_outputs = model.generate(
            **inputs,
            max_new_tokens=5,  # 允许生成完整的 "Yes" 或 "No"
            temperature=1.0,   # 保持生成的随机性
            output_scores=True,
            return_dict_in_generate=True
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"error": f"Generation error: {str(e)}"}

    # 提取 logits
    logits = generated_outputs.scores[0]
    print(f"Logits shape: {logits.shape}")

    # 提取 `yes` 和 `no` 的 logits
    yes_id = 9454
    no_id = 2753
    try:
        target_logits = logits[:, [yes_id, no_id]]
        # 将 -inf 替换为 0
        target_logits = torch.nan_to_num(target_logits, nan=0.0, neginf=0.0, posinf=0.0)
        print(f"Target logits (yes_id={yes_id}, no_id={no_id}): {target_logits}")
    except Exception as e:
        print(f"Logits processing error: {str(e)}")
        return {"error": "Invalid logits values"}

    # 计算 softmax 后的置信度
    softmax_confidence = torch.softmax(target_logits, dim=-1)
    print(f"Softmax confidence: {softmax_confidence}")

    # 返回 `yes` 和 `no` 的 logits 和 softmax
    return {
        "yes_logits": target_logits[:, 0].tolist(),
        "no_logits": target_logits[:, 1].tolist(),
        "yes_confidence": softmax_confidence[:, 0].tolist(),
        "no_confidence": softmax_confidence[:, 1].tolist(),
    }

# 示例：输入帧和查询文本
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Answer strictly yes or no: Is there a person and a cat?"},
        ],
    }
]

# 获取 `yes` 和 `no` 的 logits 和置信度
yes_no_logits = generate_yes_no_logits(messages)
print(yes_no_logits)
