import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import cv2
import numpy as np

# 模型和处理器初始化
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct"
device = "auto"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path)
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

def extract_frames(video_path, num_frames):
    """
    从视频中均匀提取帧。
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式
        frames.append(frame)
    cap.release()
    return frames, frame_indices

def generate_action_queries(action, caption):
    """
    根据动作和描述生成开始、结束查询以及描述。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Here is an action: {action} "
                        f"Caption: {caption} "
                        "Please follow these steps to create a start question, end question, and description for the action:"
                        "1. Consider what is unique about the very beginning of the action that could be observed in a single video frame. Craft a yes/no question about that."
                        "2. Consider what is unique about the very end of the action that could be observed in a single video frame. Craft a yes/no question about that. Make sure it does not overlap with the start question."
                        "3. Use a sentence to summarize the key components of the action, without using adverbs. The description should differentiate the action from other actions."
                        "Output your final answer in this JSON format:\n"
                        "{"
                            f'"{action}": {{'
                                '"start": "question",\n'
                                '"end": "question",\n'
                                '"description": "description"\n'
                            "}}"
                        "}"
                        "Make sure to follow the JSON formatting exactly, with the action in <action>.\n"
                        "Do not add any other elements to the JSON. Only include the start question, end question, and description.\n"
                        "Do not include any explanatory text or preamble before the JSON. Only output the JSON."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

def analyze_video_with_generated_queries(video_path, action, caption, num_frames=200):
    """
    对视频分析，使用生成的查询进行片段判断。
    """
    queries = generate_action_queries(action, caption)
    print(f"Generated Queries: {queries}")

    frames, frame_indices = extract_frames(video_path, num_frames)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    queries_json = eval(queries)  # 将生成的 JSON 转换为字典
    start_query = queries_json[action]["start"]
    end_query = queries_json[action]["end"]

    max_start_score = -1
    max_end_score = -1
    start_time = None
    end_time = None
    start_frame_index = None
    end_frame_index = None

    for i, frame in enumerate(frames):
        yes_confidence_start, _ = generate_yes_no_logits(frame, start_query)
        yes_confidence_end, _ = generate_yes_no_logits(frame, end_query)

        # 更新最高得分的开始时刻
        if yes_confidence_start > max_start_score:
            max_start_score = yes_confidence_start
            start_time = i * (1 / fps)
            start_frame_index = frame_indices[i]
        elif yes_confidence_start == max_start_score and start_time is not None:
            start_time = min(start_time, i * (1 / fps))
            start_frame_index = min(start_frame_index, frame_indices[i])

        # 更新最高得分的结束时刻
        if yes_confidence_end > max_end_score:
            max_end_score = yes_confidence_end
            end_time = i * (1 / fps)
            end_frame_index = frame_indices[i]
        elif yes_confidence_end == max_end_score and end_time is not None:
            end_time = max(end_time, i * (1 / fps))
            end_frame_index = max(end_frame_index, frame_indices[i])

    return {
        "start_time": start_time,
        "end_time": end_time,
        "start_score": max_start_score,
        "end_score": max_end_score,
        "start_frame_index": start_frame_index,
        "end_frame_index": end_frame_index,
        "queries": queries_json
    }

# 示例输入
video_path = "example_video.mp4"
action = "Weightlifting"
caption = (
    "The video shows a weightlifter performing a clean and jerk lift. "
    "The weightlifter is seen bending down and lifting a barbell to their chest, then quickly lifting it over their head."
)

# 分析视频
video_results = analyze_video_with_generated_queries(video_path, action, caption, num_frames=200)
print("Video analysis results:")
print(video_results)
