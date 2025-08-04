import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
import cv2
import numpy as np
from PIL import Image
import json
import os


# 模型和处理器初始化
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
device = "cuda:1"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)


def generate_yes_no_logits(frame, query):
    """生成 `yes` 和 `no` 的 logits，基于单帧图像和查询"""
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": query},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    if not image_inputs:
        return {"error": "Invalid or missing image input"}
    if not text:
        return {"error": "Invalid or missing text input"}

    image_inputs = [img.resize((256, 256)) for img in image_inputs]

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    try:
        generated_outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=1.0,
            output_scores=True,
            return_dict_in_generate=True
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"error": f"Generation error: {str(e)}"}

    logits = generated_outputs.scores[0]

    yes_id = 9454
    no_id = 2753
    try:
        target_logits = logits[:, [yes_id, no_id]]
        target_logits = torch.nan_to_num(target_logits, nan=0.0, neginf=0.0, posinf=0.0)
    except Exception as e:
        print(f"Logits processing error: {str(e)}")
        return {"error": "Invalid logits values"}

    softmax_confidence = torch.softmax(target_logits, dim=-1)

    return {
        "yes_confidence": softmax_confidence[:, 0].item(),
        "no_confidence": softmax_confidence[:, 1].item(),
    }


def extract_frames(video_path, num_frames):
    """从视频中均匀提取帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, frame_indices


def smooth_logits(logits, window_size=5):
    """对 logits 数据进行滑动窗口平滑处理"""
    smoothed = []
    for i in range(len(logits)):
        start = max(0, i - window_size // 2)
        end = min(len(logits), i + window_size // 2 + 1)
        window = logits[start:end]
        smoothed.append(sum(window) / len(window))
    return smoothed


def detect_consecutive_with_tolerance(confidences, threshold=0.8, min_consecutive=4, tolerance=2):
    """检测连续多个帧 logits>threshold，允许少量中断"""
    start_indices = []
    end_indices = []

    current_start = None
    gap_count = 0

    for i, conf in enumerate(confidences):
        if conf["yes_confidence"] > threshold:
            if current_start is None:
                current_start = i
            gap_count = 0  # 重置 gap 计数
        else:
            if current_start is not None:
                gap_count += 1
                if gap_count > tolerance:  # 超过容忍范围
                    if i - gap_count - current_start >= min_consecutive:
                        start_indices.append(current_start)
                        end_indices.append(i - gap_count - 1)
                    current_start = None
                    gap_count = 0

    # 处理最后一段
    if current_start is not None and len(confidences) - current_start >= min_consecutive:
        start_indices.append(current_start)
        end_indices.append(len(confidences) - 1)

    return start_indices, end_indices


def merge_close_segments(start_segments, end_segments, max_gap=5):
    """合并开始和结束片段"""
    merged_segments = []
    i, j = 0, 0

    while i < len(start_segments) and j < len(end_segments):
        if start_segments[i][1] < end_segments[j][0]:  # 开始片段在结束片段之前
            if (end_segments[j][0] - start_segments[i][1]) <= max_gap:
                merged_segments.append((start_segments[i][0], end_segments[j][1]))
                i += 1
                j += 1
            else:
                i += 1
        else:
            j += 1

    return merged_segments


def analyze_video(video_path, queries, num_frames=200):
    """分析视频，识别多个 start 和 end 片段"""
    frames, frame_indices = extract_frames(video_path, num_frames)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    results = []

    for query in queries:
        for action, query_info in query.items():
            start_confidences = []
            end_confidences = []

            # 逐帧获取 logits
            for i, frame in enumerate(frames):
                start_logits = generate_yes_no_logits(frame, query_info["start"])
                end_logits = generate_yes_no_logits(frame, query_info["end"])

                if "error" in start_logits or "error" in end_logits:
                    continue  # 跳过错误帧

                start_confidences.append(
                    {"frame_index": frame_indices[i], "yes_confidence": start_logits["yes_confidence"]})
                end_confidences.append(
                    {"frame_index": frame_indices[i], "yes_confidence": end_logits["yes_confidence"]})

            # 平滑 logits
            start_logits = smooth_logits([conf["yes_confidence"] for conf in start_confidences])
            end_logits = smooth_logits([conf["yes_confidence"] for conf in end_confidences])

            # 检测连续片段
            start_indices, end_indices = detect_consecutive_with_tolerance(
                [{"yes_confidence": l} for l in start_logits]
            )
            end_start_indices, end_end_indices = detect_consecutive_with_tolerance(
                [{"yes_confidence": l} for l in end_logits]
            )

            # 保存所有片段
            for s_idx, e_idx in zip(start_indices, end_indices):
                start_time = start_confidences[s_idx]["frame_index"] / fps
                end_time = end_confidences[e_idx]["frame_index"] / fps
                results.append({
                    "label": action,
                    "segment": [start_time, end_time]
                })

    return results


def load_queries(json_path):
    """从 JSON 文件加载查询"""
    with open(json_path, 'r') as f:
        return json.load(f)


def main(query_file, output_file):
    """主函数，加载查询并处理视频"""
    queries_by_video = load_queries(query_file)
    final_results = {}

    for video_name, queries in queries_by_video.items():
        video_path = f"./activitynet/videos/v1-2/val/{video_name}"
        if not os.path.exists(video_path):
            print(f"视频 {video_path} 不存在，跳过")
            continue

        # 分析视频并获取结果
        results = analyze_video(video_path, queries, num_frames=200)
        if results:
            final_results[video_name] = {"annotations": results}  # 保存所有片段
            print(f"{video_name}--yes")

    # 保存结果到 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"结果已保存到 {output_file}")


# 运行代码
query_file = "anet.json"
output_file = "result_anet.json"
main(query_file, output_file)