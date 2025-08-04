import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import cv2
import numpy as np
from PIL import Image
import json
import os
from collections import defaultdict

# 模型配置（保持不变）
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
device = "cuda:1"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)


def generate_yes_no_logits(frame, action, stage):
    """优化后的单阶段判断函数"""
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)

    prompt = (
        "Analyze the image and answer strictly with 'yes' or 'no'.\n"
        f"Is this frame showing the '{stage}' stage of the '{action}' action?\n"
        "Consider:\n"
        "1. Key objects present\n"
        "2. Body posture/movement\n"
        "3. Environmental context\n"
        "Answer:"
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": frame},
            {"type": "text", "text": prompt},
        ]
    }]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[frame.resize((448, 448))],
            return_tensors="pt"
        ).to(device)

        outputs = model.generate(**inputs, max_new_tokens=3, output_scores=True, return_dict_in_generate=True)
        logits = outputs.scores[0][0]
        yes_prob = torch.softmax(torch.tensor([logits[9454], logits[2753]]), dim=0)[0].item()
        return yes_prob
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return 0.0


def extract_frames(video_path, num_frames=200):
    """优化帧提取逻辑"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算采样间隔
    interval = max(total_frames // num_frames, 1)
    frames = []
    frame_times = []

    for idx in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_times.append(idx / fps)
            if len(frames) >= num_frames:
                break

    cap.release()
    return frames, frame_times


def detect_action_intervals(frame_probs, frame_times, min_duration=1.0, threshold=0.7):
    """精准时间段检测算法"""
    intervals = []
    start_time = None

    for time, prob in zip(frame_times, frame_probs):
        if prob >= threshold:
            if start_time is None:
                start_time = time
        else:
            if start_time is not None:
                if time - start_time >= min_duration:
                    intervals.append([round(start_time, 2), round(time, 2)])
                start_time = None

    # 处理最后未结束的区间
    if start_time is not None:
        intervals.append([round(start_time, 2), round(frame_times[-1], 2)])

    return intervals


def analyze_video(video_path, action_stages):
    """核心分析逻辑"""
    frames, frame_times = extract_frames(video_path)
    if not frames:
        return []

    results = defaultdict(list)

    # 对每个动作进行分析
    for action, stages in action_stages.items():
        # 计算每帧的阶段置信度
        stage_probs = []
        for frame in frames:
            stage_conf = [generate_yes_no_logits(frame, action, stage) for stage in stages]
            stage_probs.append(max(stage_conf))  # 取最大阶段置信度

        # 检测有效时间段
        intervals = detect_action_intervals(stage_probs, frame_times)

        # 记录结果
        for seg in intervals:
            results[action].append({
                "label": action,
                "segment": seg
            })

    # 合并格式化为目标结构
    final = []
    for action, segments in results.items():
        final.extend(segments)

    return final


def main(query_file, output_file):
    """主处理流程"""
    with open(query_file) as f:
        video_actions = json.load(f)

    output = {}

    for video_name, actions in video_actions.items():
        video_path = f"./videos/{video_name}"
        if not os.path.exists(video_path):
            continue

        # 转换为阶段字典格式
        action_stages = {act: data["stages"] for act, data in actions.items()}

        # 执行分析
        annotations = analyze_video(video_path, action_stages)
        if annotations:
            output[video_name] = {"annotations": annotations}
            print(f"Processed: {video_name} ({len(annotations)} segments)")

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_file}")


# 执行示例
if __name__ == "__main__":
    main("anet_stages.json", "final_results.json")