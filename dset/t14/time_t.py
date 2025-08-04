import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import cv2
import numpy as np
from PIL import Image
import json
import os
from collections import defaultdict

# 模型配置
# model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
model_path = "/root/.cache/modelscope/hub/llava-hf/llava-1.5-7b-hf"
device = "cuda:0"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)


def generate_yes_no_logits(frame, action, stage):
    """ 计算阶段的置信度，同时返回是否为关键阶段 """
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)

    # 检测是否为关键阶段
    is_key_stage = stage.endswith("*")
    stage = stage.replace("*", "").strip()  # 移除 * 符号，避免影响提示

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
        return yes_prob, is_key_stage
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return 0.0, is_key_stage


def extract_frames(video_path, num_frames=300):
    """ 优化帧提取逻辑 """
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


def detect_action_intervals(frame_probs, frame_times, key_flags, min_duration=0.5, threshold=0.7, key_window=0):
    """
    1. 先确定关键阶段片段（带 * 的阶段）。
    2. 再判断非关键阶段是否被关键阶段包围，否则忽略。
    """
    key_intervals = []
    other_intervals = []
    start_time = None
    is_key = False

    for time, (prob, is_key_stage) in zip(frame_times, zip(frame_probs, key_flags)):
        if prob >= threshold:
            if start_time is None:
                start_time = time
                is_key = is_key_stage
        else:
            if start_time is not None:
                if time - start_time >= min_duration:
                    if is_key:
                        key_intervals.append([round(start_time, 2), round(time, 2)])
                    else:
                        other_intervals.append([round(start_time, 2), round(time, 2)])
                start_time = None
                is_key = False

    if start_time is not None:
        if is_key:
            key_intervals.append([round(start_time, 2), round(frame_times[-1], 2)])
        else:
            other_intervals.append([round(start_time, 2), round(frame_times[-1], 2)])

    # 过滤非关键片段，若它们在关键片段附近，则保留
    def is_near_key(interval):
        start, end = interval
        for key_start, key_end in key_intervals:
            if (abs(start - key_end) <= key_window) or (abs(end - key_start) <= key_window):
                return True
        return False

    filtered_other_intervals = [seg for seg in other_intervals if is_near_key(seg)]

    return key_intervals + filtered_other_intervals


def analyze_video(video_path, action_stages):
    """ 核心分析逻辑 """
    frames, frame_times = extract_frames(video_path)
    if not frames:
        return []

    results = defaultdict(list)

    for action, stages in action_stages.items():
        stage_probs = []
        key_flags = []

        for frame in frames:
            confs = [generate_yes_no_logits(frame, action, stage) for stage in stages]
            max_conf, key_stage = max(confs, key=lambda x: x[0])  # 取最大置信度的阶段
            stage_probs.append(max_conf)
            key_flags.append(key_stage)

        intervals = detect_action_intervals(stage_probs, frame_times, key_flags)

        for seg in intervals:
            results[action].append({
                "label": action,
                "segment": seg
            })

    final = []
    for action, segments in results.items():
        final.extend(segments)

    return final


def main(query_file, output_file):
    """ 主处理流程 """
    with open(query_file) as f:
        video_actions = json.load(f)

    output = {}

    for video_name, actions in video_actions.items():
        video_path = f"./thumos/videos/{video_name}"
        if not os.path.exists(video_path):
            continue

        action_stages = {act: data["stages"] for act, data in actions.items()}

        annotations = analyze_video(video_path, action_stages)
        if annotations:
            output[video_name] = {"annotations": annotations}
            print(f"Processed: {video_name} ({len(annotations)} segments)")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_file}")


# 执行示例
if __name__ == "__main__":
    main("./output/new_stages1_t14.json", "./output/key_result_t14.json")

