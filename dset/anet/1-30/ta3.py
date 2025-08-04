import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import cv2
import numpy as np
from PIL import Image
import json
import os
from collections import defaultdict

# === 路径配置 ===
video_root = "/mnt/disc1/val"
query_file = "/home/uestcxr/lichuan/output/a2.json"
output_file = "/home/uestcxr/lichuan/output/time_key_anet2.json"

# === 模型配置 ===
model_path = "/home/uestcxr/.cache/modelscope/hub/models/Qwen//Qwen2___5-VL-7B-Instruct"
device = "cuda:0"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)


def generate_yes_no_logits(frame, action, stage):
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)

    is_key_stage = stage.endswith("*")
    stage = stage.replace("*", "").strip()

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


def extract_frames(video_path, num_frames=200):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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


def detect_action_intervals(frame_probs, frame_times, key_flags, key_nums, min_duration=1.0, threshold=0.7, key_window=2.0):
    key_intervals = []
    current_start = None
    current_key = None

    segments = []

    for i, (time, prob, is_key, key_num) in enumerate(zip(frame_times, frame_probs, key_flags, key_nums)):
        if prob >= threshold:
            if current_start is None:
                current_start = time
                current_key = key_num if is_key else None
        else:
            if current_start is not None:
                end_time = time
                if end_time - current_start >= min_duration:
                    segments.append({
                        "start": round(current_start, 2),
                        "end": round(end_time, 2),
                        "key_num": current_key
                    })
                current_start = None
                current_key = None

    if current_start is not None:
        segments.append({
            "start": round(current_start, 2),
            "end": round(frame_times[-1], 2),
            "key_num": current_key
        })

    # 合并非关键阶段到邻近关键阶段
    merged = []
    for i, seg in enumerate(segments):
        if seg["key_num"] is not None:
            merged.append(seg)
        else:
            prev = merged[-1] if merged else None
            next_seg = next((s for s in segments[i+1:] if s["key_num"]), None)
            if prev and next_seg:
                prev["end"] = seg["end"]
            elif prev:
                prev["end"] = seg["end"]
            elif next_seg:
                next_seg["start"] = seg["start"]
            # 否则丢弃该段

    return merged


def analyze_video(video_path, action_stages):
    frames, frame_times = extract_frames(video_path)
    if not frames:
        return []

    results = defaultdict(list)

    for action, stages in action_stages.items():
        stage_probs = []
        key_flags = []
        key_nums = []

        stage_to_keynum = {}
        k_num = 1
        for stage in stages:
            if stage.endswith("*"):
                stage_to_keynum[stage.replace("*", "").strip()] = f"k{k_num}"
                k_num += 1

        for frame in frames:
            confs = [generate_yes_no_logits(frame, action, stage) for stage in stages]
            max_idx, (max_conf, is_key) = max(enumerate(confs), key=lambda x: x[1][0])
            clean_stage = stages[max_idx].replace("*", "").strip()
            stage_probs.append(max_conf)
            key_flags.append(is_key)
            key_nums.append(stage_to_keynum.get(clean_stage, None) if is_key else None)

        intervals = detect_action_intervals(stage_probs, frame_times, key_flags, key_nums)

        for seg in intervals:
            results[action].append({
                "label": action,
                "segment": [seg["start"], seg["end"]],
                "key_num": seg["key_num"]
            })

    final = []
    for action, segments in results.items():
        final.extend(segments)

    return final


def main():
    with open(query_file) as f:
        video_actions = json.load(f)

    output = {}

    for video_name, actions in video_actions.items():
        video_path = os.path.join(video_root, video_name)
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


if __name__ == "__main__":
    main()
