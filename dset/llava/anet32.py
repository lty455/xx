import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import cv2
import numpy as np
from PIL import Image
import json
import os
from collections import defaultdict, Counter
import multiprocessing as mp

# ====== 配置部分 ======

model_path = "/root/.cache/modelscope/hub/llava-hf/llava-1.5-7b-hf"
video_dir = "./activitynet/videos/v1-3/val"
query_file = "./output/stage_32.json"
output_file = "./output/a32.json"
merge_strategy = "majority"  # 可选: "majority" 或 "first"
num_workers = 1
device = "cuda:1"

def generate_yes_no_logits(frame, action, stage):
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)

    is_key_stage = stage.endswith("*")
    stage = stage.replace("*", "").strip()

    prompt = (
        "Answer strictly with 'yes' or 'no'.\n"
        f"Is this frame showing the '{stage}' stage of the '{action}' action?\n"
        "Consider objects, body posture, and environment.\nAnswer:"
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
    ).to(device, torch.float16)

    yes_idx = processor.tokenizer.convert_tokens_to_ids("yes")
    no_idx = processor.tokenizer.convert_tokens_to_ids("no")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            return_dict_in_generate=True,
            output_scores=True
        )

    logits = outputs.scores[0]
    yes_no_logits = logits[0, [yes_idx, no_idx]]
    yes_no_probs = torch.nn.functional.softmax(yes_no_logits, dim=-1)
    yes_prob = yes_no_probs[0].item()

    return yes_prob, is_key_stage

def extract_frames(video_path, num_frames=300):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames, 1)

    frames, frame_times = [], []
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

def detect_action_intervals_merged_majority(frame_probs, frame_times, key_flags, key_nums, min_duration=0.8, threshold=0.7):
    segments = []
    current_start = None
    current_keynums = []

    for time, prob, key_num in zip(frame_times, frame_probs, key_nums):
        if prob >= threshold:
            if current_start is None:
                current_start = time
                current_keynums = []
            if key_num:
                current_keynums.append(key_num)
        else:
            if current_start is not None:
                end_time = time
                if end_time - current_start >= min_duration:
                    key_num_final = Counter(current_keynums).most_common(1)[0][0] if current_keynums else None
                    segments.append({
                        "start": round(current_start, 2),
                        "end": round(end_time, 2),
                        "key_num": key_num_final
                    })
                current_start = None
                current_keynums = []

    if current_start is not None:
        key_num_final = Counter(current_keynums).most_common(1)[0][0] if current_keynums else None
        segments.append({
            "start": round(current_start, 2),
            "end": round(frame_times[-1], 2),
            "key_num": key_num_final
        })

    return segments

def detect_action_intervals_merged_first(frame_probs, frame_times, key_flags, key_nums, min_duration=0.8, threshold=0.7):
    segments = []
    current_start = None
    current_keynums = []

    for time, prob, key_num in zip(frame_times, frame_probs, key_nums):
        if prob >= threshold:
            if current_start is None:
                current_start = time
                current_keynums = []
            if key_num:
                current_keynums.append(key_num)
        else:
            if current_start is not None:
                end_time = time
                if end_time - current_start >= min_duration:
                    key_num_final = current_keynums[0] if current_keynums else None
                    segments.append({
                        "start": round(current_start, 2),
                        "end": round(end_time, 2),
                        "key_num": key_num_final
                    })
                current_start = None
                current_keynums = []

    if current_start is not None:
        key_num_final = current_keynums[0] if current_keynums else None
        segments.append({
            "start": round(current_start, 2),
            "end": round(frame_times[-1], 2),
            "key_num": key_num_final
        })

    return segments

def analyze_video(video_path, action_stages, strategy="majority"):
    frames, frame_times = extract_frames(video_path)
    if not frames:
        return []

    results = defaultdict(list)

    for action, stages in action_stages.items():
        stage_probs = []
        key_flags = []
        key_nums = []

        key_stage_map = {}
        k_count = 1
        for s in stages:
            if s.endswith("*"):
                key_stage_map[s.replace("*", "").strip()] = f'{k_count}'
                k_count += 1

        for frame in frames:
            confs = [generate_yes_no_logits(frame, action, stage) for stage in stages]
            max_idx, (max_prob, is_key) = max(enumerate(confs), key=lambda x: x[1][0])
            stage_name = stages[max_idx].replace("*", "").strip()
            key_num = key_stage_map.get(stage_name) if is_key else None

            stage_probs.append(max_prob)
            key_flags.append(is_key)
            key_nums.append(key_num)

        if strategy == "first":
            intervals = detect_action_intervals_merged_first(stage_probs, frame_times, key_flags, key_nums)
        else:
            intervals = detect_action_intervals_merged_majority(stage_probs, frame_times, key_flags, key_nums)

        for seg in intervals:
            if seg["key_num"]:
                results[action].append({
                    "label": action,
                    "segment": [seg["start"], seg["end"]],
                    "key_num": seg["key_num"]
                })

    final = []
    for action, segments in results.items():
        final.extend(segments)
    return final

def video_worker_loop(video_queue, result_queue):
    import torch

    global device, model, processor
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path,use_fast=True)

    while True:
        task = video_queue.get()
        if task is None:
            break

        video_name, actions = task
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            result_queue.put((video_name, None))
            continue

        action_stages = {act: data["stages"] for act, data in actions.items()}
        annotations = analyze_video(video_path, action_stages, strategy=merge_strategy)

        if annotations:
            print(f"[PID {os.getpid()}] Processed {video_name} ({len(annotations)} segments)")
            result_queue.put((video_name, {"annotations": annotations}))
        else:
            result_queue.put((video_name, None))

def main():
    with open(query_file) as f:
        video_actions = json.load(f)

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            output = json.load(f)
        done_videos = set(output.keys())
    else:
        output = {}
        done_videos = set()

    video_items = [(k, v) for k, v in video_actions.items() if k not in done_videos]
    print(f"Total videos to process: {len(video_items)} (Skipped {len(done_videos)})")

    video_queue = mp.Queue()
    result_queue = mp.Queue()

    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=video_worker_loop, args=(video_queue, result_queue))
        p.start()
        workers.append(p)

    for item in video_items:
        video_queue.put(item)

    for _ in range(num_workers):
        video_queue.put(None)

    for _ in range(len(video_items)):
        video_name, result = result_queue.get()
        if result:
            output[video_name] = result
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

    for p in workers:
        p.join()

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
