import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import cv2
import numpy as np
from PIL import Image
import json
import os
from collections import defaultdict
import multiprocessing as mp

# ====== 配置部分 ======
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "./activitynet/videos/v1-3/val"
query_file = "./output/stage_31.json"
output_file = "./output/s00.json"
num_workers = 3
device = "cuda:0"

# ====== 视频分析核心函数 ======

def generate_yes_no_logits(frame, action, stage):
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


def extract_frames(video_path, num_frames=400):
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


def analyze_video(video_path, action_stages):
    frames, frame_times = extract_frames(video_path)
    if not frames:
        return []

    results = defaultdict(list)

    for action, stages in action_stages.items():
        # 仅使用非关键阶段
        non_key_stages = [s.strip() for s in stages if not s.endswith("*")]
        if not non_key_stages:
            continue

        stage_probs = []

        for frame in frames:
            confs = [generate_yes_no_logits(frame, action, stage) for stage in non_key_stages]
            max_prob = max(confs)
            stage_probs.append(max_prob)

        # 合并段落
        segments = []
        current_start = None
        for time, prob in zip(frame_times, stage_probs):
            if prob >= 0.7:
                if current_start is None:
                    current_start = time
            else:
                if current_start is not None:
                    end_time = time
                    if end_time - current_start >= 0.8:
                        segments.append({
                            "start": round(current_start, 2),
                            "end": round(end_time, 2)
                        })
                    current_start = None
        if current_start is not None:
            segments.append({
                "start": round(current_start, 2),
                "end": round(frame_times[-1], 2)
            })

        for seg in segments:
            results[action].append({
                "label": action,
                "segment": [seg["start"], seg["end"]]
            })

    final = []
    for action, segments in results.items():
        final.extend(segments)
    return final


# ====== 子进程主循环 ======

def video_worker_loop(video_queue, result_queue):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch

    global device, model, processor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

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
        annotations = analyze_video(video_path, action_stages)

        if annotations:
            print(f"[PID {os.getpid()}] Processed {video_name} ({len(annotations)} segments)")
            result_queue.put((video_name, {"annotations": annotations}))
        else:
            result_queue.put((video_name, None))


# ====== 主控函数 ======

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
        video_queue.put(None)  # 终止信号

    for _ in range(len(video_items)):
        video_name, result = result_queue.get()
        if result:
            output[video_name] = result
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

    for p in workers:
        p.join()

    print(f"Results saved to {output_file}")

# ====== 执行入口 ======
if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
