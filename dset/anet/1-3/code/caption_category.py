import os
import re
import time
import torch
import cv2
import multiprocessing
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===== 配置参数 =====
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "./activitynet/videos/v1-3/val"
# model_path = "/home/uestcxr/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"
# video_dir = "/mnt/disc1/val"
input_txt_file = "./output/error_3_category.txt"
output_txt_file = "./output/error_3_caption.txt"
device = "cuda:1"
target_frames = 25
timeout_per_video = 60  # 每个视频最多处理秒数


def calculate_fps(video_path, target_frames=target_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    return 30 * target_frames / total_frames


def child_worker(video_queue, result_queue):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    while True:
        task = video_queue.get()
        if task == "STOP":
            break

        video_name, category, video_path = task
        try:
            dynamic_fps = calculate_fps(video_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 360 * 420,
                            "fps": dynamic_fps,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"The video contains multiple athletic actions. "
                                f"Please focus on identifying and describing the {category} action only. "
                                f"Provide a detailed description of the {category} action, "
                                "including the people's stance, motion, and other relevant details specific to this action. "
                                "Do not generate duplicate text descriptions."
                            ),
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            caption = re.sub(r"\s+", " ", output_text[0]).strip()
            result_queue.put(("SUCCESS", video_name, category, caption))
        except Exception as e:
            result_queue.put(("ERROR", video_name, category, str(e)))


def start_new_worker(video_queue, result_queue):
    p = multiprocessing.Process(target=child_worker, args=(video_queue, result_queue))
    p.start()
    return p


def main():
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)

    processed_videos = set()
    if os.path.exists(output_txt_file):
        with open(output_txt_file, "r") as f:
            for line in f:
                if line.strip():
                    video_name = line.split("*-")[0].strip()
                    processed_videos.add(video_name)

    with open(input_txt_file, "r") as f:
        video_category_lines = f.readlines()

    video_tasks = []
    for line in video_category_lines:
        if " - " not in line:
            continue
        video_name, category = line.strip().split(" - ")
        if video_name not in processed_videos:
            video_path = os.path.join(video_dir, video_name)
            video_tasks.append((video_name, category, video_path))

    video_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    worker = start_new_worker(video_queue, result_queue)

    with open(output_txt_file, "a") as txt_file:
        count = 0
        for task in video_tasks:
            video_name, category, video_path = task
            print(f"Processing {video_name}...")
            video_queue.put(task)

            start_time = time.time()
            while True:
                if not result_queue.empty():
                    status, name, cat, result = result_queue.get()
                    if status == "SUCCESS":
                        count += 1
                        txt_file.write(f"{name}*-{cat}*-{result}\n")
                        txt_file.flush()
                        print(f"{count}...{name} - done")
                    else:
                        print(f"{name} - error: {result}")
                    break
                elif time.time() - start_time > timeout_per_video:
                    print(f"{video_name} - timeout! Restarting worker...")
                    worker.terminate()
                    worker.join()
                    video_queue = multiprocessing.Queue()
                    result_queue = multiprocessing.Queue()
                    worker = start_new_worker(video_queue, result_queue)
                    break
                else:
                    time.sleep(1)

    video_queue.put("STOP")
    worker.join()
    print(f"All captions saved to {output_txt_file}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
