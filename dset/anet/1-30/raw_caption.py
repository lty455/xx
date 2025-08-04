import os
import torch
import cv2
import re
import multiprocessing
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "/home/uestcxr/.cache/modelscope/hub/models/Qwen//Qwen2___5-VL-7B-Instruct"
video_dir = "/mnt/disc1/val"
output_txt_file = "./output/a1-3.txt"
error_log_file = "./output/anet_error_log.txt"
device = "cuda:0"

def calculate_fps(video_path, target_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    return 30 * target_frames / total_frames

def process_video(video_path):
    # 模型和处理器在子进程中加载
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    dynamic_fps = calculate_fps(video_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": dynamic_fps
                },
                {
                    "type": "text",
                    "text": "Describe the video, "
                            "emphasize the actions in the video rather than the character images or the environment",
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
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    cleaned_text = re.sub(r"\s+", " ", output_text[0]).strip()
    return cleaned_text

def safe_process(video_path, result_queue, error_queue):
    try:
        result = process_video(video_path)
        result_queue.put((video_path, result))
    except Exception as e:
        error_queue.put((video_path, str(e)))

def main():
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)

    processed_videos = set()
    if os.path.exists(output_txt_file):
        with open(output_txt_file, "r") as f:
            for line in f:
                if line.strip():
                    video_name = line.split(" *- ")[0].strip()
                    processed_videos.add(video_name)

    with open(output_txt_file, "a") as txt_file, open(error_log_file, "a") as error_log:
        for video_name in sorted(os.listdir(video_dir)):
            if not video_name.endswith(".mp4") or video_name in processed_videos:
                continue
            video_path = os.path.join(video_dir, video_name)

            result_queue = multiprocessing.Queue()
            error_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=safe_process, args=(video_path, result_queue, error_queue))
            p.start()
            p.join(timeout=600)

            if p.exitcode != 0:
                error_log.write(f"{video_name} - error: process killed or failed with exitcode {p.exitcode}\n")
                error_log.flush()
                print(f"{video_name} - error: process killed or failed")
                continue

            if not result_queue.empty():
                _, result = result_queue.get()
                txt_file.write(f"{video_name} *- {result}\n")
                txt_file.flush()
                print(f"{video_name} -  processed")
            elif not error_queue.empty():
                _, err = error_queue.get()
                error_log.write(f"{video_name} - error: {err}\n")
                error_log.flush()
                print(f"{video_name} -  error: {err}")
            else:
                error_log.write(f"{video_name} - error: unknown failure\n")
                error_log.flush()
                print(f"{video_name} -  error: unknown failure")

    print(f" Finished. Results saved to {output_txt_file}")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
