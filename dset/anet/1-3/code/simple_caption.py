import os
import torch
import cv2
import re
import multiprocessing as mp
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "/home/uestcxr/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "/mnt/disc1/val/val2"
output_txt_file = "./output/a1-2.txt"
error_log_file = "./output/anet2_error_log.txt"
device = "cuda:0"
TIMEOUT = 90  # 秒

def calculate_fps(video_path, target_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    return 30 * target_frames / total_frames

def run_model(video_path, model_path, device, return_dict):
    try:
        torch.cuda.set_device(int(device.split(":")[-1]))  # 设置当前设备
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
                        "text": "Describe the video, emphasize the actions in the video rather than the character images or the environment",
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
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        cleaned_text = re.sub(r"\s+", " ", output_text[0]).strip()
        return_dict["result"] = cleaned_text
    except Exception as e:
        return_dict["error"] = str(e)

def main():
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
    summ = 0

    # 读取已处理视频
    processed_videos = set()
    if os.path.exists(output_txt_file):
        with open(output_txt_file, "r") as f:
            for line in f:
                if line.strip():
                    video_name = line.split(" *- ")[0].strip()
                    processed_videos.add(video_name)

    # 获取未处理视频并按文件大小排序
    video_files = [
        (video_name, os.path.getsize(os.path.join(video_dir, video_name)))
        for video_name in os.listdir(video_dir)
        if video_name.endswith(".mp4") and video_name not in processed_videos
    ]
    video_files.sort(key=lambda x: x[1])

    with open(output_txt_file, "a") as txt_file, open(error_log_file, "a") as error_log:
        for video_name, _ in video_files:
            video_path = os.path.join(video_dir, video_name)
            print(f"Processing: {video_path}")

            manager = mp.Manager()
            return_dict = manager.dict()
            p = mp.Process(target=run_model, args=(video_path, model_path, device, return_dict))
            p.start()
            p.join(timeout=TIMEOUT)

            if p.is_alive():
                p.terminate()
                p.join()
                error_log.write(f"{video_name} - timeout after {TIMEOUT} seconds\n")
                error_log.flush()
                print(f"{video_name} - timeout")
                continue

            if "error" in return_dict:
                error_log.write(f"{video_name} - error: {return_dict['error']}\n")
                error_log.flush()
                print(f"{video_name} - error: {return_dict['error']}")
                continue

            result = return_dict.get("result", "").strip()
            txt_file.write(f"{video_name} *- {result}\n")
            txt_file.flush()
            summ += 1
            print(f"{video_name} - processed, {summ}")

    print(f"Finished. Results saved to {output_txt_file}")

if __name__ == "__main__":
    main()
