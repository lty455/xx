import os
import torch
import cv2
import re
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "./activitynet/videos/v1-3/val"
output_txt_file = "./output/a1-3.txt"
error_log_file = "./output/anet_error_log.txt"
device = "cuda:1"

def calculate_fps(video_path, target_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    return 30 * target_frames / total_frames

def process_video(video_path, model, processor):
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
    return cleaned_text

def main():
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
    summ=0
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    print("Model loaded.")

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
            try:
                result = process_video(video_path, model, processor)
                summ+=1
                txt_file.write(f"{video_name} *- {result}\n")
                txt_file.flush()
                print(f"{video_name} - processed,{summ}")
            except Exception as e:
                error_log.write(f"{video_name} - error: {str(e)}\n")
                error_log.flush()
                print(f"{video_name} - error: {str(e)}")

    print(f"Finished. Results saved to {output_txt_file}")

if __name__ == "__main__":
    main()
