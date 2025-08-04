import os
import torch
import cv2
import re
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ===== 配置参数 =====
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "./activitynet/videos/v1-3/val"
input_txt_file = "./output/a1-3-category.txt"
output_txt_file = "./output/a1-3-caption.txt"
device = "cuda:1"
target_frames = 20  # 用于动态fps

# ===== 模型加载 =====
print("Loading model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
print("Model loaded.")

# ===== 函数定义 =====
def calculate_fps(video_path, target_frames=target_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    return 30 * target_frames / total_frames

def generate_caption_for_video(video_name, category, video_path):
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

    return re.sub(r"\s+", " ", output_text[0]).strip()

# ===== 主流程 =====
def main():
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
    summ=0
    # 已处理视频集合
    processed_videos = set()
    if os.path.exists(output_txt_file):
        with open(output_txt_file, "r") as f:
            for line in f:
                if line.strip():
                    video_name = line.split("*-")[0].strip()
                    processed_videos.add(video_name)

    # 读取输入类别列表
    with open(input_txt_file, "r") as f:
        video_category_lines = f.readlines()

    with open(output_txt_file, "a") as txt_file:
        for line in video_category_lines:
            if " *- " not in line:
                continue
            video_name, category = line.strip().split(" *- ")
            if video_name in processed_videos:
                print(f"{video_name} - skipped")
                continue

            video_path = os.path.join(video_dir, video_name)
            try:
                caption = generate_caption_for_video(video_name, category, video_path)
                summ+=1
                txt_file.write(f"{video_name}*-{category}*-{caption}\n")
                txt_file.flush()
                print(f"{summ}...{video_name} - processed")
            except Exception as e:
                print(f"{video_name} - error: {e}")

    print(f"All captions saved to {output_txt_file}")

if __name__ == "__main__":
    main()
