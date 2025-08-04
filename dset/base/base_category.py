import os
import json
import torch
import cv2
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "./activitynet/videos/v1-2/val"
video2action_file = "./output/a_category.txt"
output_json_file = "./output/base_anet_category.json"
device = "cuda:0"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

def read_video2action(txt_path):
    mapping = {}
    with open(txt_path, "r") as f:
        for line in f:
            if '-' in line:
                parts = line.strip().split(" - ")
                if len(parts) == 2:
                    video, action = parts
                    mapping[video.strip()] = action.strip()
    return mapping

def calculate_fps(video_path, target_frames=35):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    fps = 30 * target_frames / total_frames
    return fps

def clean_tuple_output(model_output, action_label):
    """
    清理输出格式为 [(x1,y1), (x2,y2)] 的字符串，转换为带label的 annotation 格式
    """
    model_output = model_output.strip()
    if model_output.startswith("```"):
        model_output = model_output.split("```")[-1].strip()
    try:
        parsed = eval(model_output, {"__builtins__": {}})
        if not isinstance(parsed, list):
            raise ValueError("Parsed output is not a list.")
        annotations = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                annotations.append({
                    "segment": [float(item[0]), float(item[1])],
                    "label": action_label
                })
        return annotations
    except Exception as e:
        raise ValueError(f"Model output is not a valid list of tuples: {e}\n{model_output}")

def process_video(video_path, video_name, action):
    fps = calculate_fps(video_path)

    # 👉 新的 prompt
    prompt = (
        f"Please extract all temporal segments in this video where the action '{action}' is happening. "
        f"Only return a list of (start_time, end_time) tuples like [(x1,y1), (x2,y2)] without any other words."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": prompt,
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

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return clean_tuple_output(output_text, action)

def main():
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    video2action = read_video2action(video2action_file)
    output_data = {}

    for video_name, action in video2action.items():
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f"{video_name} - no video")
            continue
        try:
            annotations = process_video(video_path, video_name, action)
            output_data[video_name] = {"annotations": annotations}
            print(f"{video_name} - yyyyyyy")
        except Exception as e:
            print(f"{video_name} - nnnnnn: {e}")

    with open(output_json_file, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"all had to {output_json_file}")

if __name__ == "__main__":
    main()
