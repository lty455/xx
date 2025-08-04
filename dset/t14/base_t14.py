import os
import json
import torch
import cv2
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "./thumos/videos"
output_json_file = "./output/base_t14.json"
device = "cuda:1"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)


def calculate_fps(video_path, target_frames=50):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    fps = 30 * target_frames / total_frames
    return fps


def clean_json_output(model_output):
    """
    清理模型输出，去除 ```json 和 ```，并解析 JSON
    """
    model_output = model_output.strip()
    if model_output.startswith("```json"):
        model_output = model_output[7:].strip()
    if model_output.endswith("```"):
        model_output = model_output[:-3].strip()
    try:
        return json.loads(model_output)
    except json.JSONDecodeError:
        raise ValueError("Model output is not valid JSON.")


def process_video(video_path, video_name):
    fps = calculate_fps(video_path)

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
                    "text": (
                        "The video contains multiple instances of various athletic actions including"
                        "CricketBowling, CricketShot, VolleyballSpiking, "
                        "JavelinThrow, Shotput, TennisSwing, GolfSwing, ThrowDiscus, Billiards, CleanAndJerk, "
                        "LongJump, Diving, CliffDiving, BasketballDunk, HighJump, HammerThrow, SoccerPenalty, "
                        "BaseballPitch, FrisbeeCatch, PoleVault. "
                        "For each action instance, output in JSON format: "
                        "[{'label': 'action', 'segment': [start_time, end_time]}]. "
                        "List all action instances separately without any other words."
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
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return clean_json_output(output_text)


def main():
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    video_annotations = {}

    for video_name in os.listdir(video_dir):
        if video_name.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_name)
            try:
                annotations = process_video(video_path, video_name)
                video_annotations[video_name] = {"annotations": annotations}
                print(f"{video_name} - 成功")
            except Exception as e:
                print(f"{video_name} - 失败: {e}")

    with open(output_json_file, "w") as json_file:
        json.dump(video_annotations, json_file, indent=4)

    print(f"所有视频的类别已保存到 {output_json_file}")


if __name__ == "__main__":
    main()
