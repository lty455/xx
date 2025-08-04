import os
import torch
import cv2
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 模型和处理器初始化
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct"
device = "cuda"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path)

def calculate_fps(video_path, target_frames=500):
    """
    动态计算 fps，以满足目标提取帧数量
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    fps = target_frames / total_frames
    return fps

def generate_action_category(video_path, dynamic_fps):
    """
    生成视频中的动作类别。
    """
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
                        "Please identify and list the action categories present in the video. "
                        "Only list the actions that appear in the video, and provide a clear response with the action categories included."
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

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def generate_caption(video_path, category, dynamic_fps):
    """
    生成视频中特定动作的开始和结束描述。
    """
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

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def generate_query(action, caption):
    """
    根据动作和描述生成查询。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Here is an action: {action}\n"
                        f"Caption: {caption}\n"
                        "Please follow these steps to create a start question, end question, and description for the action:\n"
                        "1. Consider what is unique about the very beginning of the action that could be observed in a single video frame. Craft a yes/no question about that.\n"
                        "2. Consider what is unique about the very end of the action that could be observed in a single video frame. Craft a yes/no question about that. Make sure it does not overlap with the start question.\n"
                        "3. Use a short sentence to summarize the key components of the action, without using adverbs. The description should differentiate the action from other actions.\n"
                        "Output your final answer in this JSON format:\n"
                        "{\n"
                        f'"{action}": {{\n'
                        '"start": "question",\n'
                        '"end": "question",\n'
                        '"description": "description"\n'
                        "}}\n"
                        "}"
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# 定义 Dataset
class VideoDataset(Dataset):
    def __init__(self, video_dir, target_frames=80):
        self.video_dir = video_dir
        self.video_files = [
            os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")
        ]
        self.target_frames = target_frames

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        dynamic_fps = self.calculate_fps(video_path)
        return video_path, dynamic_fps

    def calculate_fps(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames == 0:
            raise ValueError(f"Failed to get total frames from the video: {video_path}")
        return self.target_frames * 30 / total_frames

# 主函数
output_txt_file = "./output/video_queries.txt"
os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)

def main():
    video_dir = "./activitynet/videos/v1-2/test"
    video_dataset = VideoDataset(video_dir)
    data_loader = DataLoader(video_dataset, batch_size=1, shuffle=False)

    with open(output_txt_file, "w") as txt_file:
        for batch in data_loader:
            video_path, dynamic_fps = batch
            video_path = video_path[0]
            dynamic_fps = float(dynamic_fps[0].item())  # 确保 dynamic_fps 是浮点数
            video_name = os.path.basename(video_path).replace(".mp4", "")

            try:
                # 获取动作类别
                categories = generate_action_category(video_path, dynamic_fps).split(", ")

                # 对每个类别生成描述和查询
                for category in categories:
                    caption = generate_caption(video_path, category, dynamic_fps)
                    query = generate_query(category, caption)

                    # 写入文本文件
                    txt_file.write(f"{video_name}, {category}, {query}\n")

                print(f"{video_name} + 成功")

            except Exception as e:
                print(f"Failed to process {video_path}: {e}")

    print(f"Results saved to {output_txt_file}")

if __name__ == "__main__":
    main()
