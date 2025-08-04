import os
import torch
import cv2
import json
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 模型路径和视频目录
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct"
video_dir = "./thumos/videos/"
output_file = "./output/video_actions.json"

# 加载模型和处理器
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_path)

# 创建输出文件夹
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 自定义 Dataset
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

# 获取视频动作类别
def get_video_categories(video_path, dynamic_fps):
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
                        "The video may contain multiple instances of various athletic actions such as CricketBowling, CricketShot, VolleyballSpiking, "
                        "JavelinThrow, Shotput, TennisSwing, GolfSwing, ThrowDiscus, Billiards, CleanAndJerk, "
                        "LongJump, Diving, CliffDiving, BasketballDunk, HighJump, HammerThrow, SoccerPenalty, "
                        "BaseballPitch, FrisbeeCatch, PoleVault. "
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
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].split(", ")

# 生成指定动作类别的 Caption
def generate_caption(video_path, dynamic_fps, category):
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
                        "Do not describe the people's action or the trajectory of the ball after it has been bowled. "
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
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# 根据 Caption 生成 Query
def generate_query(action, caption):
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
                        "3. Use a sentence to summarize the key components of the action, without using adverbs. The description should differentiate the action from other actions.\n"
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
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return json.loads(output_text[0])

# 主函数
def main():
    video_dataset = VideoDataset(video_dir)
    data_loader = DataLoader(video_dataset, batch_size=1, shuffle=False)

    results = {}

    for batch in data_loader:
        video_path, dynamic_fps = batch
        video_path = video_path[0]
        dynamic_fps = dynamic_fps[0]
        video_name = os.path.basename(video_path).replace(".mp4", "")

        try:
            # 获取动作类别
            categories = get_video_categories(video_path, dynamic_fps)
            results[video_name] = {}

            # 对每个类别生成描述和查询
            for category in categories:
                caption = generate_caption(video_path, dynamic_fps, category)
                query = generate_query(category, caption)
                results[video_name][category] = {"caption": caption, "query": query}

        except Exception as e:
            print(f"Failed to process {video_path}: {e}")

    # 保存结果到 JSON 文件
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
