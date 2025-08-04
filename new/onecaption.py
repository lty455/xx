import torch
import cv2
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# 指定模型和处理器的本地路径
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct"
de="cuda:1"
# Load the model from the local path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=de,
    local_files_only=True  # 确保只从本地加载文件
)
video_path = "./activitynet/videos/v1-2/test/v_BW7_eGchA_M.mp4"
def calculate_fps(video_path, target_frames=600):
    """
    动态计算 fps，以满足目标提取帧数量
    :param video_path: 视频路径
    :param target_frames: 目标提取帧数量
    :return: 动态计算的 fps
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    fps = target_frames / total_frames
    return fps

# 计算动态 fps
dynamic_fps = calculate_fps(video_path)
# Load the processor from the local path
processor = AutoProcessor.from_pretrained(model_path)
category="Weightlifting"
# 定义消息，其中包括视频和文本内容
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": dynamic_fps,  # 动态计算的 fps
            },
            {
                "type": "text",
                "text": (
                    f"The video contains multiple athletic actions. "
                    f"Please focus on identifying and describing the {category} action only. "
                    f"Provide a detailed description of the {category} action at the beginning, "
                    "including the people's stance, motion, and other relevant details specific to this action. "
                    "Do not describe the people's action or the trajectory of the ball after it has been bowled. "
                    "Do not generate duplicate text descriptions."
                ),
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": dynamic_fps,  # 动态计算的 fps
            },
            {
                "type": "text",
                "text": (
                    f"The video contains multiple athletic actions. "
                    f"Please focus on identifying and describing the {category} action only. "
                    f"Provide a detailed description of the {category} action at the end, "
                    "including the people's stance, motion, and other relevant details specific to this action. "
                    "Do not describe the people's action or the trajectory of the ball after it has been bowled. "
                    "Do not generate duplicate text descriptions."
                ),
            },
        ],
    },
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(de)

# Increase the max_new_tokens parameter to allow for longer output
generated_ids = model.generate(**inputs, max_new_tokens=512)  # Increase max_new_tokens for more detailed description
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 输出模型生成的描述
print(output_text[0])
