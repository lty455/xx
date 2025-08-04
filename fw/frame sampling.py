import os
import cv2
import torch
from PIL import Image
from modelscope import AutoProcessor, AutoModel

# 视频存储目录
video_dir = "./activitynet/videos/v1-2/val"

# 加载模型和处理器
model_name = "/root/.cache/modelscope/hub/AI-ModelScope/siglip-so400m-patch14-384"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)


def read_action_file(file_path):
    """ 读取动作类别文件，返回字典 {视频名: 动作类别} """
    video_actions = {}
    with open(file_path, 'r') as f:
        for line in f:
            if " - " in line:
                video, action = line.strip().split(" - ")
                video_actions[video] = action
    return video_actions


def extract_frames(video_path, frame_interval=30):
    """ 从视频中按固定间隔提取帧，返回 (帧列表, 帧索引列表) """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_indices = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            frame_indices.append(frame_count)  # 记录帧编号
        frame_count += 1
    cap.release()
    return frames, frame_indices


def compute_similarity(frames, frame_indices, text):
    """ 计算帧与文本的相似度，并输出相似度 > 0.6 的帧编号 """
    inputs = processor(text=[text] * len(frames), images=frames, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image).squeeze().tolist()

    high_sim_frames = [frame_indices[i] for i, prob in enumerate(probs) if prob > 0.6]
    return high_sim_frames


# 读取动作类别
video_actions = read_action_file("./output/anet_caption.txt")

# 计算每个视频帧与动作类别的相似度，并筛选相似度 > 0.6 的帧编号
for video, action in video_actions.items():
    video_path = os.path.join(video_dir, video)  # 组合路径
    if not os.path.exists(video_path):
        print(f"视频文件 {video} 不存在，跳过...")
        continue

    frames, frame_indices = extract_frames(video_path)
    if not frames:
        print(f"无法从 {video} 提取帧")
        continue

    high_sim_frames = compute_similarity(frames, frame_indices, action)
    print(f"{video} ({action}): 高相似度帧编号 {high_sim_frames}")
