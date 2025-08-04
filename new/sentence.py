import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import os

# 视频路径
video_path = "./thumos/videos/video_test_0000004.mp4"

# SigLip 模型路径
model_dir = "/root/.cache/modelscope/hub/AI-ModelScope/siglip-so400m-patch14-384"

# 加载模型与处理器
model = AutoModel.from_pretrained(model_dir)
processor = AutoProcessor.from_pretrained(model_dir)

# 定义描述子句
captions = [
    "The bowler's stance is wide, with the legs apart and the body slightly bent forward, indicating readiness to bowl.",
    "The bowler takes a few steps backward, building momentum, before releasing the ball with a quick, powerful motion of the arm and body."
    "The man is wearing a blue and white shirt and black trousers while bowling."
    "He is seen throwing the ball with his right hand and his body is twisted as he throws the ball.",
]

# 视频帧提取函数
def extract_frames(video_path, output_dir="./frames", fps=2):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(video_fps / fps)  # 间隔帧数
    frame_idx = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_idx}.png")
            cv2.imwrite(frame_path, frame)
            saved_frames.append((frame_path, frame_idx / video_fps))  # 保存帧与时间戳
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(saved_frames)} frames.")  # 调试输出
    return saved_frames
def match_frames_to_action(frame_data, captions, model, processor, threshold=0.5):
    frame_scores = []  # 存储每帧的最高匹配得分和时间戳

    for frame_path, timestamp in frame_data:
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(text=captions, images=image, padding="max_length", return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)  # 每子句的匹配概率

        # 取所有子句的最大匹配得分
        max_score = probs.squeeze().max().item()
        frame_scores.append((timestamp, max_score))

        # 打印每帧的最大得分
        print(f"Frame {frame_path} - Max Score: {max_score:.4f}")

    # 筛选出得分高于阈值的帧
    high_relevance_frames = [ts for ts, score in frame_scores if score >= threshold]
    return high_relevance_frames

def aggregate_time_segments(timestamps, max_gap=1):
    if not timestamps:
        return []

    timestamps = sorted(timestamps)
    segments = []
    start = timestamps[0]
    end = timestamps[0]

    for t in timestamps[1:]:
        if t - end <= max_gap:
            end = t
        else:
            segments.append((start, end))
            start = t
            end = t

    segments.append((start, end))
    return segments

# 提取帧
frame_data = extract_frames(video_path, fps=2)  # 每秒提取 2 帧


# 匹配帧与描述
high_relevance_frames = match_frames_to_action(frame_data, captions, model, processor, threshold=0.5)
#print(f"High relevance frames: {high_relevance_frames}")  # 打印高相关帧

# 聚合时间段

# 聚合时间段
time_segments = aggregate_time_segments(high_relevance_frames)

# 输出时间段
print("Action Time Segments:")
for segment in time_segments:
    print(f"  {segment[0]:.2f}s - {segment[1]:.2f}s")
