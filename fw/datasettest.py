import os
import cv2
from torch.utils.data import Dataset, DataLoader

# 视频文件夹路径
video_dir = "./thumos/videos/"
output_file = "out.txt"

# 自定义 Dataset
class VideoDataset(Dataset):
    def __init__(self, video_dir, target_frames=80):
        """
        初始化视频数据集
        :param video_dir: 视频文件夹路径
        :param target_frames: 每个视频目标提取的帧数
        """
        self.video_dir = video_dir
        self.video_files = [
            os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")
        ]
        self.target_frames = target_frames

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        total_frames, dynamic_fps = self.calculate_fps(video_path)
        return os.path.basename(video_path), total_frames, dynamic_fps

    def calculate_fps(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames == 0:
            raise ValueError(f"Failed to get total frames from the video: {video_path}")
        dynamic_fps = self.target_frames * 30 / total_frames
        return total_frames, dynamic_fps


# 定义 DataLoader
video_dataset = VideoDataset(video_dir)
data_loader = DataLoader(video_dataset, batch_size=1, shuffle=False)

# 处理视频数据并写入结果
with open(output_file, "w") as f_out:
    for batch in data_loader:
        video_file, total_frames, dynamic_fps = batch
        video_file = video_file[0]  # 解包文件名
        total_frames = total_frames.item()  # 转为整数
        dynamic_fps = dynamic_fps.item()  # 转为浮点数

        # 写入结果到文件
        f_out.write(f"File: {video_file}, Total Frames: {total_frames}, Dynamic FPS: {dynamic_fps:.4f}\n")
        print(f"Processed: {video_file}")
