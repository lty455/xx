import json
import openai
# 输入文件路径
json1_path = "./filtered_annotation_anet.json"
json2_path = "./time_ket_anet.json"
output_txt_path = "./other_video.txt"

# 读取两个 JSON 文件
with open(json1_path, "r") as f1:
    data1 = json.load(f1)

with open(json2_path, "r") as f2:
    data2 = json.load(f2)

# 提取 video names
videos_json1 = set(f"{k}.mp4" if not k.endswith(".mp4") else k for k in data1.keys())
videos_json2 = set(data2.keys())  # json2 的 key 都是带 .mp4 的

# 找出只在 json1 中的
missing_videos = sorted(videos_json1 - videos_json2)

# 写入到输出 txt 文件
with open(output_txt_path, "w") as out_file:
    for video_name in missing_videos:
        out_file.write(video_name + "\n")

print(f"共找到 {len(missing_videos)} 个视频名，仅存在于 json1 中，已保存至 {output_txt_path}")
