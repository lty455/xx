import os
import json

# 定义 JSON 文件目录和视频文件目录
annotation_path = "./activitynet/annotations/activitynet1-2.json"
video_path = "./activitynet/videos/v1-2/test"

# 读取 JSON 文件
with open(annotation_path, 'r') as file:
    data = json.load(file)

# 获取视频文件名（去掉扩展名）
video_files = {os.path.splitext(file)[0] for file in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, file))}

# 构建新 JSON 数据，提取视频中包含动作类别和持续时间的标注
filtered_data = {}
empty_annotations = []  # 用于记录 annotations 为空的视频
for key, value in data.items():
    if key in video_files:
        # 检查 annotations 是否为空或缺少有效数据
        if not value.get("annotations", []):
            empty_annotations.append(key)
        else:
            for annotation in value["annotations"]:
                # 检查 segment 和 label 的完整性
                if "segment" not in annotation or "label" not in annotation:
                    empty_annotations.append(key)
                    break

        # 构建提取数据
        filtered_data[key] = {
            "duration_second": value.get("duration_second", None),
            "duration_frame": value.get("duration_frame", None),
            "annotations": value.get("annotations", []),
            "feature_frame": value.get("feature_frame", None),
        }

# 保存为新的 JSON 文件
output_json_file = './activitynet/annotations/filtered_activitynet.json'
with open(output_json_file, 'w') as file:
    json.dump(filtered_data, file, indent=4)

# 输出统计结果
video_count_in_json = len(filtered_data)  # 提取的标注视频数
video_file_count = len(video_files)      # 视频文件数
action_count = sum(len(video['annotations']) for video in filtered_data.values())

print(f"成功提取标注并保存到文件: {output_json_file}")
print(f"提取的视频数: {video_count_in_json}")
print(f"视频文件的数目: {video_file_count}")
print(f"提取的视频中包含的动作类别标注总数: {action_count}")

# 输出 annotations 为空的视频
if empty_annotations:
    print(f"以下视频的 annotations 为空或不完整（无标注信息）:")
    for video_id in empty_annotations:
        print(video_id)
