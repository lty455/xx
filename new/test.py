import json

# JSON 文件路径
json_file_path = "../json/activitynet1-2.json"

# 读取 JSON 文件
with open(json_file_path, "r") as file:
    data = json.load(file)

# 提取所有的标签
labels = []
for video_id, video_data in data.items():
    for annotation in video_data.get("annotations", []):
        labels.append(annotation["label"])

# 去重并统计种类总数和标签总数
unique_labels = set(labels)
total_label_types = len(unique_labels)
total_labels = len(labels)

# 输出所有标签和种类总数
print("所有的标签:")
for label in unique_labels:
    print(label)

print(f"\n种类总个数: {total_label_types}")
print(f"标签的总个数: {total_labels}")
