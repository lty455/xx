import json
from collections import Counter

# 读取 JSON 文件
with open('./time_ket_anet.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计所有 label
# label_counter = Counter()
#
# for video_id, video_data in data.items():
#     for annotation in video_data.get("annotations", []):
#         label = annotation.get("label")
#         if label:
#             label_counter[label] += 1
#
# # 输出统计结果
# print("Label 统计:")
# for label, count in label_counter.items():
#     print(f"{label}: {count} 次")

print(f"\n总共有 {len(data)} 种不同的 label")