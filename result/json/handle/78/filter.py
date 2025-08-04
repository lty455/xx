import json

# 读取结果文件
with open('../../0326/key_result_t14.json', 'r') as f:
    results = json.load(f)

# 读取标注文件# 找出双方都有的视频 ID
with open('../../0326/annotation_t14.json', 'r') as f:
    annotations = json.load(f)

# 获取两个文件中的视频 ID
# 去掉 results 中的 .mp4 后缀
results_video_ids = {vid.replace('.mp4', '') for vid in results.keys()}
annotations_video_ids = set(annotations.keys())

common_video_ids = results_video_ids.intersection(annotations_video_ids)

# 过滤结果文件（保留 .mp4 后缀）
filtered_results = {f"{video_id}.mp4": results[f"{video_id}.mp4"] for video_id in common_video_ids}

# 过滤标注文件
filtered_annotations = {video_id: annotations[video_id] for video_id in common_video_ids}

# 保存过滤后的结果文件
with open('../../0326/key_result_t14.json', 'w') as f:
    json.dump(filtered_results, f, indent=4)

# 保存过滤后的标注文件
with open('../../0326/annotation_t14.json', 'w') as f:
    json.dump(filtered_annotations, f, indent=4)