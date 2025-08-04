import json


def filter_short_segments(annotations, min_duration=2):
    """过滤持续时间小于指定阈值的片段"""
    filtered = []
    for ann in annotations:
        start, end = ann["segment"]
        duration = end - start
        if duration >= min_duration:
            filtered.append(ann)
    return filtered


def process_json(input_file, output_file, min_duration=2):
    """处理JSON文件并保存结果"""
    with open(input_file, "r") as f:
        data = json.load(f)

    for video_info in data.values():
        annotations = video_info["annotations"]
        # 先过滤短片段
        filtered = filter_short_segments(annotations, min_duration)
        video_info["annotations"] = filtered

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
# 示例使用
process_json('../../0321/result2.json', '../0321/result3.json', min_duration=1.0)