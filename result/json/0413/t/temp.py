import json
from collections import Counter


def count_labels(json_path):
    """
    统计JSON文件中所有不同的label及其出现次数

    参数:
    json_path (str): JSON文件路径

    返回:
    tuple: (标签种类数量, 标签到次数的映射)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 {json_path} 不存在")
        return 0, {}
    except json.JSONDecodeError:
        print(f"错误: 文件 {json_path} 不是有效的JSON格式")
        return 0, {}

    label_counter = Counter()

    # 遍历每个视频及其标注
    for video_info in data.values():
        annotations = video_info.get('annotations', [])
        for annotation in annotations:
            label = annotation.get('label', '')
            if label:
                label_counter[label] += 1

    return len(label_counter), dict(label_counter)


if __name__ == "__main__":
    json_path = 't14_512.json'  # 替换为实际的JSON文件路径
    label_count, label_stats = count_labels(json_path)

    print(f"发现 {label_count} 种不同的标签")
    print("\n标签统计详情:")
    for label, count in sorted(label_stats.items()):
        print(f"{label}: {count} 次")