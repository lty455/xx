import json
from collections import defaultdict


def count_labels_in_json(json_file_path):
    """
    统计JSON文件中每个标签出现在多少个不同视频中

    参数:
    json_file_path (str): JSON文件路径

    返回:
    dict: 标签及其出现的视频个数的字典
    """
    # 用于存储标签及其出现的视频集合
    label_videos = defaultdict(set)

    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 遍历每个视频及其标注
        for video_id, video_data in data.items():
            # 获取当前视频的所有标注
            annotations = video_data.get('annotations', [])
            # 用于记录当前视频中已经统计过的标签
            counted_labels = set()

            for annotation in annotations:
                label = annotation.get('label', '')
                # 如果标签不为空且当前视频中尚未统计过该标签
                if label and label not in counted_labels:
                    # 将视频ID添加到该标签对应的视频集合中
                    label_videos[label].add(video_id)
                    # 标记该标签已在当前视频中统计过
                    counted_labels.add(label)

    except FileNotFoundError:
        print(f"错误：找不到文件 '{json_file_path}'")
        return {}
    except json.JSONDecodeError:
        print(f"错误：文件 '{json_file_path}' 不是有效的JSON格式")
        return {}
    except Exception as e:
        print(f"发生未知错误：{e}")
        return {}

    # 将集合长度转换为出现次数
    label_counts = {label: len(videos) for label, videos in label_videos.items()}
    return label_counts


def main():
    # 请替换为实际的JSON文件路径
    json_file_path = 'time_ket_anet.json'

    # 统计标签出现次数
    label_counts = count_labels_in_json(json_file_path)

    # 按出现次数降序排序
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

    # 输出结果
    print("标签统计结果（按出现的视频个数降序排列）：")
    print("标签名称".ljust(50), "出现的视频个数")
    print("-" * 70)
    for label, count in sorted_labels:
        print(f"{label.ljust(50)}{count}")

    # 输出动作类别总数
    print("\n动作类别总数:", len(label_counts))


if __name__ == "__main__":
    main()