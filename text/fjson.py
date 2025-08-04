import json

def count_videos_in_json(input_file_path):
    # 读取过滤后的JSON文件
    with open(input_file_path, 'r') as input_file:
        data = json.load(input_file)

    # 统计视频数量
    video_count = len(data)

    return video_count

# 示例使用：调用函数统计视频数量
input_file_path = '../result/json/0321/result.json'  # 你的新生成的 JSON 文件路径

video_count = count_videos_in_json(input_file_path)

print(f"Number of videos in the filtered data: {video_count}")
