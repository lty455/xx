import json


def process_json_file(input_file, output_file):
    """
    处理JSON文件，修改视频名称
    :param input_file: 输入JSON文件路径
    :param output_file: 输出JSON文件路径
    """
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 创建新的字典来存储处理后的数据
        processed_data = {}

        # 处理每个视频条目
        for video_name, video_data in data.items():
            # 如果视频名以"v_"开头，去掉"v_"
            if video_name.startswith("v_"):
                new_video_name = video_name[2:]
            else:
                new_video_name = video_name

            # 将处理后的视频名和对应数据添加到新字典
            processed_data[new_video_name] = video_data

        # 将处理后的数据写入新的JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"处理完成，已保存到 {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误：无法解析 {input_file} 中的JSON内容")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    # 直接指定输入和输出文件路径
    input_file = 'a13/a12.json'  # 修改为你的输入文件路径
    output_file = 'a13/a12.json'  # 修改为你的输出文件路径

    # 调用处理函数
    process_json_file(input_file, output_file)