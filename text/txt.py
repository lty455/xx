def update_video_categories(txt1_path, txt2_path, output_txt2_path):
    # 读取txt1文件并建立映射字典
    video_action_map = {}
    with open(txt1_path, "r") as txt1_file:
        for line in txt1_file:
            video_name, action = line.strip().split(" - ")
            video_action_map[video_name] = action

    # 读取txt2文件并进行替换
    updated_lines = []
    with open(txt2_path, "r") as txt2_file:
        for line in txt2_file:
            video_name = line.strip().split(" - ")[0]  # 获取视频名称
            if video_name in video_action_map:
                updated_line = f"{video_name} - {video_action_map[video_name]}"
            else:
                updated_line = line.strip()  # 如果没有找到视频名称，保持原样
            updated_lines.append(updated_line)

    # 保存更新后的内容到新的文件
    with open(output_txt2_path, "w") as output_file:
        for updated_line in updated_lines:
            output_file.write(updated_line + "\n")

    print(f"更新后的内容已保存到 {output_txt2_path}")


# 使用示例
txt1_path = "txt/anet/video_categories1_.txt"  # 包含视频名称和动作种类的文件路径
txt2_path = "txt/anet/video_categories1.txt"  # 需要更新的文件路径
output_txt2_path = "txt/anet/anet_category.txt"  # 输出更新后的文件路径

update_video_categories(txt1_path, txt2_path, output_txt2_path)
