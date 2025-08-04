def replace_video_descriptions(file1, file2, output_file):
    # 读取文件1中的内容
    with open(file1, 'r', encoding='utf-8') as f1:
        file1_content = f1.readlines()

    # 读取文件2中的内容
    with open(file2, 'r', encoding='utf-8') as f2:
        file2_content = f2.readlines()

    # 将文件1的内容按视频名存储为字典
    video_descriptions = {}
    for line in file1_content:
        parts = line.strip().split('--', 2)
        if len(parts) >= 3:
            video_name = parts[0]  # 视频名
            description = parts[1:]  # 描述
            video_descriptions[video_name] = '--'.join(description)

    # 替换文件2中的内容
    output_content = []
    replace_count = 0  # 计数器，记录替换的视频个数

    for line in file2_content:
        parts = line.strip().split('--', 2)
        if len(parts) >= 3:
            video_name = parts[0]  # 视频名
            if video_name in video_descriptions:
                # 用文件1中的描述替换文件2中的描述
                new_line = video_name + '--' + parts[1] + '--' + video_descriptions[video_name]
                output_content.append(new_line)
                replace_count += 1  # 增加替换计数
            else:
                output_content.append(line.strip())  # 如果找不到对应的视频名，则保持原样
        else:
            output_content.append(line.strip())  # 如果没有符合的视频名，保持原样

    # 将结果写入新文件
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write("\n".join(output_content))

    return replace_count  # 返回替换的个数

# 使用示例
file1 = 'acap.txt'  # 输入文件1的路径
file2 = './acap1.txt'  # 输入文件2的路径
output_file = 'anet_captionx.txt'  # 输出文件路径

replace_count = replace_video_descriptions(file1, file2, output_file)
print(f"替换了 {replace_count} 个视频描述")
