import json

# 读取 JSON 文件，动作编号到动作名称的映射
with open("./a_category_num.json", "r", encoding="utf-8") as f:
    action_map = json.load(f)

# 读取包含视频名和动作编号的 txt 文件
with open("./anet_category_num.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 处理每一行并替换编号为对应的动作名称
output_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        video_name, action_id = line.split(" - ")
        action_name = action_map.get(action_id.strip(), "Unknown")
        output_lines.append(f"{video_name} - {action_name}")
    except ValueError:
        print(f"格式错误：{line}")
        continue

# 输出结果到新 txt 文件
with open("a_label_num.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("处理完成，结果已保存到 video_action_names.txt")
