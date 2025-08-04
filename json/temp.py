import json
import re

# 读取 JSON 文件
with open('./anet_action_class.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

output_lines = []
for video, action in data.items():
    # 去除 \n\n 或 \n 及其前面的内容
    clean_action = re.sub(r'.*\\n\\n', '', action)  # 如果是转义的 \\n\\n
    clean_action = re.sub(r'.*\\n', '', clean_action)  # 如果是转义的 \\n
    clean_action = re.sub(r'.*\n\n', '', clean_action)  # 实际换行
    clean_action = re.sub(r'.*\n', '', clean_action)  # 实际换行
    output_lines.append(f"{video} - {clean_action.strip()}")

# 写入 TXT 文件
with open('./ac.txt', 'w', encoding='utf-8') as f:
    for line in output_lines:
        f.write(line + '\n')
