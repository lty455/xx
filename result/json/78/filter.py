import json

def load_validation_subset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 只保留 subset 为 validation 的条目
    validation_data = {
        vid: info for vid, info in data.items() if info.get("subset") == "validation"
    }

    return validation_data

# 示例调用
json_file_path = "./ac1-3.json"  # ← 替换为实际文件路径
validation_entries = load_validation_subset(json_file_path)

# 可选：输出统计信息
print(f"总共 {len(validation_entries)} 个 validation 视频条目")

# 可选：保存为新文件
with open("./ac1-3.json", 'w') as f:
    json.dump(validation_entries, f, indent=4)
