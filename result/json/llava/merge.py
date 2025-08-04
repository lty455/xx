import json


def merge_json(file1_path, file2_path, output_path):
    # 读取第一个JSON文件
    with open(file1_path, 'r', encoding='utf-8') as f:
        json1 = json.load(f)

    # 读取第二个JSON文件
    with open(file2_path, 'r', encoding='utf-8') as f:
        json2 = json.load(f)

    # 合并：仅添加json2中不存在于json1的键
    merged = json1.copy()
    for key, value in json2.items():
        if key not in merged:
            merged[key] = value

    # 保存合并结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


# 使用示例（替换为实际文件路径）
merge_json(
    file1_path="llava_anet.json",
    file2_path="a32.json",
    output_path="llava_anet.json"
)