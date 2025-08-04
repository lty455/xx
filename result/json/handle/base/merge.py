import json

input_json = "./base_anet_category.json"
output_json = "./base_anet_category.json"

def merge_segments(segments):
    if not segments:
        return []

    # Sort by start time
    segments.sort(key=lambda x: x["segment"][0])
    merged = [segments[0]]

    for current in segments[1:]:
        prev = merged[-1]
        if prev["label"] != current["label"]:
            # 不同动作不合并
            merged.append(current)
            continue

        prev_start, prev_end = prev["segment"]
        curr_start, curr_end = current["segment"]

        if (curr_start <= prev_end)|(curr_start-prev_end<=1):
            # 有交集或紧邻，合并
            merged[-1]["segment"][1] = max(prev_end, curr_end)
        else:
            merged.append(current)
    return merged

def main():
    with open(input_json, "r") as f:
        data = json.load(f)

    merged_data = {}
    for video, content in data.items():
        segments = content.get("annotations", [])
        merged_segments = merge_segments(segments)
        merged_data[video] = {"annotations": merged_segments}

    with open(output_json, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"已处理完所有视频片段合并，结果保存为 {output_json}")

if __name__ == "__main__":
    main()
