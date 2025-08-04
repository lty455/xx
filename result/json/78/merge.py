import json


def merge_annotations(annotations, threshold):
    if not annotations:
        return []
    # 按开始时间排序
    sorted_ann = sorted(annotations, key=lambda x: x['segment'][0])
    merged = []
    for ann in sorted_ann:
        if not merged:
            merged.append({
                'label': ann['label'],
                'segment': [ann['segment'][0], ann['segment'][1]]
            })
        else:
            last = merged[-1]
            if ann['label'] == last['label']:
                gap = ann['segment'][0] - last['segment'][1]
                if gap <= threshold:
                    # 合并，更新结束时间为两者较大的那个
                    new_end = max(last['segment'][1], ann['segment'][1])
                    merged[-1]['segment'][1] = new_end
                else:
                    merged.append({
                        'label': ann['label'],
                        'segment': [ann['segment'][0], ann['segment'][1]]
                    })
            else:
                merged.append({
                    'label': ann['label'],
                    'segment': [ann['segment'][0], ann['segment'][1]]
                })
    return merged


def process_json(input_file, output_file, threshold):
    with open(input_file, 'r') as f:
        data = json.load(f)

    for video_info in data.values():
        annotations = video_info['annotations']
        merged_annotations = merge_annotations(annotations, threshold)
        video_info['annotations'] = merged_annotations

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


# 示例使用
process_json('input.json', 'output.json', threshold=1.0)