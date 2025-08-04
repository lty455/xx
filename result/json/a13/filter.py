import json

def extract_unique_labels(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            unique_labels = set()
            for video_id in data:
                annotations = data[video_id].get('annotations', [])
                for annotation in annotations:
                    label = annotation.get('label')
                    if label:
                        unique_labels.add(label)
            return sorted(unique_labels)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式")
        return []
    except Exception as e:
        print(f"发生未知错误：{e}")
        return []

if __name__ == "__main__":
    file_path = "./a13-annotation.json" # 请替换为实际文件路径
    labels = extract_unique_labels(file_path)
    print("所有唯一标签：")
    for label in labels:
        print(label)