import json

# 读取标注文件（真实类别）
annotation_path = "filtered_annotation_t14.json"
with open(annotation_path, "r") as f:
    gt_data = json.load(f)

# 读取预测的类别文件
category_path = "t14_category.txt"
with open(category_path, "r") as f:
    pred_lines = f.readlines()

# 解析预测的类别
pred_categories = {}
for line in pred_lines:
    parts = line.strip().split(" - ")
    if len(parts) == 2:
        video_id = parts[0].replace(".mp4", "")  # 去掉 .mp4 后缀
        categories = set(parts[1].split(", "))  # 预测类别转换为集合
        pred_categories[video_id] = categories

# 计算 TP, FP, FN
tp, fp, fn = 0, 0, 0
all_videos = set(gt_data.keys()) | set(pred_categories.keys())

# 记录 FP 和 FN 详情
error_details = {}  # {video_id: {"predicted": [...], "ground_truth": [...], "extra": [...], "missing": [...]}}

for video_id in all_videos:
    gt_labels = {ann["label"] for ann in gt_data.get(video_id, {}).get("annotations", [])}
    pred_labels = pred_categories.get(video_id, set())

    tp += len(gt_labels & pred_labels)  # 预测正确的类别
    fp_errors = pred_labels - gt_labels  # 预测了但不属于 GT 的类别
    fn_errors = gt_labels - pred_labels  # GT 里有但预测缺失的类别

    fp += len(fp_errors)
    fn += len(fn_errors)

    if fp_errors or fn_errors:  # 只有有 FP/FN 时才存储
        error_details[video_id] = {
            "predicted": list(pred_labels),
            "ground_truth": list(gt_labels),
            "extra": list(fp_errors),   # FP
            "missing": list(fn_errors)  # FN
        }

# 计算 Precision, Recall, F1-score
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# 输出统计结果
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# 输出 FP + FN 详情
print("\nFP & FN Details (per video):")
for video_id, info in error_details.items():
    print(f"Video: {video_id}")
    print(f"  Predicted Categories: {info['predicted']}")
    print(f"  Ground Truth Categories: {info['ground_truth']}")
    print(f"  False Positives (extra categories): {info['extra']}")
    print(f"  False Negatives (missing categories): {info['missing']}")
    print("-" * 50)
