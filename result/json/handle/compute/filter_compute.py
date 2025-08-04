import json
import numpy as np


def compute_iou(pred_segment, gt_segment):
    """计算两个时间段的 IoU"""
    inter_start = max(pred_segment[0], gt_segment[0])
    inter_end = min(pred_segment[1], gt_segment[1])
    if inter_end <= inter_start:
        return 0.0  # 没有交集
    intersection = inter_end - inter_start
    union = (pred_segment[1] - pred_segment[0]) + (gt_segment[1] - gt_segment[0]) - intersection
    return intersection / union


def evaluate_tal(pred_file, gt_file, iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """计算 TP, FP, FN，并返回 mAP 相关指标"""

    # 读取 JSON 文件
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    with open(gt_file, 'r') as f:
        ground_truths = json.load(f)

    # 统一视频 ID 格式（去掉 .mp4 并取交集）
    results_video_ids = {vid.replace('.mp4', '') for vid in predictions.keys()}
    annotations_video_ids = set(ground_truths.keys())
    common_video_ids = results_video_ids.intersection(annotations_video_ids)

    # 过滤预测和标注数据
    filtered_predictions = {vid: predictions[f"{vid}.mp4"] for vid in common_video_ids}
    filtered_annotations = {vid: ground_truths[vid] for vid in common_video_ids}

    # 存储每个视频的 TP, FP, FN
    video_metrics = {}
    ap_per_iou = {iou: [] for iou in iou_thresholds}
    precision_per_iou = {iou: [] for iou in iou_thresholds}
    recall_per_iou = {iou: [] for iou in iou_thresholds}

    for video_id in common_video_ids:
        video_metrics[video_id] = {}
        gt_instances = filtered_annotations[video_id]["annotations"]
        pred_instances = filtered_predictions.get(video_id, {}).get("annotations", [])

        for iou_thresh in iou_thresholds:
            video_metrics[video_id][iou_thresh] = {"TP": 0, "FP": 0, "FN": 0}

        for iou_thresh in iou_thresholds:
            tp, fp, fn = 0, 0, 0
            matched_gt = set()

            for pred in pred_instances:
                pred_label = pred["label"]
                pred_segment = pred["segment"]
                best_iou, best_gt_idx = 0, None

                for gt_idx, gt in enumerate(gt_instances):
                    if gt["label"] != pred_label:
                        continue
                    iou = compute_iou(pred_segment, gt["segment"])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, gt_idx

                if best_iou >= iou_thresh:
                    if best_gt_idx not in matched_gt:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1
                else:
                    fp += 1

            fn = len(gt_instances) - len(matched_gt)

            video_metrics[video_id][iou_thresh]["TP"] = tp
            video_metrics[video_id][iou_thresh]["FP"] = fp
            video_metrics[video_id][iou_thresh]["FN"] = fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            ap = precision * recall

            ap_per_iou[iou_thresh].append(ap)
            precision_per_iou[iou_thresh].append(precision)
            recall_per_iou[iou_thresh].append(recall)

    # 计算 mAP, Precision, Recall
    mAP_scores = {iou: np.mean(ap_per_iou[iou]) for iou in iou_thresholds}
    avg_mAP = np.mean(list(mAP_scores.values()))
    avg_precision = {iou: np.mean(precision_per_iou[iou]) for iou in iou_thresholds}
    avg_recall = {iou: np.mean(recall_per_iou[iou]) for iou in iou_thresholds}

    return mAP_scores, avg_mAP, avg_precision, avg_recall, video_metrics


# 运行评估
pred_file = "../../0330/new_key_t142.json"
gt_file = "../../78/filtered_annotation_t14.json"
mAP_scores, avg_mAP, avg_precision, avg_recall, video_metrics = evaluate_tal(pred_file, gt_file)

# 打印每个视频的 TP, FP, FN
for video_id, metrics in video_metrics.items():
    print(f"视频 ID: {video_id}")
    for iou in metrics:
        print(f"  IoU {iou}: TP={metrics[iou]['TP']}, FP={metrics[iou]['FP']}, FN={metrics[iou]['FN']}")
    print()

# 打印 mAP, Precision, Recall
print("mAP Scores:")
print(json.dumps({iou: round(score, 3) for iou, score in mAP_scores.items()}, indent=4))

print("Precision Scores:")
print(json.dumps({iou: round(score, 3) for iou, score in avg_precision.items()}, indent=4))

print("Recall Scores:")
print(json.dumps({iou: round(score, 3) for iou, score in avg_recall.items()}, indent=4))

print("Average mAP:", round(avg_mAP, 3))
