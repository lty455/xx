import json
import numpy as np

def compute_iou(pred_segment, gt_segment):
    """计算两个时间段的 IoU"""
    inter_start = max(pred_segment[0], gt_segment[0])
    inter_end = min(pred_segment[1], gt_segment[1])
    if inter_end <= inter_start:
        return 0.0
    intersection = inter_end - inter_start
    union = (pred_segment[1] - pred_segment[0]) + (gt_segment[1] - gt_segment[0]) - intersection
    return intersection / union

def validate_labels(gt_instances, pred_instances):
    """验证预测与标注动作类别一致性"""
    if not gt_instances or not pred_instances:
        return False

    gt_labels = {ann["label"] for ann in gt_instances}
    pred_labels = {ann["label"] for ann in pred_instances}

    return (len(gt_labels) == 1) and (gt_labels == pred_labels)

def evaluate_tal(pred_file, gt_file, iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7], use_label_validation=True):
    """改进版评估函数，加入过滤和类别一致性检查"""
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    with open(gt_file, 'r') as f:
        ground_truths = json.load(f)

    predictions = {vid.replace('.mp4', ''): preds for vid, preds in predictions.items()}
    valid_video_ids = set(predictions.keys()).intersection(ground_truths.keys())

    metrics = {
        'ap': {t: [] for t in iou_thresholds},
        'precision': {t: [] for t in iou_thresholds},
        'recall': {t: [] for t in iou_thresholds},
        'video_details': {}
    }

    for video_id in valid_video_ids:
        gt_instances = ground_truths[video_id]["annotations"]
        pred_instances = predictions.get(video_id, {}).get("annotations", [])

        if use_label_validation and not validate_labels(gt_instances, pred_instances):
            continue

        video_metrics = {}
        for iou_thresh in iou_thresholds:
            tp, fp, fn = 0, 0, 0
            matched_gt = set()

            for pred in pred_instances:
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_instances):
                    iou = compute_iou(pred["segment"], gt["segment"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh:
                    if best_gt_idx not in matched_gt:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1
                else:
                    fp += 1

            fn = len(gt_instances) - len(matched_gt)

            video_metrics[iou_thresh] = {
                'TP': tp,
                'FP': fp,
                'FN': fn
            }

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            ap = precision * recall

            metrics['ap'][iou_thresh].append(ap)
            metrics['precision'][iou_thresh].append(precision)
            metrics['recall'][iou_thresh].append(recall)

        metrics['video_details'][video_id] = video_metrics

    final_metrics = {
        'mAP': {t: np.mean(metrics['ap'][t]) for t in iou_thresholds},
        'avg_precision': {t: np.mean(metrics['precision'][t]) for t in iou_thresholds},
        'avg_recall': {t: np.mean(metrics['recall'][t]) for t in iou_thresholds}
    }

    return final_metrics, metrics['video_details']

pred_file = "../../0326/result.json"
gt_file = "../../78/filtered_annotation_t14.json"
final_metrics, video_metrics = evaluate_tal(pred_file, gt_file, use_label_validation=False)

print("\n" + "=" * 50)
print("视频级详细指标:")
for video_id, metrics in video_metrics.items():
    print(f"\n视频 ID: {video_id}")
    for iou in metrics:
        print(f"  IoU {iou}: TP={metrics[iou]['TP']}, FP={metrics[iou]['FP']}, FN={metrics[iou]['FN']}")

print("\n" + "=" * 50)


print("\nPrecision Scores:")
print(json.dumps({f"IoU {k}": round(v, 3) for k, v in final_metrics['avg_precision'].items()}, indent=4))

print("\nRecall Scores:")
print(json.dumps({f"IoU {k}": round(v, 3) for k, v in final_metrics['avg_recall'].items()}, indent=4))

print("mAP Scores:")
print(json.dumps({f"IoU {k}": round(v, 3) for k, v in final_metrics['mAP'].items()}, indent=4))
avg_mAP = np.mean(list(final_metrics['mAP'].values()))
print(f"\nAverage mAP: {round(avg_mAP, 3)}")
print("=" * 50)
