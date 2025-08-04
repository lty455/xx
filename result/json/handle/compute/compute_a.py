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

def compute_ap(gt_segments, pred_segments, iou_threshold):
    """基于 COCO 风格的 AP 计算方式（插值面积法）"""
    pred_segments = sorted(pred_segments, key=lambda x: x.get("score", 1.0), reverse=True)
    gt_used = [False] * len(gt_segments)
    tp = np.zeros(len(pred_segments))
    fp = np.zeros(len(pred_segments))

    for i, pred in enumerate(pred_segments):
        found = False
        for j, gt in enumerate(gt_segments):
            if gt["label"] != pred["label"]:
                continue
            if compute_iou(pred["segment"], gt["segment"]) >= iou_threshold:
                if not gt_used[j]:
                    tp[i] = 1
                    gt_used[j] = True
                    found = True
                    break
        if not found:
            fp[i] = 1

    if tp.sum() + fp.sum() == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    recall = tp_cum / (len(gt_segments) + 1e-6)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap

def evaluate_tal(pred_file, gt_file, iou_thresholds=[0.5, 0.75, 0.95]):
    """仅对两个文件中都存在的视频计算 TP/FP/FN 和 mAP"""
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    with open(gt_file, 'r') as f:
        ground_truths = json.load(f)

    predictions = {vid.replace('.mp4', ''): preds for vid, preds in predictions.items()}
    common_video_ids = set(ground_truths.keys()) & set(predictions.keys())

    video_metrics = {}
    ap_per_iou = {iou: [] for iou in iou_thresholds}
    precision_per_iou = {iou: [] for iou in iou_thresholds}
    recall_per_iou = {iou: [] for iou in iou_thresholds}

    for video_id in common_video_ids:
        video_metrics[video_id] = {}
        gt_instances = ground_truths[video_id]["annotations"]
        pred_instances = predictions.get(video_id, {}).get("annotations", [])

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

            video_metrics[video_id][iou_thresh] = {"TP": tp, "FP": fp, "FN": fn}

            ap = compute_ap(gt_instances, pred_instances, iou_thresh)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            ap_per_iou[iou_thresh].append(ap)
            precision_per_iou[iou_thresh].append(precision)
            recall_per_iou[iou_thresh].append(recall)

    mAP_scores = {iou: np.mean(ap_per_iou[iou]) for iou in iou_thresholds}
    avg_mAP = np.mean(list(mAP_scores.values()))
    avg_precision = {iou: np.mean(precision_per_iou[iou]) for iou in iou_thresholds}
    avg_recall = {iou: np.mean(recall_per_iou[iou]) for iou in iou_thresholds}

    return mAP_scores, avg_mAP, avg_precision, avg_recall, video_metrics

# ==== 调用评估 ====
gt_file = "../../78/filtered_annotation_anet.json"
pred_file = "../base/base_anet_category.json"
mAP_scores, avg_mAP, avg_precision, avg_recall, video_metrics = evaluate_tal(pred_file, gt_file)

# ==== 输出结果 ====
print("mAP Scores:")
print(json.dumps({k: round(v, 3) for k, v in mAP_scores.items()}, indent=4))

print("Average mAP:", round(avg_mAP, 3))
