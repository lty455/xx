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
    """计算 TP, FP, FN，并返回每个视频的三项指标"""
    # 读取 JSON 文件
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    with open(gt_file, 'r') as f:
        ground_truths = json.load(f)

    # 统一 video ID 格式（去掉预测文件中的 `.mp4`）
    predictions = {vid.replace('.mp4', ''): preds for vid, preds in predictions.items()}

    # 存储每个视频在不同 IoU 阈值下的 TP, FP, FN
    video_metrics = {}

    # 存储不同 IoU 阈值下的 AP、Precision 和 Recall
    ap_per_iou = {iou: [] for iou in iou_thresholds}
    precision_per_iou = {iou: [] for iou in iou_thresholds}
    recall_per_iou = {iou: [] for iou in iou_thresholds}

    for video_id in ground_truths:
        video_metrics[video_id] = {}
        gt_instances = ground_truths[video_id]["annotations"]
        pred_instances = predictions.get(video_id, {}).get("annotations", [])

        # 初始化每个 IoU 阈值的 TP, FP, FN
        for iou_thresh in iou_thresholds:
            video_metrics[video_id][iou_thresh] = {"TP": 0, "FP": 0, "FN": 0}

        # 遍历所有 IoU 阈值
        for iou_thresh in iou_thresholds:
            tp, fp, fn = 0, 0, 0
            matched_gt = set()

            # 遍历预测实例
            for pred in pred_instances:
                pred_label = pred["label"]
                pred_segment = pred["segment"]
                best_iou, best_gt_idx = 0, None

                # 遍历标注实例
                for gt_idx, gt in enumerate(gt_instances):
                    if gt["label"] != pred_label:
                        continue  # 类别不匹配跳过
                    iou = compute_iou(pred_segment, gt["segment"])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, gt_idx

                # 判断是否匹配成功
                if best_iou >= iou_thresh:
                    if best_gt_idx not in matched_gt:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1  # 重复匹配视为 FP
                else:
                    fp += 1  # IoU 不足视为 FP

            # 计算 FN
            fn = len(gt_instances) - len(matched_gt)

            # 保存当前视频的 TP, FP, FN
            video_metrics[video_id][iou_thresh]["TP"] = tp
            video_metrics[video_id][iou_thresh]["FP"] = fp
            video_metrics[video_id][iou_thresh]["FN"] = fn

            # 计算 Precision 和 Recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # 近似计算 AP
            ap = precision * recall

            # 存储指标
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
pred_file = "../0321/result.json"
gt_file = "../0321/annotation.json"
mAP_scores, avg_mAP, avg_precision, avg_recall, video_metrics = evaluate_tal(pred_file, gt_file)

# 打印每个视频的 TP, FP, FN
for video_id, metrics in video_metrics.items():
    print(f"视频 ID: {video_id}")
    for iou in metrics:
        print(f"  IoU {iou}: TP={metrics[iou]['TP']}, FP={metrics[iou]['FP']}, FN={metrics[iou]['FN']}")
    print()  # 空行分隔

# 打印 mAP, Precision, Recall
print("mAP Scores:")
mAP_scores_rounded = {iou: round(score, 3) for iou, score in mAP_scores.items()}  # 保留三位小数
print(json.dumps(mAP_scores_rounded, indent=4))

print("Precision Scores:")
precision_rounded = {iou: round(score, 3) for iou, score in avg_precision.items()}  # 保留三位小数
print(json.dumps(precision_rounded, indent=4))

print("Recall Scores:")
recall_rounded = {iou: round(score, 3) for iou, score in avg_recall.items()}  # 保留三位小数
print(json.dumps(recall_rounded, indent=4))

print("Average mAP:", round(avg_mAP, 3))
