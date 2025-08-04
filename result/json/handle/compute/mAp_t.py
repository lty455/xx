import json
import numpy as np
from collections import defaultdict

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def normalize_keys(data):
    """规范化视频名称，去掉 .mp4 后缀"""
    return {k.replace(".mp4", ""): v for k, v in data.items()}

def iou(seg1, seg2):
    """计算两个时间段的 IOU 值"""
    s1, e1 = seg1
    s2, e2 = seg2
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e2, e1) - min(s1, s2)
    return inter / union if union > 0 else 0

def compute_ap(gt_segments, pred_segments, iou_threshold):
    """基于 COCO 风格的 AP 计算方式"""
    pred_segments = sorted(pred_segments, key=lambda x: x.get("score", 1.0), reverse=True)
    gt_used = [False] * len(gt_segments)
    tp = np.zeros(len(pred_segments))
    fp = np.zeros(len(pred_segments))

    for i, pred in enumerate(pred_segments):
        found = False
        for j, gt in enumerate(gt_segments):
            if gt["label"] != pred["label"]:
                continue
            if iou(pred["segment"], gt["segment"]) >= iou_threshold:
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

def evaluate_map(gt_data, pred_data, iou_thresholds):
    """评估 mAP"""
    common_keys = set(gt_data.keys()) & set(pred_data.keys())
    print(f"Common video count: {len(common_keys)}")
    results = defaultdict(float)

    for iou_t in iou_thresholds:
        ap_list = []
        for vid in common_keys:
            gt_ann = [{"segment": ann["segment"], "label": ann["label"]} for ann in gt_data[vid]["annotations"]]
            pred_ann = pred_data[vid]["annotations"]
            ap = compute_ap(gt_ann, pred_ann, iou_t)
            ap_list.append(ap)
        results[f"mAP@{iou_t:.2f}"] = np.mean(ap_list) if ap_list else 0.0

    results["avg_mAP"] = np.mean([v for k, v in results.items() if k.startswith("mAP@")])
    return results

# === 主程序入口 ===
if __name__ == "__main__":
    gt_file = "../../78/filtered_annotation_t14.json"
    pred_file = "../../llava/llava_t14.json"
    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    gt_data = normalize_keys(load_json(gt_file))
    pred_data = normalize_keys(load_json(pred_file))

    results = evaluate_map(gt_data, pred_data, iou_thresholds)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
