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


def iouu(seg1, seg2):
    """用于 key_num 相同段合并的 IOU 定义"""
    s1, e1 = seg1
    s2, e2 = seg2
    inter = e1 - s1 + e2 - s2
    union = max(e2, e1) - min(s1, s2)
    return inter / union if union > 0 else 0


def merge_segments(segments, merge_iou_threshold=0.5, merge_diff_keynum=True):
    """合并相邻的时间片段，可控制是否合并 key_num 不同的段"""
    merged = []
    i = 0
    while i < len(segments):
        curr = segments[i].copy()
        j = i + 1
        while j < len(segments):
            next_seg = segments[j]
            if curr["label"] != next_seg["label"]:
                break

            curr_key = int(curr["key_num"].replace('k', ''))
            next_key = int(next_seg["key_num"].replace('k', ''))

            curr_seg = curr["segment"]
            next_seg_coords = next_seg["segment"]

            if curr["key_num"] != next_seg["key_num"]:
                if merge_diff_keynum and next_key > curr_key:
                    curr["segment"][1] = next_seg_coords[1]
                    j += 1
                else:
                    break
            else:  # key_num 相同，检查 IOU
                iou_val = iouu(curr_seg, next_seg_coords)
                if iou_val > merge_iou_threshold:
                    curr["segment"][1] = next_seg_coords[1]
                    j += 1
                else:
                    break
        merged.append(curr)
        i = j
    return merged


def compute_ap(gt_segments, pred_segments, iou_threshold):
    """基于 COCO 风格的 AP 计算方式（面积法）"""
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

    # 使 recall 单调递增，precision 单调递减（插值）
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 查找 recall 值变化点
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap


def evaluate_map(gt_data, pred_data, iou_thresholds, merge_iou_threshold, merge_diff_keynum=True):
    """评估 mAP，并返回每个 IoU 阈值下 AP 最好的视频 ID"""
    common_keys = set(gt_data.keys()) & set(pred_data.keys())
    print(f"Common video count: {len(common_keys)}")
    results = defaultdict(float)
    best_videos = {}

    for iou_t in iou_thresholds:
        ap_list = []
        video_ap_map = {}
        for vid in common_keys:
            gt_ann = [{"segment": ann["segment"], "label": ann["label"]} for ann in gt_data[vid]["annotations"]]
            pred_ann_raw = pred_data[vid]["annotations"]
            pred_ann = merge_segments(
                sorted(pred_ann_raw, key=lambda x: x["segment"][0]),
                merge_iou_threshold,
                merge_diff_keynum
            )
            ap = compute_ap(gt_ann, pred_ann, iou_t)
            ap_list.append(ap)
            video_ap_map[vid] = ap

        best_vid = max(video_ap_map.items(), key=lambda x: x[1])
        best_videos[f"Best@{iou_t:.2f}"] = best_vid
        results[f"mAP@{iou_t:.2f}"] = np.mean(ap_list) if ap_list else 0.0

    results["avg_mAP"] = np.mean([v for k, v in results.items() if k.startswith("mAP@")])
    return results, best_videos


# === 主程序入口 ===
if __name__ == "__main__":
    # ground truth 和预测路径
    gt_file = "../../a13/a13-annotation.json"
    pred_file = "../../0619/anet-1-3.json"

    iou_thresholds = [0.5, 0.75, 0.95]
    merge_iou_thresholds = [0.8]
    merge_diff_keynum = True

    gt_data = normalize_keys(load_json(gt_file))
    pred_data_original = normalize_keys(load_json(pred_file))

    for merge_iou in merge_iou_thresholds:
        print(f"\n====== Results with merge_iou_threshold = {merge_iou} (merge_diff_keynum={merge_diff_keynum}) ======")

        pred_data_raw = normalize_keys(load_json(pred_file))
        pred_data_merged = normalize_keys(load_json(pred_file))

        results, best_videos = evaluate_map(gt_data, pred_data_merged, iou_thresholds, merge_iou, merge_diff_keynum)
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

        print(">> Best videos per IoU:")
        for iou_key, (vid, ap_score) in best_videos.items():
            print(f"{iou_key}: Video ID = {vid}, AP = {ap_score:.4f}")

        # 分析合并前后 AP 差异大的视频（在 IoU=0.5 下）
        iou_t_for_analysis = 0.5
        common_keys = set(gt_data.keys()) & set(pred_data_raw.keys())
        ap_deltas = []

        for vid in common_keys:
            gt_ann = [{"segment": ann["segment"], "label": ann["label"]} for ann in gt_data[vid]["annotations"]]
            pred_ann_raw = pred_data_raw[vid]["annotations"]

            pred_ann_before = sorted(pred_ann_raw, key=lambda x: x["segment"][0])
            pred_ann_after = merge_segments(pred_ann_before, merge_iou, merge_diff_keynum)

            ap_before = compute_ap(gt_ann, pred_ann_before, iou_t_for_analysis)
            ap_after = compute_ap(gt_ann, pred_ann_after, iou_t_for_analysis)

            delta = ap_after - ap_before
            ap_deltas.append((vid, ap_before, ap_after, delta))

        ap_1_videos = [(vid, before, after, delta) for (vid, before, after, delta) in ap_deltas if
                       abs(after - 1.0) < 1e-6]

        # 排序并取前3
        top_ap1 = sorted(ap_1_videos, key=lambda x: -x[3])[:3]

        print("\n>> Top-3 videos with AP=1.0 after merging and highest improvement:")
        if not top_ap1:
            print("No video found with AP=1.0 after merging.")
        else:
            for idx, (vid, before, after, delta) in enumerate(top_ap1, 1):
                print(f"Top-{idx}: Video = {vid}, Before = {before:.4f}, After = {after:.4f}, ΔAP = {delta:.4f}")
