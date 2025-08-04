import json
import numpy as np
from collections import defaultdict
import random


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def normalize_keys(data):
    return {k.replace(".mp4", ""): v for k, v in data.items()}


def iou(seg1, seg2):
    s1, e1 = seg1
    s2, e2 = seg2
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e2, e1) - min(s1, s2)
    return inter / union if union > 0 else 0


def iouu(seg1, seg2):
    s1, e1 = seg1
    s2, e2 = seg2
    inter = e1 - s1 + e2 - s2
    union = max(e2, e1) - min(s1, s2)
    return inter / union if union > 0 else 0


def merge_segments(segments, merge_iou_threshold=0.5, merge_diff_keynum=True):
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
            else:
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


def evaluate_map(gt_data, pred_data, iou_thresholds, merge_iou_threshold, merge_diff_keynum=True, label_filter=None):
    common_keys = set(gt_data.keys()) & set(pred_data.keys())
    results = {f"mAP@{iou:.2f}": [] for iou in iou_thresholds}

    for vid in common_keys:
        gt_ann = [{"segment": ann["segment"], "label": ann["label"]} for ann in gt_data[vid]["annotations"]]
        pred_ann_raw = pred_data[vid]["annotations"]
        pred_ann = merge_segments(
            sorted(pred_ann_raw, key=lambda x: x["segment"][0]),
            merge_iou_threshold,
            merge_diff_keynum
        )
        if label_filter is not None:
            gt_ann = [ann for ann in gt_ann if ann["label"] in label_filter]
            pred_ann = [ann for ann in pred_ann if ann["label"] in label_filter]
        if len(gt_ann) == 0:
            continue

        for iou_t in iou_thresholds:
            ap = compute_ap(gt_ann, pred_ann, iou_t)
            results[f"mAP@{iou_t:.2f}"].append(ap)

    mean_results = {}
    for k, v in results.items():
        mean_results[k] = np.mean(v) if v else 0.0
    mean_results["avg_mAP"] = np.mean([v for k, v in mean_results.items() if k.startswith("mAP@")])
    return mean_results


if __name__ == "__main__":
    gt_file = "../../a13/a13-annotation.json"
    # pred_file = "../../llava/llava_anet.json"
    pred_file = "../../0619/anet-1-3.json"

    gt_data = normalize_keys(load_json(gt_file))
    pred_data = normalize_keys(load_json(pred_file))

    # iou_thresholds = [0.5,0.75,0.95]
    iou_thresholds = np.arange(0.5,1.0,0.05)
    merge_iou_thresholds = [0.9,1.0]
    merge_diff_keynum = False
    label_count_threshold = 0
    seeds = (1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999)

    # 🔄 统计 label 出现频次（从 ground truth 获取）
    label_to_videos = defaultdict(set)
    for vid, info in gt_data.items():
        for ann in info["annotations"]:
            label_to_videos[ann["label"]].add(vid)

    label_video_count = {label: len(vset) for label, vset in label_to_videos.items()}
    filtered_labels = [label for label, count in label_video_count.items() if count > label_count_threshold]

    for merge_iou in merge_iou_thresholds:
        pred_data = normalize_keys(load_json(pred_file))
        print(f"\n====== Averaged Results with merge_iou_threshold = {merge_iou} ======")
        seen_maps = defaultdict(list)
        unseen_maps = defaultdict(list)

        for seed in seeds:
            random.seed(seed)
            labels = filtered_labels.copy()
            random.shuffle(labels)
            split = int(len(labels) * 0.75)
            seen_labels = labels[:split]
            unseen_labels = labels[split:]

            seen_result = evaluate_map(gt_data, pred_data, iou_thresholds, merge_iou, merge_diff_keynum, label_filter=seen_labels)
            unseen_result = evaluate_map(gt_data, pred_data, iou_thresholds, merge_iou, merge_diff_keynum, label_filter=unseen_labels)

            for k in seen_result:
                seen_maps[k].append(seen_result[k])
                unseen_maps[k].append(unseen_result[k])

        # print("---- Seen Results ----")
        # for k in iou_thresholds:
        #     key = f"mAP@{k:.2f}"
        #     print(f"{key}: {np.mean(seen_maps[key]):.4f}")
        # print(f"avg_mAP: {np.mean(seen_maps['avg_mAP']):.4f}")

        print("---- Unseen Results ----")
        for k in iou_thresholds:
            key = f"mAP@{k:.2f}"
            # print(f"{key}: {np.mean(unseen_maps[key]):.4f}")
        print(f"avg_mAP: {np.mean(unseen_maps['avg_mAP']):.3f}")
