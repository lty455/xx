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


def compute_ap(gt_segments, pred_segments, iou_threshold, return_tp_fp=False):
    """基于 VOC 插值精度-召回曲线计算方式的 AP"""
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

    if return_tp_fp:
        return tp.sum(), fp.sum()

    if tp.sum() + fp.sum() == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (len(gt_segments) + 1e-6)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)

    # 插值 AP（VOC 方式）
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def evaluate_map(gt_data, pred_data, iou_thresholds, merge_iou_threshold, merge_diff_keynum=True,
                 return_best_videos=False):
    """评估 mAP，支持控制是否合并 key_num 不同的段"""
    common_keys = set(gt_data.keys()) & set(pred_data.keys())
    print(f"Common video count: {len(common_keys)}")
    results = defaultdict(float)

    best_videos = {iou_t: None for iou_t in iou_thresholds}
    max_improvements = {iou_t: -float('inf') for iou_t in iou_thresholds}

    for iou_t in iou_thresholds:
        ap_list = []
        video_ap_map = {}

        for vid in common_keys:
            gt_ann = [{"segment": ann["segment"], "label": ann["label"]} for ann in gt_data[vid]["annotations"]]

            # 计算合并前的TP和FP
            pred_ann_raw = pred_data[vid]["annotations"]
            tp_before, fp_before = compute_ap(gt_ann, pred_ann_raw, iou_t, return_tp_fp=True)
            precision_before = tp_before / (tp_before + fp_before) if (tp_before + fp_before) > 0 else 0

            # 计算合并后的TP和FP
            pred_ann = merge_segments(
                sorted(pred_ann_raw, key=lambda x: x["segment"][0]),
                merge_iou_threshold,
                merge_diff_keynum
            )
            tp_after, fp_after = compute_ap(gt_ann, pred_ann, iou_t, return_tp_fp=True)
            precision_after = tp_after / (tp_after + fp_after) if (tp_after + fp_after) > 0 else 0

            # 计算精度提升
            precision_improvement = precision_after - precision_before

            # 记录最佳视频
            if return_best_videos and precision_improvement > max_improvements[iou_t]:
                max_improvements[iou_t] = precision_improvement
                best_videos[iou_t] = {
                    'video_id': vid,
                    'tp_before': tp_before,
                    'fp_before': fp_before,
                    'tp_after': tp_after,
                    'fp_after': fp_after,
                    'precision_before': precision_before,
                    'precision_after': precision_after,
                    'improvement': precision_improvement
                }

            ap = compute_ap(gt_ann, pred_ann, iou_t)
            ap_list.append(ap)
            video_ap_map[vid] = ap

        results[f"mAP@{iou_t:.1f}"] = np.mean(ap_list) if ap_list else 0.0

    results["avg_mAP"] = np.mean([v for k, v in results.items() if k.startswith("mAP@")])

    if return_best_videos:
        return results, best_videos
    return results


# === 主程序入口 ===
if __name__ == "__main__":
    # 文件路径
    gt_file = "../../78/filtered_annotation_anet.json"
    pred_file = "../../0413/a/time_ket_anet.json"
    # gt_file = "../../78/filtered_annotation_t14.json"
    # pred_file = "../../0413/t/t14_512.json"

    # 加载数据
    gt_data = normalize_keys(load_json(gt_file))
    pred_data = normalize_keys(load_json(pred_file))

    # IOU 阈值和合并 IOU 阈值设置
    iou_thresholds = [0.5,0.75,0.95]
    merge_iou_thresholds = [0.7, 0.8, 0.9, 1.0]

    # 控制是否合并不同 key_num 的段
    merge_diff_keynum = False  # 可以设为 False 来禁止合并不同 key_num 段

    # 计算不同合并 IOU 阈值下的结果
    for merge_iou in merge_iou_thresholds:
        print(f"\n====== Results with merge_iou_threshold = {merge_iou} (merge_diff_keynum={merge_diff_keynum}) ======")

        # 只在 merge_iou=0.7 时计算并输出最佳视频信息
        if merge_iou == 0.7:
            results, best_videos = evaluate_map(
                gt_data,
                pred_data,
                iou_thresholds,
                merge_iou,
                merge_diff_keynum,
                return_best_videos=True
            )
            for k, v in results.items():
                print(f"{k}: {v:.4f}")

            # 输出各IOU阈值下受合并影响最大的视频信息
            print("\nBest improved videos at each IOU threshold:")
            for iou_t in sorted(best_videos.keys()):
                video_info = best_videos[iou_t]
                if video_info:
                    print(f"\nIOU threshold: {iou_t:.1f}")
                    print(f"Video ID: {video_info['video_id']}")
                    print(
                        f"Before merge: TP={video_info['tp_before']}, FP={video_info['fp_before']}, Precision={video_info['precision_before']:.4f}")
                    print(
                        f"After merge: TP={video_info['tp_after']}, FP={video_info['fp_after']}, Precision={video_info['precision_after']:.4f}")
                    print(f"Precision improvement: {video_info['improvement']:.4f}")
        else:
            # 其他合并阈值正常计算但不输出最佳视频
            results = evaluate_map(
                gt_data,
                pred_data,
                iou_thresholds,
                merge_iou,
                merge_diff_keynum,
                return_best_videos=False
            )
            for k, v in results.items():
                print(f"{k}: {v:.4f}")