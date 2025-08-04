import os
import json
import logging
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoModel
)

###############################################################################
# 1. Thumos14Dataset
###############################################################################
class Thumos14Dataset(Dataset):
    def __init__(
        self,
        split,
        video_folder,
        json_file,
        sample_rate,
        num_frames,
        default_fps,
        max_seq_len,
        trunc_thresh,
        crop_ratio,
        input_dim,
        num_classes,
        video_ext,
        force_upsampling
    ):
        super().__init__()
        self.split = split
        self.video_folder = video_folder
        self.json_file = json_file
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.default_fps = default_fps
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.video_ext = video_ext
        self.force_upsampling = force_upsampling

        # 提取 label_mapping
        self.label_mapping = self._extract_label_mapping()

        # 加载视频列表
        self.data_list = self._load_json_db()

    def _extract_label_mapping(self):
        """
        从 Thumos14 的 JSON 文件中提取 "label" -> "label_id" 的映射（只处理指定 split）。
        """
        with open(self.json_file, "r") as f:
            data = json.load(f)

        mapping = {}
        for vid, vinfo in data["database"].items():
            # 只处理指定的 split（例如 test）
            if vinfo["subset"].lower() != self.split:
                continue
            for ann in vinfo.get("annotations", []):
                lbl = ann["label"]
                lid = ann["label_id"]
                if lbl not in mapping:
                    mapping[lbl] = lid

        return mapping

    def _load_json_db(self):
        """
        加载 JSON 数据库，返回符合指定 split 的视频及其注释。
        """
        with open(self.json_file, "r") as f:
            data = json.load(f)
        db = data["database"]

        out = {}
        for vid, vinfo in db.items():
            if vinfo["subset"].lower() != self.split:
                continue
            ann_list = []
            for ann in vinfo.get("annotations", []):
                ann_list.append({
                    "label": ann["label"],
                    "start": ann["segment"][0],
                    "end": ann["segment"][1],
                    "label_id": ann["label_id"]
                })
            out[vid] = ann_list
        return out

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vids = list(self.data_list.keys())
        vid = vids[idx]
        ann = self.data_list[vid]
        video_path = os.path.join(self.video_folder, vid + self.video_ext)
        frames = self._load_and_sample_frames(video_path)
        return {
            "video_id": vid,
            "frames": frames,
            "gts": ann
        }

    def _load_and_sample_frames(self, video_path):
        """
        加载视频并按采样率采样帧。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")

        frames = []
        idx = 0
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.sample_rate == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                cnt += 1
                if cnt >= self.num_frames:
                    break
            idx += 1
        cap.release()

        if len(frames) < self.num_frames and self.force_upsampling and len(frames) > 0:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        return frames

###############################################################################
# 2. 辅助函数
###############################################################################
def frames_to_pil(frames_np):
    """
    将 NumPy 数组帧转换为 PIL 图像对象。
    """
    preprocessed = []
    for idx, frame in enumerate(frames_np):
        try:
            pil_image = Image.fromarray(frame)
            preprocessed.append(pil_image)
        except Exception as e:
            logging.error(f"处理帧 {idx} 时出错: {e}")
    return preprocessed

def process_vision_info(messages):
    """
    处理消息中的图像信息，返回图像输入和视频输入。
    """
    image_inputs = []
    video_inputs = []
    for msg in messages:
        for content in msg["content"]:
            if content["type"] == "image":
                image_url = content["image"]
                if image_url.startswith("http"):
                    # 下载图像
                    from urllib.request import urlopen
                    from io import BytesIO
                    try:
                        with urlopen(image_url) as response:
                            img = Image.open(BytesIO(response.read())).convert("RGB")
                            image_inputs.append(img)
                    except Exception as e:
                        logging.error(f"下载或处理图像失败: {e}")
                else:
                    # 加载本地图像
                    try:
                        img = Image.open(content["image"]).convert("RGB")
                        image_inputs.append(img)
                    except Exception as e:
                        logging.error(f"加载本地图像失败: {e}")
            elif content["type"] == "video":
                # 处理视频输入（如有需要）
                video_inputs.append(content["video"])
    return image_inputs, video_inputs

def qwen_generate_text(model, processor, messages, max_new_tokens=256):
    """
    使用 Qwen2-VL 生成文本。
    """
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # 去掉输入部分
    gen_trim = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    output_text = processor.batch_decode(
        gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text

def generate_caption_with_short_and_detailed(llm_model, llm_processor, frames_pil, categories, max_images=1):
    """
    使用 Qwen2-VL 模型生成包含 short_description 和 detailed_description 的多实例 JSON。
    """
    cat_str = ", ".join(categories)
    # 构建 messages 列表，包含图像和文本
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": frames_pil[i] if i < len(frames_pil) else frames_pil[-1],  # 使用前 max_images 张图像
                },
                {"type": "text", "text": f"Describe this video containing the following categories: {cat_str}."},
            ],
        }
    ]
    text_out = qwen_generate_text(llm_model, llm_processor, messages, max_new_tokens=512)
    logging.info(f"Detailed + short desc:\n{text_out}")
    return text_out

def parse_step1_caption(llm_text):
    """
    解析 Step1 生成的 JSON Caption。
    """
    try:
        arr = json.loads(llm_text)
        if isinstance(arr, list):
            return arr
        return []
    except json.JSONDecodeError:
        logging.error(f"无法解析 Step1 的 JSON:\n{llm_text}")
        return []

def split_detailed_description(llm_model, llm_processor, instance_id, detailed_desc):
    """
    将 detailed_description 拆分为多个短动作句子。
    """
    prompt = f"""
We have the detailed description of {instance_id}:

\"\"\"{detailed_desc}\"\"\"

Please split it into multiple short action sentences in JSON:
{{
  "sub_actions":[
    "...",
    "...",
    ...
  ]
}}
"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    txt = qwen_generate_text(llm_model, llm_processor, messages, max_new_tokens=256)
    try:
        data = json.loads(txt)
        return data.get("sub_actions", [])
    except json.JSONDecodeError:
        logging.error(f"无法解析拆分后的 JSON:\n{txt}")
        return []

def generate_start_end_query_for_sentence(llm_model, llm_processor, instance_id, short_sent):
    """
    为每个子动作句子生成 start_query 和 end_query。
    """
    prompt = f"""
Sub-action of {instance_id}: "{short_sent}"
Generate JSON:
{{
  "start_query":"...",
  "end_query":"..."
}}
"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    txt = qwen_generate_text(llm_model, llm_processor, messages, max_new_tokens=128)
    try:
        data = json.loads(txt)
        logging.info(f"生成的查询: {data}")
        return data
    except json.JSONDecodeError:
        logging.error(f"无法解析生成的查询 JSON:\n{txt}")
        return {"start_query": "", "end_query": ""}

def siglip_confidence(siglip_model, siglip_processor, pil_image, text_query):
    """
    使用 SigLIP 计算某帧图像与查询的相似度分数。
    """
    inputs = siglip_processor(
        text=[text_query],
        images=pil_image,
        return_tensors="pt",
        padding="max_length"
    ).to(siglip_model.device)
    with torch.no_grad():
        out = siglip_model(**inputs)
    logit = out.logits_per_image
    prob = torch.sigmoid(logit)
    confidence = prob[0][0].item()
    return confidence

def locate_sub_event(siglip_model, siglip_processor, frames_pil, start_q, end_q, threshold=0.5):
    """
    根据 start_query 和 end_query 定位子事件的开始和结束帧。
    """
    conf_s = []
    conf_e = []
    for idx, fr in enumerate(frames_pil):
        sc = siglip_confidence(siglip_model, siglip_processor, fr, start_q)
        ec = siglip_confidence(siglip_model, siglip_processor, fr, end_q)
        conf_s.append(sc)
        conf_e.append(ec)
        logging.info(f"Frame {idx}: start_conf={sc:.4f}, end_conf={ec:.4f}")

    intervals = []
    starts = [i for i, v in enumerate(conf_s) if v > threshold]
    ends = [i for i, v in enumerate(conf_e) if v > threshold]

    for s in starts:
        for e in ends:
            if e > s:
                intervals.append((s, e))
    return intervals

def merge_atomic_intervals(interval_lists):
    """
    合并同一子事件的多个原子描述区间，返回 (minStart, maxEnd)。
    """
    all_starts = [iv[0] for intervals in interval_lists for iv in intervals]
    all_ends = [iv[1] for intervals in interval_lists for iv in intervals]
    if not all_starts or not all_ends:
        return []
    return [(min(all_starts), max(all_ends))]

###############################################################################
# 3. mAP 计算函数
###############################################################################
def load_ground_truth(json_file, subset='test'):
    """
    加载 Ground Truth（仅指定 subset）。
    """
    with open(json_file,"r") as f:
        data=json.load(f)
    db=data["database"]
    out={}
    for vid,vinfo in db.items():
        if vinfo["subset"].lower()!=subset:
            continue
        annlist=[]
        for ann in vinfo.get("annotations",[]):
            annlist.append({
                "label":ann["label"],
                "start":ann["segment"][0],
                "end":ann["segment"][1],
                "label_id":ann["label_id"]
            })
        out[vid]=annlist
    return out

def compute_iou(a,b):
    """
    计算两个区间的 IoU。
    """
    s_a,e_a=a
    s_b,e_b=b
    inter_s=max(s_a,s_b)
    inter_e=min(e_a,e_b)
    inter_len=max(0, inter_e-inter_s)
    union_len=max(e_a,e_b)- min(s_a,s_b)
    return inter_len/(union_len+1e-8)

def nms(intervals, iou_threshold=0.5):
    """
    对区间应用非极大抑制（NMS）。
    """
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x.get("score",1.0), reverse=True)
    selected = []
    while sorted_intervals:
        current = sorted_intervals.pop(0)
        selected.append(current)
        sorted_intervals = [
            it for it in sorted_intervals
            if it["cat_id"] != current["cat_id"] or compute_iou((it["start"], it["end"]), (current["start"], current["end"])) < iou_threshold
        ]
    return selected

def compute_mAP(predictions, ground_truths, iou_thresholds=[0.5,0.55,0.6,0.65,0.7]):
    """
    计算 mAP。
    """
    res={}
    for thr in iou_thresholds:
        TP_total=0; FP_total=0; FN_total=0
        for vid,preds in predictions.items():
            gts= ground_truths.get(vid,[])
            gt_map={}
            for g in gts:
                lid= g["label_id"]
                if lid not in gt_map:
                    gt_map[lid]=[]
                gt_map[lid].append((g["start"],g["end"]))

            pred_map={}
            for p in preds:
                lid=p["cat_id"]
                if lid<0:  # ignore
                    continue
                if lid not in pred_map:
                    pred_map[lid]=[]
                pred_map[lid].append((p["start"],p["end"]))

            for lid, intervals_pred in pred_map.items():
                intervals_gt= gt_map.get(lid, [])
                if not intervals_gt:
                    FP_total+= len(intervals_pred)
                    continue
                matched_gt=set()
                for ipred in intervals_pred:
                    best_iou=0
                    best_idx=-1
                    for gidx, ivg in enumerate(intervals_gt):
                        if gidx in matched_gt:
                            continue
                        iou= compute_iou(ipred, ivg)
                        if iou>best_iou:
                            best_iou=iou
                            best_idx=gidx
                    if best_iou>=thr:
                        TP_total+=1
                        matched_gt.add(best_idx)
                    else:
                        FP_total+=1
                FN_total+= (len(intervals_gt)- len(matched_gt))

            for lid, intervals_gt in gt_map.items():
                if lid not in pred_map:
                    FN_total+= len(intervals_gt)
        prec= TP_total/(TP_total+FP_total) if (TP_total+FP_total)>0 else 0
        rec= TP_total/(TP_total+FN_total) if (TP_total+FN_total)>0 else 0
        f1= 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        res[thr]={
            "Precision":prec,
            "Recall":rec,
            "F1-Score":f1
        }
    return res

###############################################################################
# 4. 主测试函数
###############################################################################
def main_test():
    """
    主测试函数，执行整个数据集的零样本动作定位。
    """
    # 1. 设置日志记录
    logging.basicConfig(
        filename="test_all_videos.log",
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )
    logger = logging.getLogger()

    # 2. 初始化加速器
    accelerator = Accelerator()
    device = accelerator.device

    # 3. 加载配置
    with open("config.yaml","r") as f:
        cfg=yaml.load(f, Loader=yaml.FullLoader)

    # 4. 构建 dataset
    dataset=Thumos14Dataset(
        split="test",
        video_folder=cfg['dataset']['video_folder'],
        json_file=cfg['dataset']['json_file'],
        sample_rate=cfg['dataset']['sample_rate'],
        num_frames=cfg['dataset']['num_frames'],
        default_fps=cfg['dataset']['default_fps'],
        max_seq_len=cfg['dataset']['max_seq_len'],
        trunc_thresh=cfg['dataset']['trunc_thresh'],
        crop_ratio=cfg['dataset']['crop_ratio'],
        input_dim=cfg['dataset']['input_dim'],
        num_classes=cfg['dataset']['num_classes'],
        video_ext=cfg['dataset']['video_ext'],
        force_upsampling=cfg['dataset']['force_upsampling']
    )
    data_loader=DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['loader']['num_workers'])
    data_loader=accelerator.prepare(data_loader)

    # 5. 提取 label_mapping
    label_mapping= dataset.label_mapping
    print("类别名称与类别ID的映射：")
    logger.info("类别名称与类别ID的映射：")
    for label_name, lid in label_mapping.items():
        print(f"{label_name} => {lid}")
        logger.info(f"{label_name} => {lid}")

    # 6. 构建 categories 列表
    categories= list(label_mapping.keys())  # e.g. ["CricketBowling","CricketShot",...]

    # 7. 加载 Qwen2-VL 模型与处理器
    logger.info("加载 Qwen2-VL 模型与处理器...")
    print("加载 Qwen2-VL 模型与处理器...")
    try:
        qwen_model= Qwen2VLForConditionalGeneration.from_pretrained(
            cfg['qwen_model_path'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        qwen_processor= AutoProcessor.from_pretrained(cfg['qwen_model_path'], local_files_only=True)
        qwen_model= accelerator.prepare(qwen_model)
    except Exception as e:
        logger.error(f"加载 Qwen2-VL 模型时出错: {e}")
        print(f"加载 Qwen2-VL 模型时出错: {e}")
        return

    # 8. 遍历数据集 => 生成 short_description + detailed_description
    logger.info("开始 Step1: 生成多实例 Caption...")
    print("开始 Step1: 生成多实例 Caption...")
    step1_captions={}
    for batch_item in tqdm(data_loader, desc="Step1: short + detailed"):
        vid= batch_item["video_id"][0]
        frames_np= batch_item["frames"][0]
        frames_pil= frames_to_pil(frames_np)
        if not frames_pil:
            step1_captions[vid]="[]"
            continue

        # 生成 Caption
        try:
            caption_text = generate_caption_with_short_and_detailed(qwen_model, qwen_processor, frames_pil, categories, max_images=1)  # 限制传入的图像数量
            step1_captions[vid]= caption_text
        except Exception as e:
            logger.error(f"生成 Caption 时出错，视频ID: {vid}, 错误: {e}")
            step1_captions[vid]="[]"

    # 9. 写 Step1 结果到文件
    with open("step1_short_detailed_captions.json","w") as f:
        json.dump(step1_captions, f, indent=4)
    print("Step1 完成 => step1_short_detailed_captions.json")
    logger.info("Step1 完成 => step1_short_detailed_captions.json")

    # 10. 加载 SigLIP 模型与处理器
    logger.info("加载 SigLIP 模型与处理器...")
    print("加载 SigLIP 模型与处理器...")
    try:
        siglip_model= AutoModel.from_pretrained(cfg['siglip_model_path'], local_files_only=True).half()
        siglip_model= siglip_model.to(device)
        siglip_processor= AutoProcessor.from_pretrained(cfg['siglip_model_path'], local_files_only=True)
        siglip_model= accelerator.prepare(siglip_model)
        threshold= cfg.get("zero_shot_threshold", 0.5)
    except Exception as e:
        logger.error(f"加载 SigLIP 模型时出错: {e}")
        print(f"加载 SigLIP 模型时出错: {e}")
        return

    # 11. Step2: 解析,拆分,定位,合并
    logger.info("开始 Step2: 拆分 detailed_description 并定位...")
    print("开始 Step2: 拆分 detailed_description 并定位...")
    predictions={}
    for batch_item in tqdm(data_loader, desc="Step2: detail-split-locate"):
        vid= batch_item["video_id"][0]
        frames_np= batch_item["frames"][0]
        frames_pil= frames_to_pil(frames_np)
        text_json= step1_captions.get(vid, "[]")

        # 解析 JSON
        try:
            sub_events= json.loads(text_json)
            if not isinstance(sub_events,list):
                sub_events=[]
        except json.JSONDecodeError:
            logger.error(f"解析 JSON 时出错，视频ID: {vid}")
            sub_events=[]

        intervals_video=[]
        for se in sub_events:
            inst_id= se.get("instance_id","Unknown #?")
            short_desc= se.get("short_description","")
            order= se.get("order",9999)
            detail= se.get("detailed_description","")

            # 拆分 detailed_description
            splitted_actions= split_detailed_description(qwen_model, qwen_processor, inst_id, detail)

            sub_intervals_all=[]
            for action_sent in splitted_actions:
                # 生成 start_query 和 end_query
                q_str_dict= generate_start_end_query_for_sentence(qwen_model, qwen_processor, inst_id, action_sent)
                st_q= q_str_dict.get("start_query","")
                ed_q= q_str_dict.get("end_query","")

                # 使用 SigLIP 定位
                try:
                    intervals_line= locate_sub_event(siglip_model, siglip_processor, frames_pil, st_q, ed_q, threshold=threshold)
                    sub_intervals_all.append(intervals_line)
                except Exception as e:
                    logger.error(f"定位子事件时出错，视频ID: {vid}, 子动作: {action_sent}, 错误: {e}")

            # 合并区间
            merged_sub= merge_atomic_intervals(sub_intervals_all)

            # 提取类别 ID
            base_cat= inst_id.split()[0]  # e.g. "CricketBowling"
            cat_id= label_mapping.get(base_cat, -1)
            if cat_id == -1:
                logger.warning(f"未找到类别 {base_cat} 的 label_id，视频ID: {vid}")
                print(f"警告: 未找到类别 {base_cat} 的 label_id，视频ID: {vid}")

            for (sf,ef) in merged_sub:
                intervals_video.append({
                    "start": sf,
                    "end": ef,
                    "category": inst_id,
                    "cat_id": cat_id,          # 这里用到 label_mapping
                    "order": order,
                    "score": 1.0                # 可根据实际情况调整
                })

        # 应用非极大抑制（NMS）
        intervals_video= nms(intervals_video, iou_threshold=0.5)
        # 按 order 排序并强制顺序
        intervals_video= sorted(intervals_video, key=lambda x:x["order"])
        final=[]
        for iv in intervals_video:
            if not final:
                final.append(iv)
            else:
                last= final[-1]
                if iv["order"] > last["order"]:
                    if iv["start"] >= last["end"]:
                        final.append(iv)
                    else:
                        logging.info(f"discard {iv} due to order conflict with {last}")
                else:
                    final.append(iv)
        predictions[vid]= final
        print(f"Video {vid} => final intervals:", final)

    # 12. 保存预测结果
    output_file= os.path.join(cfg['output_folder'], "predictions_all_videos.json")
    os.makedirs(cfg['output_folder'], exist_ok=True)
    with open(output_file,"w") as f:
        json.dump(predictions,f,indent=4)
    print(f"Step2 完成 => {output_file}")
    logger.info(f"Step2 完成 => {output_file}")

    # 13. 加载 Ground Truth 并计算 mAP
    ground_truth= load_ground_truth(cfg['dataset']['json_file'], subset='test')
    res_map= compute_mAP(predictions, ground_truth, iou_thresholds=[0.5,0.55,0.6,0.65,0.7])
    print("mAP 计算结果:")
    for iou, stats in res_map.items():
        print(f"IoU={iou} => {stats}")
        logger.info(f"IoU={iou} => {stats}")

    print("\n测试完成。")
    logger.info("测试完成。")

###############################################################################
# 5. 执行主测试函数
###############################################################################
if __name__=="__main__":
    main_test()

###############################################################################
# 6. 加载 Ground Truth 函数
###############################################################################
def load_ground_truth(json_file, subset='test'):
    """
    加载 Ground Truth（仅指定 subset）。
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    db = data["database"]
    out = {}
    for vid, vinfo in db.items():
        if vinfo["subset"].lower() != subset:
            continue
        annlist = []
        for ann in vinfo.get("annotations", []):
            annlist.append({
                "label": ann["label"],
                "start": ann["segment"][0],
                "end": ann["segment"][1],
                "label_id": ann["label_id"]
            })
        out[vid] = annlist
    return out
