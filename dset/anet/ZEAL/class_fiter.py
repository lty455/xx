from PIL import Image
import cv2
import torch
import os
import json
from modelscope import AutoProcessor, AutoModel
from tqdm import tqdm

# 检查CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型和处理器
model = AutoModel.from_pretrained("/root/.cache/modelscope/hub/AI-ModelScope/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("/root/.cache/modelscope/hub/AI-ModelScope/siglip-so400m-patch14-384")

# 将模型移至指定设备
model = model.to(device)

# 动作类别列表
action_categories = [
    "Applying sunscreen", "Archery", "Arm wrestling", "Assembling bicycle", "BMX",
    "Baking cookies", "Ballet", "Bathing dog", "Baton twirling", "Beach soccer",
    "Beer pong", "Belly dance", "Blow-drying hair", "Blowing leaves", "Braiding hair",
    "Breakdancing", "Brushing hair", "Brushing teeth", "Building sandcastles", "Bullfighting",
    "Bungee jumping", "Calf roping", "Camel ride", "Canoeing", "Capoeira",
    "Carving jack-o-lanterns", "Changing car wheel", "Cheerleading", "Chopping wood", "Clean and jerk",
    "Cleaning shoes", "Cleaning sink", "Cleaning windows", "Clipping cat claws", "Cricket",
    "Croquet", "Cumbia", "Curling", "Cutting the grass", "Decorating the Christmas tree",
    "Disc dog", "Discus throw", "Dodgeball", "Doing a powerbomb", "Doing crunches",
    "Doing fencing", "Doing karate", "Doing kickboxing", "Doing motocross", "Doing nails",
    "Doing step aerobics", "Drinking beer", "Drinking coffee", "Drum corps", "Elliptical trainer",
    "Fixing bicycle", "Fixing the roof", "Fun sliding down", "Futsal", "Gargling mouthwash",
    "Getting a haircut", "Getting a piercing", "Getting a tattoo", "Grooming dog", "Grooming horse",
    "Hammer throw", "Hand car wash", "Hand washing clothes", "Hanging wallpaper", "Having an ice cream",
    "High jump", "Hitting a pinata", "Hopscotch", "Horseback riding", "Hula hoop",
    "Hurling", "Ice fishing", "Installing carpet", "Ironing clothes", "Javelin throw",
    "Kayaking", "Kite flying", "Kneeling", "Knitting", "Laying tile",
    "Layup drill in basketball", "Long jump", "Longboarding", "Making a cake", "Making a lemonade",
    "Making a sandwich", "Making an omelette", "Mixing drinks", "Mooping floor", "Mowing the lawn",
    "Paintball", "Painting", "Painting fence", "Painting furniture", "Peeling potatoes",
    "Ping-pong", "Plastering", "Plataform diving", "Playing accordion", "Playing badminton",
    "Playing bagpipes", "Playing beach volleyball", "Playing blackjack", "Playing congas", "Playing drums",
    "Playing field hockey", "Playing flauta", "Playing guitarra", "Playing harmonica", "Playing ice hockey",
    "Playing kickball", "Playing lacrosse", "Playing piano", "Playing polo", "Playing pool",
    "Playing racquetball", "Playing rubik cube", "Playing saxophone", "Playing squash", "Playing ten pins",
    "Playing violin", "Playing water polo", "Pole vault", "Polishing forniture", "Polishing shoes",
    "Powerbocking", "Preparing pasta", "Preparing salad", "Putting in contact lenses", "Putting on makeup",
    "Putting on shoes", "Rafting", "Raking leaves", "Removing curlers", "Removing ice from car",
    "Riding bumper cars", "River tubing", "Rock climbing", "Rock-paper-scissors", "Rollerblading",
    "Roof shingle removal", "Rope skipping", "Running a marathon", "Sailing", "Scuba diving",
    "Sharpening knives", "Shaving", "Shaving legs", "Shot put", "Shoveling snow",
    "Shuffleboard", "Skateboarding", "Skiing", "Slacklining", "Smoking a cigarette",
    "Smoking hookah", "Snatch", "Snow tubing", "Snowboarding", "Spinning",
    "Spread mulch", "Springboard diving", "Starting a campfire", "Sumo", "Surfing",
    "Swimming", "Swinging at the playground", "Table soccer", "Tai chi", "Tango",
    "Tennis serve with ball bouncing", "Throwing darts", "Trimming branches or hedges", "Triple jump", "Tug of war",
    "Tumbling", "Using parallel bars", "Using the balance beam", "Using the monkey bar", "Using the pommel horse",
    "Using the rowing machine", "Using uneven bars", "Vacuuming floor", "Volleyball", "Wakeboarding",
    "Walking the dog", "Washing dishes", "Washing face", "Washing hands", "Waterskiing",
    "Waxing skis", "Welding", "Windsurfing", "Wrapping presents", "Zumba"
]


# 视频处理函数
def process_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing {video_path}, FPS: {fps}")

    frame_count = 0
    processed_frame_count = 0
    total_scores = None

    while cap.isOpened() and processed_frame_count < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % round(fps) == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            inputs = processor(text=action_categories, images=pil_image, padding="max_length", return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image)

            if total_scores is None:
                total_scores = probs[0].cpu().numpy()
            else:
                total_scores += probs[0].cpu().numpy()

            processed_frame_count += 1


        frame_count += 1

    cap.release()
    print(f"  Total frames processed: {processed_frame_count}")

    if total_scores is None:
        return None

    top_indices = total_scores.argsort()[-5:][::-1]
    top_actions = [action_categories[i] for i in top_indices]
    return top_actions


# 读取已存在的JSON结果或创建新的
def load_or_create_results(json_path):
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}. Creating new results file.")
    return {}


# 保存单个视频结果到JSON文件
def save_single_result(results, json_path):
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)


# 处理目录下的所有视频
def process_directory(directory, json_path, max_frames=30):
    results = load_or_create_results(json_path)
    video_files = [f for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mkv'))]

    # 过滤掉已经处理过的视频
    processed_videos = set(results.keys())
    videos_to_process = [f for f in video_files if os.path.splitext(f)[0] not in processed_videos]

    print(f"Found {len(video_files)} video files in {directory}")
    print(f"{len(processed_videos)} videos already processed, {len(videos_to_process)} to process")

    for video_file in tqdm(videos_to_process, desc="Processing videos"):
        video_path = os.path.join(directory, video_file)
        video_name = os.path.splitext(video_file)[0]

        top_actions = process_video(video_path, max_frames)
        results[video_name] = top_actions or []

        # 处理完一个视频后立即保存结果
        save_single_result(results, json_path)
        print(f"Results for {video_name} saved to {json_path}")

    return results


# 使用示例
if __name__ == "__main__":
    directory = "./activitynet/videos/v1-2/val"  # 替换为实际目录路径
    json_output = "actions.json"
    max_frames = 30

    print(f"Starting to process videos in {directory}...")
    results = process_directory(directory, json_output, max_frames)

