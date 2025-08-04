import torch
import cv2
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor,Qwen2_5_VLForConditionalGeneration

# 指定模型和处理器的本地路径
video_path="./activitynet/videos/v1-2/val/v_q4QPF-qNBTY.mp4"
# video_path = "./thumos/videos/video_test_0000270.mp4"
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct"

device="cuda"
# Load the model from the local path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True  # 确保只从本地加载文件
)

# Load the processor from the local path
processor = AutoProcessor.from_pretrained(model_path,use_fast=True)
def calculate_fps(video_path, target_frames=600):
    """
    动态计算 fps，以满足目标提取帧数量
    :param video_path: 视频路径
    :param target_frames: 目标提取帧数量
    :return: 动态计算的 fps
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    fps = target_frames / total_frames
    return fps

# 计算动态 fps
dynamic_fps = calculate_fps(video_path)
# 定义消息，其中包括视频和文本内容
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": dynamic_fps,  # 动态计算的 fps
            },
            {
                "type": "text",
                "text": (
                        "The video may contain multiple instances of various athletic actions,the action included"
                        ###thumos
                        # "CricketBowling, CricketShot, VolleyballSpiking, "
                        # "JavelinThrow,  Shotput, TennisSwing, GolfSwing, ThrowDiscus, Billiards, CleanAndJerk, "
                        # "LongJump, Diving, CliffDiving, BasketballDunk, HighJump, HammerThrow, SoccerPenalty, "
                        # "BaseballPitch, FrisbeeCatch, PoleVault. "
                        ######activitynet
                        "Beer pong,Kneeling,Tumbling,Sharpening knives,Playing water polo,Scuba diving,"
                        "Arm wrestling,Playing bagpipes,Riding bumper cars,Surfing,Hopscotch,Gargling mouthwash,"
                        "Playing violin,Plastering,Changing car wheel,Horseback riding,Playing congas,Walking the dog,"
                        "Rafting,Hurling,Removing curlers,Playing beach volleyball,Windsurfing,Using parallel bars,"
                        "Playing drums,Playing badminton,Getting a piercing,Camel ride,Sailing,Wrapping presents,"
                        "Hand washing clothes,Braiding hair,Longboarding,Doing motocross,Vacuuming floor,"
                        "Blow-drying hair,Cricket,Smoking hookah,Doing fencing,Playing harmonica,Spinning,"
                        "Playing blackjack,Discus throw,Playing flauta,Swimming,Ice fishing,Spread mulch,"
                        "Canoeing,Mowing the lawn,Capoeira,Trimming branches or hedges,"
                        "Preparing salad,Beach soccer,BMX,Playing kickball,Shoveling snow,Cheerleading,"
                        "Removing ice from car,Calf roping,Breakdancing,Mopping floor,Powerbocking,"
                        "Kite flying,Getting a tattoo,Cleaning shoes,Running a marathon,Shaving legs,"
                        "Starting a campfire,River tubing,Zumba,Putting on makeup,Playing ten pins,Raking leaves,"
                        "Doing karate,High jump,Futsal,Grooming dog,Wakeboarding,Swinging at the playground,"
                        "Playing lacrosse,Archery,Playing saxophone,Long jump,Paintball,Tango,Rope skipping,"
                        "Throwing darts,Roof shingle removal,Ping-pong,Making a sandwich,Tennis serve with ball bouncing,"
                        "Triple jump,Skiing,Peeling potatoes,Doing step aerobics,Building sandcastles,Elliptical trainer,"
                        "Baking cookies,Rock-paper-scissors,Playing piano,Snowboarding,Preparing pasta,Croquet,"
                        "Playing guitarra,Cleaning windows,Skateboarding,Playing squash,Polishing shoes,"
                        "Smoking a cigarette,Installing carpet,Using the balance beam,Drum corps,Playing polo,"
                        "Hammer throw,Baton twirling,Using uneven bars,Doing crunches,Tai chi,Kayaking,Doing a powerbomb,"
                        "Grooming horse,Using the pommel horse,Belly dance,Clipping cat claws,Putting in contact lenses,"
                        "Playing ice hockey,Tug of war,Brushing hair,Welding,Mixing drinks,"
                        "Washing hands,Having an ice cream,Chopping wood,Platform diving,Layup drill in basketball,"
                        "Clean and jerk,Hitting a pinata,Snow tubing,Decorating the Christmas tree,Pole vault,"
                        "Washing face,Hand car wash,Doing kickboxing,Fixing the roof,Dodgeball,Playing pool,"
                        "Assembling bicycle,Shuffleboard,Curling,Bullfighting,Rollerblading,Snatch,Disc dog,"
                        "Fixing bicycle,Polishing furniture,Javelin throw,Playing accordion,Bathing dog,Washing dishes,"
                        "Playing racquetball,Shaving,Shot put,Drinking coffee,Hanging wallpaper,Cumbia,Springboard diving,"
                        "Ballet,Rock climbing,Ironing clothes,Drinking beer,Blowing leaves,Using the monkey bar,"
                        "Fun sliding down,Playing field hockey,Getting a haircut,Hula hoop,Waterskiing,"
                        "Carving jack-o-lanterns,Doing nails,Cutting the grass,Sumo,Making a cake,Painting fence,"
                        "Using the rowing machine,Brushing teeth,Applying sunscreen,Making a lemonade,Painting furniture,"
                        "Painting,Putting on shoes,Volleyball,Knitting,Making an omelette,Playing rubik's cube,"
                        "Cleaning sink,Bungee jumping,Slacklining,Table soccer,Waxing skis,Laying tile."
                        "Only list the action category that appear in the video without other words."
                ),
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

# Increase the max_new_tokens parameter to allow for longer output
generated_ids = model.generate(**inputs, max_new_tokens=128)  # Adjusted for more concise output
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# 输出模型生成的动作类别列表
print(output_text[0])
