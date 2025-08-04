import os
import torch
import cv2
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 模型和处理器的本地路径
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "./activitynet/videos/v1-2/val"
output_txt_file = "./output/anet_second_category.txt"
video_list_file = "./output/wrong_anet.txt"  # Path to your video list text file
device = "cuda:0"

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)

# 加载处理器
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)


def calculate_fps(video_path, target_frames=600):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    return target_frames / total_frames


def process_video(video_path):
    dynamic_fps = calculate_fps(video_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": dynamic_fps,
                },
                {
                    "type": "text",
                    "text": (
                        "The video contains one and only one action. "
                        "please return the action name without any extra words"
                        "Only return the action number without any extra words. "
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
                    ),
                },
            ],
        }
    ]

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

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()


def read_video_list(file_path):
    with open(file_path, 'r') as f:
        # Read lines, strip whitespace, and remove empty lines
        return [line.strip() for line in f if line.strip()]


def main():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)

    # Read the list of videos to process
    video_list = read_video_list(video_list_file)

    # Add .mp4 extension to video names if not present
    video_list = [v if v.endswith('.mp4') else f"{v}.mp4" for v in video_list]

    with open(output_txt_file, "w") as txt_file:
        for video_name in os.listdir(video_dir):
            # Only process videos that are in our list
            if video_name.endswith(".mp4") and video_name in video_list:
                video_path = os.path.join(video_dir, video_name)
                try:
                    categories = process_video(video_path)
                    txt_file.write(f"{video_name} - {categories}\n")
                    print(f"{video_name} - yyyyyyy")
                except Exception as e:
                    print(f"{video_name} - nnnnnnnn: {e}")

    print(f"all had to {output_txt_file}")


if __name__ == "__main__":
    main()