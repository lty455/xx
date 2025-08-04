import os
import json
import torch
import cv2
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "./activitynet/videos/v1-2/val"
output_json_file = "./output/base_anet.json"
device = "cuda:0"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True
)

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)


def calculate_fps(video_path, target_frames=35):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("Failed to get total frames from the video.")
    fps = 30 * target_frames / total_frames
    return fps


def clean_json_output(model_output):
    """
    清理模型输出，去除 ```json 和 ```，并解析 JSON
    """
    model_output = model_output.strip()
    if model_output.startswith("```json"):
        model_output = model_output[7:].strip()
    if model_output.endswith("```"):
        model_output = model_output[:-3].strip()
    try:
        return json.loads(model_output)
    except json.JSONDecodeError:
        raise ValueError("Model output is not valid JSON.")


def process_video(video_path, video_name):
    fps = calculate_fps(video_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": (
                        "The video contains multiple instances of various athletic actions including"
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
                        "For each action instance, output in JSON format: "
                        "[{'label': 'action', 'segment': [start_time, end_time]}]. "
                        "List all action instances separately without any other words."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return clean_json_output(output_text)


def main():
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    video_annotations = {}

    for video_name in os.listdir(video_dir):
        if video_name.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_name)
            try:
                annotations = process_video(video_path, video_name)
                video_annotations[video_name] = {"annotations": annotations}
                print(f"{video_name} - 成功")
            except Exception as e:
                print(f"{video_name} - 失败: {e}")

    with open(output_json_file, "w") as json_file:
        json.dump(video_annotations, json_file, indent=4)

    print(f"所有视频的类别已保存到 {output_json_file}")


if __name__ == "__main__":
    main()
