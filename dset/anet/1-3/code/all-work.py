import os
import time
import torch
import cv2
import multiprocessing as mp
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "/home/uestcxr/.cache/modelscope/hub/models/Qwen/Qwen2___5-VL-7B-Instruct"
video_dir = "/mnt/disc1/val"
input_txt_file = "./output/error_videos.txt"
output_txt_file = "./output/error_3_category.txt"
device = "cuda:0"
TIMEOUT = 120

model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

def calculate_fps(video_path, target_frames=20):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames == 0:
        raise ValueError("No frames in video.")
    return 30 * target_frames / total_frames

def process_single_video(video_path):
    load_model()
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
                        "please return the action name without any extra words. "
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def video_worker(video_list, result_queue):
    for video_path in video_list:
        try:
            result = process_single_video(video_path)
            result_queue.put(("success", os.path.basename(video_path), result))
        except Exception as e:
            result_queue.put(("error", os.path.basename(video_path), str(e)))


def load_target_videos():
    with open(input_txt_file, 'r') as f:
        return set(line.strip() + '.mp4' for line in f if line.strip())


def load_done_videos(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r') as f:
        return set(line.split('-')[0].strip() for line in f if '-' in line)


def run_with_timeout(video_list):
    result_queue = mp.Queue()
    process = mp.Process(target=video_worker, args=(video_list, result_queue))
    process.start()
    process.join(TIMEOUT)

    results = []
    if process.is_alive():
        process.terminate()
        process.join()
        print(f"[Timeout] Child process killed after {TIMEOUT}s")
    else:
        while not result_queue.empty():
            results.append(result_queue.get())

    return results


def main():
    mp.set_start_method("spawn")
    target_videos = load_target_videos()
    done_videos = load_done_videos(output_txt_file)
    all_videos = [f for f in os.listdir(video_dir) if
                  f.endswith(".mp4") and f in target_videos and f not in done_videos]
    all_videos_paths = [os.path.join(video_dir, f) for f in all_videos]

    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)

    with open(output_txt_file, 'a') as f:
        i = 0
        while i < len(all_videos_paths):
            sublist = all_videos_paths[i:i + 10]  # 每轮处理10个
            results = run_with_timeout(sublist)

            for status, video_name, info in results:
                if status == "success":
                    f.write(f"{video_name} - {info}\n")
                    f.flush()
                    print(f"{video_name}-yyyyyyyyy")
                else:
                    print(f"{video_name}-nnnnnnnnn{info}")
            i += 10

    print(f"[Done] Results saved to {output_txt_file}")


if __name__ == "__main__":
    main()