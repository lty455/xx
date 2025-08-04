import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Specify the model path and device
model_path = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
de = "cuda:0"

# Load the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=de,
    local_files_only=True  # Ensure only local files are used
)

# Load the processor
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)


# Function to process the action and caption
def process_action_and_caption(action, caption):
    # Build the prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Here is an action: {action}\n"
                        f"Caption: {caption}\n"
                        "Please follow these steps to create a start question, end question, and description for the action:\n"
                        "1. Consider what is unique about the very beginning of the action that could be observed in a single video frame. Craft a yes/no question about that.\n"
                        "2. Consider what is unique about the very end of the action that could be observed in a single video frame. Craft a yes/no question about that. Make sure it does not overlap with the start question.\n"
                        "3. Use a sentence to summarize the key components of the action, without using adverbs. The description should differentiate the action from other actions.\n"
                        "Output your final answer in this JSON format:\n"
                        "{\n"
                        f'"{action}": {{\n'
                        '"start": "question",\n'
                        '"end": "question",\n'
                        '"description": "description"\n'
                        "}}\n"
                        "}\n"
                        "Make sure to follow the JSON formatting exactly, with the action in <action>.\n"
                        "Do not add any other elements to the JSON. Only include the start question, end question, and description.\n"
                        "Do not include any explanatory text or preamble before the JSON. Only output the JSON."
                    ),
                },
            ],
        }
    ]

    # Process input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    # Transfer to CUDA
    inputs = inputs.to(de)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    # Extract the generated content
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# Read video names, actions, and captions from the text file
input_file = "./output/acap1.txt"  # Replace with the path to your txt file
output_data = {}

with open(input_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        video_name, action, caption = line.strip().split('--', 2)

        # Process each action and caption
        output_json = process_action_and_caption(action, caption)

        # Parse the generated JSON
        try:
            json_data = json.loads(output_json)
            output_data[video_name] = json_data
            print(f"{video_name}-yes")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for video {video_name}: {e}")

# Save the output to a JSON file
output_file = "anet.json"
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Output saved to {output_file}")
