import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from collections import defaultdict

# 配置参数
MODEL_PATH = "/root/.cache/modelscope/hub/Qwen/Qwen2___5-VL-7B-Instruct"
DEVICE = "cuda:0"
INPUT_FILE = "./output/thumos14_caption.txt"
OUTPUT_FILE = "./output/new_stages_t14.json"

# 初始化模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)


def generate_action_stages(action, caption):
    """生成带独有标记的动作阶段分解"""
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"""Analyze the action and break it into distinct stages. Follow these rules:
                        1. Action: {action}
                        2. Context: {caption}
                        3. Decompose into 2-5 essential stages
                        4. Each stage should be the people do sth.
                        5. Avoid overlapping stages
                        6. Maintain chronological order
                        7. Mark stages unique to this action with * (only if the stage is distinctive and unlikely to appear in other actions)
                        Output format:                        

                        Long jump:People are running up,people jump*,People jumping in the air*,People landing
                        
                        ONLY output the text in that format without any extra content."""
        }]
    }]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 处理格式并验证
        generated_text = generated_text.replace('\n', ' ').replace(' ,', ',').strip()
        if ':' not in generated_text:
            raise ValueError("格式错误: 缺少冒号分隔符")

        # 解析带*标记的阶段
        _, stages_part = generated_text.split(':', 1)
        stages_list = [stage.strip() for stage in stages_part.split(',') if stage.strip()]

        return {action: {"stages": stages_list}}
    except Exception as e:
        print(f"生成错误: {str(e)}")
        return {action: {"stages": []}}


# 其余函数保持原样...
def process_input_file():
    """处理输入文件并生成结构化的输出"""
    output_data = defaultdict(dict)

    with open(INPUT_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析行数据，格式为 video_name -- action -- caption
                video_name, action, caption = line.strip().split('--', 2)
                action = action.strip()

                # 生成阶段分解
                result = generate_action_stages(action, caption)

                # 合并结果
                if action in result:
                    output_data[video_name][action] = result[action]
                    print(f"已处理: {video_name} - {action}")
                else:
                    print(f"结果中缺少动作: {video_name}:{action}")

            except ValueError:
                print(f"格式错误 @ 行 {line_num}: {line.strip()}")
            except json.JSONDecodeError:
                print(f"JSON解析失败 @ 行 {line_num}")

    # 转换默认字典为普通字典
    return dict(output_data)


if __name__ == "__main__":
    # 执行处理流程
    final_output = process_input_file()

    # 保存结果为 JSON 文件
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"成功生成文件: {OUTPUT_FILE}")
    print(f"处理视频总数: {len(final_output)}")
    print(f"总动作类型: {sum(len(v) for v in final_output.values())}")