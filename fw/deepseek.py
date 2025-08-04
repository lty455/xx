import json
from collections import defaultdict
from openai import OpenAI

# 配置参数
API_KEY = "sk-5c49cbc523614be8a145a80c023d33d3"
BASE_URL = "https://api.deepseek.com"
INPUT_FILE = "./output/anet_caption2.txt"
OUTPUT_FILE = "./output/new_stages1_anet.json"

# 初始化 DeepSeek 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_action_stages(action, caption):
    """使用 DeepSeek 生成带独有标记的动作阶段分解"""
    prompt = f"""Analyze the action and break it into distinct stages. Follow these rules:
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
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "You are a helpful assistant"},
                      {"role": "user", "content": prompt}],
            temperature=0.5,
            stream=False
        )
        generated_text = response.choices[0].message.content.strip()
        generated_text = generated_text.replace('\n', ' ').replace(' ,', ',').strip()

        # 解析返回的文本
        if ':' not in generated_text:
            raise ValueError("格式错误: 缺少冒号分隔符")
        _, stages_part = generated_text.split(':', 1)
        stages_list = [stage.strip() for stage in stages_part.split(',') if stage.strip()]
        return {action: {"stages": stages_list}}
    except Exception as e:
        print(f"生成错误: {str(e)}")
        return {action: {"stages": []}}


def process_input_file():
    """改进后的输入处理逻辑"""
    output_data = defaultdict(dict)
    with open(INPUT_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                line = line.strip()
                # 智能分割逻辑
                if '.mp4--' not in line:
                    raise ValueError("视频名未包含 .mp4-- 分隔符")
                video_part, remaining = line.split('.mp4--', 1)
                video_name = video_part + '.mp4'

                # 二次分割动作和描述
                if '--' not in remaining:
                    raise ValueError("缺少动作或描述分隔符")
                action, caption = remaining.split('--', 1)
                action, caption = action.strip(), caption.strip()

                # 生成阶段分解
                result = generate_action_stages(action, caption)
                if action in result:
                    output_data[video_name][action] = result[action]
                    print(f"已处理: {video_name} - {action}")
                else:
                    print(f"结果中缺少动作: {video_name}:{action}")
            except Exception as e:
                print(f"处理错误 @ 行 {line_num}: {str(e)} - 行内容: {line}")

    return dict(output_data)


if __name__ == "__main__":
    final_output = process_input_file()
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"成功生成文件: {OUTPUT_FILE}")
    print(f"处理视频总数: {len(final_output)}")
    print(f"总动作类型: {sum(len(v) for v in final_output.values())}")