import json
import sys

json1_path = "../text/error-3-stage.json"
json2_path = "../text/a1-3-stage.json"
output_path = "../text/stage_3.json"

def merge_json(json1_path, json2_path, output_path):
    """
    合并两个JSON文件，将json1的内容更新到json2中
    """
    try:
        # 读取第一个JSON文件
        with open(json1_path, 'r', encoding='utf-8') as f:
            json1_data = json.load(f)

        # 读取第二个JSON文件
        with open(json2_path, 'r', encoding='utf-8') as f:
            json2_data = json.load(f)

        # 合并JSON数据
        # 遍历json1中的每个键值对
        for key, value in json1_data.items():
            # 如果键已存在于json2中，则更新该键的值
            if key in json2_data:
                json2_data[key] = value
            # 否则，将新键值对添加到json2中
            else:
                json2_data[key] = value

        # 将合并后的JSON数据写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json2_data, f, ensure_ascii=False, indent=2)

        print(f"合并完成！结果已保存到 {output_path}")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误：JSON格式无效 - 请检查文件 {json1_path} 和 {json2_path}")
        sys.exit(1)
    except Exception as e:
        print(f"发生未知错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    # 默认文件路径


    # 可以通过命令行参数指定文件路径
    if len(sys.argv) > 1:
        json1_path = sys.argv[1]
    if len(sys.argv) > 2:
        json2_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]

    print(f"正在合并 {json1_path} 和 {json2_path}...")
    merge_json(json1_path, json2_path, output_path)    