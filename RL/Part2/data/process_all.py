"""
完整数据处理流水线：
1. SFT → RL 格式转换（提取 image/question/answer）
2. 2round 和 single 的 base64 图片落地为本地文件
3. 替换 question 中的路径为真实相对路径
4. comp 直接转换（无图片）
"""
import json
import os
import re
import base64
import io
from PIL import Image

# ==================== 配置 ====================
SFT_DIR = "Part2/data/sft_data"
RL_DIR = "Part2/data/rl_data"
IMG_DIR = "Part2/data/images"

os.makedirs(RL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


def extract_answer(response: str) -> str | None:
    """从 response 中提取 <answer>xxx</answer> 的内容"""
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    return match.group(1).strip() if match else None


def decode_base64_image(image_field):
    """解码 base64 图片字段，返回 PIL Image"""
    if not image_field:
        return None
    raw = image_field[0] if isinstance(image_field, list) else image_field
    if not isinstance(raw, str):
        return None
    try:
        if "," in raw:
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)
        return Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        print(f"    解码失败: {e}")
        return None


def extract_old_path(question: str) -> str | None:
    """从 question 中提取旧的 User Image Path"""
    match = re.search(r'### User Image Path:\*\*\s*"([^"]+)"', question)
    return match.group(1) if match else None


def replace_path_in_question(question: str, new_path: str) -> str:
    """把 question 中的旧路径替换为新相对路径"""
    question = re.sub(
        r'(### User Image Path:\*\*\s*)"[^"]+"',
        rf'\1"{new_path}"',
        question
    )
    return question


def process_split(split_name: str, has_image: bool):
    """处理一个 split：SFT → RL + 图片落地"""
    input_path = os.path.join(SFT_DIR, f"{split_name}.json")
    output_path = os.path.join(RL_DIR, f"{split_name}_local.json")

    print(f"\n{'='*50}")
    print(f"处理 {split_name}: {input_path}")
    print(f"{'='*50}")

    processed = []
    success = 0
    fail = 0
    no_answer = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            question = item["question"]
            response = item["response"]
            image_field = item.get("image", [])

            # 提取 answer
            answer = extract_answer(response)
            if answer is None:
                no_answer += 1
                # 用 response 的最后 50 字兜底
                answer = response.strip()[-50:]

            # 处理图片（2round 和 single）
            if has_image:
                pil_img = decode_base64_image(image_field)
                if pil_img is None:
                    print(f"  [{idx:03d}] ❌ 图片解码失败，跳过")
                    fail += 1
                    continue

                new_filename = f"{split_name}_{idx:03d}.jpg"
                new_abs_path = os.path.join(IMG_DIR, new_filename)
                new_rel_path = f"./Part2/data/images/{new_filename}"

                pil_img.save(new_abs_path, "JPEG")
                new_question = replace_path_in_question(question, new_rel_path)

                new_item = {
                    "image": new_rel_path,
                    "question": new_question,
                    "answer": answer,
                }
            else:
                # comp：无图片
                new_item = {
                    "image": [],
                    "question": question,
                    "answer": answer,
                }

            processed.append(new_item)
            success += 1

    # 保存 RL 数据
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  成功: {success}, 失败: {fail}, 无answer兜底: {no_answer}")
    print(f"  输出: {output_path}")
    return success


# ==================== 执行处理 ====================
if __name__ == "__main__":
    total = 0
    total += process_split("2round", has_image=True)
    total += process_split("comp", has_image=False)
    total += process_split("single", has_image=True)

    print(f"\n{'='*50}")
    print(f"全部处理完成！总计: {total} 条")
    print(f"图片保存到: {IMG_DIR}")
    print(f"RL 数据保存到: {RL_DIR}")

    # 验证
    print(f"\n--- 验证输出 ---")
    for split in ["2round_local", "comp_local", "single_local"]:
        path = os.path.join(RL_DIR, f"{split}.json")
        with open(path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"  {split}.json: {count} 条")
