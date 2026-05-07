"""
修复脚本：纠正 rewrite_data_for_base64.py 搞反的问题。

处理逻辑：
- comp（无图片）：把 base64 输出要求还原为"只返回结果"，删除图片路径相关描述。
- single / 2round（有图片）：添加沙盒代码执行说明 + base64 输出要求。

运行方式:
    python Part2/data/fix_data_for_base64.py
"""
import json
import os


data_dir = os.path.join(os.path.dirname(__file__), "rl_data")

# ========== comp 的修复文本 ==========
# 当前 comp 中可能存在的 base64 描述（rewrite_data_for_base64.py 写入的）
COMP_BASE64_TEXT = (
    "At the end of the code, print the processed image in the exact format: "
    "data:image/png;base64,<base64_string>. "
    "The system will automatically convert it to an image for the next round. "
    "Do not output raw base64 strings without this prefix."
)

# comp 原始 SFT 中的旧文本（刚生成 rl_data 时可能存在）
COMP_OLD_TEXT = (
    "At the end of the code, print the path of the processed image "
    "(processed_path) or the result for further processing in a sandbox environment."
)

# comp 最终应变为：只保留返回结果，完全不提图片路径 / base64
COMP_RESULT_ONLY_TEXT = (
    "At the end of the code, print the result for further processing in a sandbox environment."
)

# ========== single / 2round 的沙盒说明（要求输出文件路径） ==========
SANDBOX_GUIDE = (
    "\n### **Code Execution Guidelines:**\n\n"
    "You may write Python code to process the image. The code will be executed in a secure sandbox, "
    "and its output will be provided back to you for further analysis.\n"
    "* All Python code snippets **must** be wrapped as follows:\n"
    "    <code>\n"
    "    ```python\n"
    "    # Load the image using the User Image Path provided above.\n"
    "    ```\n"
    "    </code>\n"
    "* At the end of the code, save the processed image to `sandbox_temp_dir` and print its full file path. "
    "The system will automatically load it for the next round."
)


def has_image(item: dict) -> bool:
    """判断数据项是否包含图片。"""
    image = item.get("image", [])
    if isinstance(image, list):
        return len(image) > 0
    return bool(image)


def fix_comp_question(question: str) -> str:
    """
    comp 专用：
    - 如果存在 base64 描述，替换为 result-only 描述
    - 如果存在原始旧描述，也替换为 result-only 描述
    """
    modified = False
    if COMP_BASE64_TEXT in question:
        question = question.replace(COMP_BASE64_TEXT, COMP_RESULT_ONLY_TEXT)
        modified = True
    if COMP_OLD_TEXT in question:
        question = question.replace(COMP_OLD_TEXT, COMP_RESULT_ONLY_TEXT)
        modified = True
    return question, modified


def fix_image_question(question: str) -> str:
    """
    single / 2round 专用：
    - 将旧的 base64 输出要求替换为文件路径输出要求
    - 如果没有沙盒说明，在 Output Format 之前插入
    """
    # 如果已经包含路径要求，认为已经是正确状态
    if "save the processed image to `sandbox_temp_dir`" in question:
        return question, False
    
    # 如果存在旧的 base64 描述，替换为新的路径要求
    old_base64_text = (
        "* At the end of the code, print the processed image in the exact format: "
        "data:image/png;base64,<base64_string>. "
        "The system will automatically convert it to an image for the next round. "
        "Do not output raw base64 strings without this prefix."
    )
    if old_base64_text in question:
        question = question.replace(
            old_base64_text,
            "* At the end of the code, save the processed image to `sandbox_temp_dir` and print its full file path. "
            "The system will automatically load it for the next round."
        )
        # 同时更新代码注释
        question = question.replace(
            "# Use 'user_image_path' to load the original image.",
            "# Load the image using the User Image Path provided above."
        )
        return question, True
    
    # 如果没有沙盒说明，在 Output Format 之前插入
    if "### **Output Format" in question:
        question = question.replace(
            "### **Output Format",
            SANDBOX_GUIDE + "\n\n### **Output Format"
        )
        return question, True
    
    # 兜底：如果找不到 Output Format，追加到末尾
    question += "\n" + SANDBOX_GUIDE
    return question, True


def main():
    files = {
        "single_local.json": fix_image_question,
        "2round_local.json": fix_image_question,
        "comp_local.json": fix_comp_question,
    }

    for fname, fix_fn in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"[SKIP] File not found: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        modified_count = 0
        new_lines = []

        for line in lines:
            item = json.loads(line)
            question = item["question"]
            new_question, modified = fix_fn(question)
            if modified:
                item["question"] = new_question
                modified_count += 1
            new_lines.append(json.dumps(item, ensure_ascii=False) + "\n")

        if modified_count > 0:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"[MODIFIED] {fname}: {modified_count}/{len(lines)} items updated.")
        else:
            print(f"[NO CHANGE] {fname}: all items already correct.")

    print("\nDone.")


if __name__ == "__main__":
    main()
