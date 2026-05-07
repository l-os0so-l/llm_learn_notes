"""
沙箱执行模块（完善版）：
1. 安全代码检查（白名单 import、危险操作过滤）
2. 多轮持续运行（检测到 <answer> 前循环执行代码）
3. 工作目录隔离
4. 输出智能截断（防止超长报错/stdout 污染对话历史）
"""
import re
import subprocess
import tempfile
import os
import ast
from typing import Optional, List, Dict, Tuple
from PIL import Image
import base64
import io
# ==================== 安全配置 ====================
# 允许的 import 模块白名单
IMPORT_WHITELIST = {
    'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFilter',
    'cv2', 'numpy', 'np', 'matplotlib', 'matplotlib.pyplot', 'plt',
    'math', 'random', 'statistics', 'fractions', 'decimal',
    'json', 'csv', 'io', 'base64', 'copy', 'itertools', 'collections',
    'os', 'os.path',
}

# 危险函数黑名单（模块.函数 或 函数名）
BANNED_CALLS = {
    'os.system', 'os.popen', 'os.spawn', 'os.exec', 'os.fork', 'os.kill',
    'subprocess.call', 'subprocess.run', 'subprocess.Popen',
    'subprocess.check_output', 'subprocess.check_call',
    'eval', 'exec', 'compile', '__import__',
}

# 文件写入函数：如果目标路径在 WORK_DIR（原始图片目录）而非临时目录，则拒绝
WRITE_CALLS = {
    'plt.savefig', 'cv2.imwrite', 'cv2.imwrite',
}

# 允许的工作目录（代码只能在这里读写）
WORK_DIR = os.path.abspath("./Part2/data/images")
# 图片输出临时目录（由外部训练循环每 step 提供）
_TEMP_DIR = None

def set_temp_dir(path: Optional[str]):
    """设置当前 step 的临时输出目录。由 train.py 在每 step 开始时调用。"""
    global _TEMP_DIR
    _TEMP_DIR = os.path.abspath(path) if path else None


# ==================== 输出截断配置 ====================
MAX_SANDBOX_OUTPUT = 400  # 最终进入对话历史的输出上限
EXEC_TRUNCATE = 1500      # subprocess 原始输出先截断到此，防止内存中存超长字符串

def _is_base64_image(text: str) -> bool:
    """检测文本是否为 data:image/...;base64,... 格式的图片数据（避免截断破坏编码）"""
    text = text.strip()
    return text.startswith('data:image/') or bool(re.search(r'data:image/[^;]+;base64,', text))


def extract_base64_images(text: str) -> Tuple[str, List]:
    """
    从文本中提取 base64 编码的图片（仅限 data:image/...;base64,... 格式），解码为 PIL.Image 后返回。
    不再支持裸 base64 字符串——模型必须在 prompt 指导下使用带前缀的格式。

    Returns:
        (剩余文本, 图片列表)
    """
    images = []
    # 匹配 data:image/xxx;base64,xxx 格式，base64 数据可能包含换行/空格/制表符
    pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)'
    matches = list(re.finditer(pattern, text))
    if matches:
        # 移除所有 base64 数据（含跨行），保留其他文本
        cleaned = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+', '[Image output]', text)
        for m in matches:
            try:
                # 解码前去掉 base64 中的空白字符（换行、空格、制表符等）
                b64_clean = m.group(1).replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
                img_data = base64.b64decode(b64_clean)
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                images.append(img)
            except Exception:
                continue
        return cleaned, images

    return text, []


def _raw_truncate(text: str, max_len: int = EXEC_TRUNCATE) -> str:
    """
    原始执行输出的安全阀截断。
    不再做简单的前缀 [:1500] 截断，而是首尾保留，避免丢失错误栈尾部。
    """
    if len(text) <= max_len:
        return text
    # 错误信息尾部更重要，多保留尾部
    if "Error" in text or "Traceback" in text or "exception" in text.lower():
        head_len = max_len // 4
        tail_len = max_len * 3 // 4
    else:
        head_len = max_len // 2
        tail_len = max_len // 2
    return (
        f"{text[:head_len]}\n"
        f"... [原始输出过长：{len(text)} 字符，已截断] ...\n"
        f"{text[-tail_len:]}"
    )


def _smart_truncate(text: str, max_len: int = MAX_SANDBOX_OUTPUT) -> str:
    """
    智能截断长文本。
    - 正常输出：保留头尾，中间省略
    - 超长报错：优先保留尾部（实际错误原因通常在最后几行）
    - base64 图片：直接提示，避免截断破坏编码
    """
    text = text.strip()
    if len(text) <= max_len:
        return text

    # 检测 base64 图片输出
    # if _is_base64_image(text):
    #     preview = text[:200]
    #     return f"{preview}\n... [图片/base64 数据，共 {len(text)} 字符，已截断] ..."

    # 报错信息优先看尾部
    if "Error" in text or "Traceback" in text or "exception" in text.lower():
        head = min(100, max_len // 4)
        tail = max_len - head - 50
    else:
        head = min(200, max_len // 3)
        tail = max_len - head - 50

    tail = max(tail, 50)
    head = max(head, 30)

    truncated = (
        f"{text[:head]}\n"
        f"... [截断：原输出 {len(text)} 字符] ...\n"
        f"{text[-tail:]}"
    )
    return truncated


def _get_allowed_dirs() -> List[str]:
    """返回当前允许读写的目录列表。"""
    dirs = [WORK_DIR]
    if _TEMP_DIR is not None:
        dirs.append(_TEMP_DIR)
    return dirs


def _is_path_allowed(path: str) -> bool:
    """检查文件路径是否在允许的目录内"""
    try:
        abs_path = os.path.abspath(path)
        for d in _get_allowed_dirs():
            if abs_path.startswith(d + os.sep) or abs_path == d:
                return True
        return False
    except Exception:
        return False


def _check_ast_safety(code: str) -> tuple[bool, str]:
    """
    用 AST 检查代码安全性。
    返回：(是否安全, 错误信息)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        # 检查 import
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split('.')[0]
                if top_module not in IMPORT_WHITELIST and alias.name not in IMPORT_WHITELIST:
                    return False, f"Forbidden import: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            top_module = node.module.split('.')[0] if node.module else ''
            if top_module not in IMPORT_WHITELIST and node.module not in IMPORT_WHITELIST:
                return False, f"Forbidden import from: {node.module}"

        # 检查函数调用
        elif isinstance(node, ast.Call):
            func_name = _get_call_name(node.func)
            if func_name in BANNED_CALLS:
                return False, f"Forbidden function call: {func_name}"
            
            # 检查写入函数（plt.savefig, cv2.imwrite 等）
            if func_name in WRITE_CALLS:
                if node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        safe, msg = _is_temp_dir_only(first_arg.value)
                        if not safe:
                            return False, f"{func_name} forbidden outside temp dir: {msg}"
            
            # 检查 .save() 方法调用（如 img.save(path)）
            elif func_name.endswith('.save'):
                if node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        safe, msg = _is_temp_dir_only(first_arg.value)
                        if not safe:
                            return False, f".save() forbidden outside temp dir: {msg}"
            
            # 特别检查 open() 调用的路径
            if func_name == 'open':
                safe, msg = _check_open_call(node)
                if not safe:
                    return False, msg

        # 禁止 __import__
        elif isinstance(node, ast.Name) and node.id == '__import__':
            return False, "Forbidden: __import__"

    return True, ""


def _get_call_name(node) -> str:
    """从 AST Call 节点提取函数全名，如 'os.system'"""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))


def _is_write_mode(mode: str) -> bool:
    """判断文件打开模式是否为写入模式。"""
    return any(c in mode for c in 'wax+')


def _is_temp_dir_only(path: str) -> tuple[bool, str]:
    """
    检查路径是否只在临时目录内（不在 WORK_DIR 内）。
    用于写入操作：只允许写入临时目录，禁止写入原始图片目录。
    """
    if _TEMP_DIR is None:
        return False, "No temp dir set, write operations are forbidden"
    try:
        abs_path = os.path.abspath(path)
        # 必须在临时目录下
        in_temp = abs_path.startswith(_TEMP_DIR + os.sep) or abs_path == _TEMP_DIR
        if not in_temp:
            return False, f"Write target must be inside temp dir ({_TEMP_DIR}), got: {path}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_open_call(node: ast.Call) -> tuple[bool, str]:
    """检查 open() 的路径和模式。写模式只允许在临时目录内。"""
    if not node.args:
        return True, ""  # open() 无参数，暂不处理
    
    first_arg = node.args[0]
    path = None
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        path = first_arg.value
        if not _is_path_allowed(path):
            return False, f"open() path not allowed: {path}"
    
    # 检查 mode 参数
    mode = 'r'  # 默认读模式
    if len(node.args) >= 2:
        mode_arg = node.args[1]
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
            mode = mode_arg.value
    # 也检查关键字参数 mode=...
    for kw in node.keywords:
        if kw.arg == 'mode' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            mode = kw.value.value
    
    # 如果是写模式，必须只在临时目录内
    if path is not None and _is_write_mode(mode):
        safe, msg = _is_temp_dir_only(path)
        if not safe:
            return False, f"open() write forbidden outside temp dir: {msg}"
    
    return True, ""


# ==================== 公共 API ====================

def extract_code(response: str) -> Optional[str]:
    """
    从模型 response 中提取 <code>...</code> 块里的 Python 代码。
    支持嵌套 ```python ... ``` 格式。
    """
    pattern = r'<code>\s*(?:```python\s*\n)?(.*?)\s*(?:```\s*)?</code>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        inner = re.search(r'```python\s*\n(.*?)```', code, re.DOTALL)
        if inner:
            code = inner.group(1).strip()
        return code if code else None
    return None


def has_answer(response: str) -> bool:
    """检查 response 中是否包含答案标记。
    兼容完整 <answer>...</answer>、未闭合 <answer>、以及 \boxed{} 格式。"""
    if re.search(r'<answer>.*?</answer>', response, re.DOTALL):
        return True
    if '<answer>' in response:
        return True
    if r'\boxed{' in response:
        return True
    return False


def execute_code(code: str, timeout: int = 30) -> dict:
    """
    安全执行 Python 代码。
    流程：AST 安全检查 → 注入 sandbox_temp_dir → 写入临时文件 → subprocess 执行。
    """
    # 1. AST 安全检查
    safe, msg = _check_ast_safety(code)
    if not safe:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'[Security Check Failed] {msg}',
            'exit_code': -1,
            'timeout': False,
        }

    # 2. 注入 sandbox_temp_dir（若已设置临时目录）
    if _TEMP_DIR is not None:
        os.makedirs(_TEMP_DIR, exist_ok=True)
        code_prefix = f"sandbox_temp_dir = {_TEMP_DIR!r}\n"
        code_with_context = code_prefix + code
    else:
        code_with_context = code

    # 3. 写入临时文件
    os.makedirs(WORK_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, encoding='utf-8', dir=WORK_DIR
    ) as f:
        f.write(code_with_context)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],  # 用绝对路径，因为 cwd 已改为项目根目录
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.abspath('.'),  # 在项目根目录执行，使相对路径（如 ./Part2/data/images/xxx.jpg）正确解析
        )
        # ===== 关键：截断前先提取图片路径，防止路径被截断丢失 =====
        stdout_full = result.stdout
        stderr_full = result.stderr
        # 从完整 stdout 中提取图片路径和图片（不受后续截断影响）
        cleaned_full, extracted_images = extract_path_images(stdout_full)

        # 对清理后的文本做首尾截断（保留错误栈尾部）
        stdout = _raw_truncate(cleaned_full, EXEC_TRUNCATE)
        stderr = _raw_truncate(stderr_full, EXEC_TRUNCATE)

        return {
            'success': result.returncode == 0,
            'stdout': stdout,
            'stderr': stderr,
            'exit_code': result.returncode,
            'timeout': False,
            'extracted_images': extracted_images,  # 预提取的图片
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Execution timed out after {timeout} seconds',
            'exit_code': -1,
            'timeout': True,
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'exit_code': -1,
            'timeout': False,
        }
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def extract_path_images(stdout: str) -> Tuple[str, List[Image.Image]]:
    """
    从 stdout 中提取图片文件路径，加载为 PIL.Image。
    只识别以常见图片后缀结尾、且位于允许目录内的路径。

    Returns:
        (清理后的文本, 图片列表)
    """
    images = []
    cleaned_lines = []
    allowed_dirs = _get_allowed_dirs()
    img_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')

    for line in stdout.strip().split('\n'):
        line_stripped = line.strip()
        # 检查是否是图片文件路径
        if line_stripped.lower().endswith(img_exts):
            abs_path = os.path.abspath(line_stripped)
            is_allowed = any(
                abs_path.startswith(d + os.sep) or abs_path == d
                for d in allowed_dirs
            )
            if is_allowed and os.path.isfile(abs_path):
                try:
                    img = Image.open(abs_path).convert('RGB')
                    images.append(img)
                    cleaned_lines.append(f"[Image output: {line_stripped}]")
                    continue
                except Exception:
                    pass
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines), images


def wrap_sandbox_output(result: dict) -> Dict:
    """
    把执行结果包装成 <sandbox_output> 标签（带智能截断）。
    使用 execute_code 预提取的图片（不受截断影响）。
    模型必须学会保存图片到文件并打印路径，不再支持 base64 输出。

    Returns:
        {
            "text": str,           # 包装后的文本，适合放入对话历史
            "images": List[PIL.Image],  # 提取出的图片（如有）
        }
    """
    # 使用 execute_code 从完整 stdout 预提取的图片（不受截断影响）
    images = result.get('extracted_images', [])

    if result['success']:
        stdout = result['stdout'].strip()
        if not stdout:
            text = "(No output)"
        else:
            text = _smart_truncate(stdout)
        # 如果代码执行成功但没有提取到有效图片，给模型明确反馈
        if not images:
            text += (
                "\n[Note: No valid image path was detected in the output. "
                "Please save the processed image to sandbox_temp_dir and print its full file path.]"
            )
    else:
        err = result['stderr'].strip()
        text = _smart_truncate(err)
        text = f"Error (exit code {result['exit_code']}):\n{text}"

    wrapped = f"<sandbox_output>\n{text}\n</sandbox_output>"
    return {"text": wrapped, "images": images}


def run_rl_loop(
    generate_fn,
    question: str,
    image_path: Optional[str] = None,
    max_rounds: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    多轮 RL 交互循环。

    Args:
        generate_fn: 函数，接收 messages 列表，返回模型生成的字符串
        question: 用户问题
        image_path: 图片路径（可选）
        max_rounds: 最大轮数，防止无限循环
        verbose: 是否打印中间过程

    Returns:
        {
            'final_answer': str,          # 最终 <answer> 内容
            'full_history': List[str],    # 每轮模型输出
            'rounds': int,                # 实际轮数
            'code_executions': int,       # 代码执行次数
        }
    """
    messages = []

    # 构造第一轮 user message
    content = []
    if image_path:
        # 按 Qwen-VL 格式在消息中插入图片标记；实际图片加载由外部 generate_fn 处理
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})

    history = []
    code_executions = 0

    for round_idx in range(max_rounds):
        if verbose:
            print(f"\n{'='*40}")
            print(f"[Round {round_idx + 1}] Model generating...")

        response = generate_fn(messages)
        history.append(response)

        if verbose:
            print(f">>> Model:\n{response[:800]}")
            if len(response) > 800:
                print("...")

        # 如果已经给出 answer，结束循环
        if has_answer(response):
            if verbose:
                print(f"[Round {round_idx + 1}] <answer> detected, stopping.")
            break

        # 提取并执行代码
        code = extract_code(response)
        if code:
            if verbose:
                print(f"[Round {round_idx + 1}] Extracted code, executing...")
            result = execute_code(code, timeout=30)
            code_executions += 1

            if verbose:
                print(f"    success={result['success']}, exit={result['exit_code']}")
                if result['stdout']:
                    print(f"    stdout: {result['stdout'][:200]}")
                if result['stderr']:
                    print(f"    stderr: {result['stderr'][:200]}")

            sandbox_result = wrap_sandbox_output(result)

            # 把本轮对话加入历史，继续下一轮
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            # sandbox 输出支持图片：若有 base64 图片，解码后嵌入 content
            sandbox_content = [{"type": "text", "text": sandbox_result["text"]}]
            for img in sandbox_result.get("images", []):
                sandbox_content.insert(0, {"type": "image", "image": img})
            messages.append({"role": "user", "content": sandbox_content})
        else:
            # 没有代码也没有 answer，可能是中间推理，继续让模型生成
            if verbose:
                print(f"[Round {round_idx + 1}] No code found, continuing...")
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            # 加一个提示让模型继续
            messages.append({"role": "user", "content": [{"type": "text", "text": "Please continue your reasoning."}]})

    # 提取最终 answer
    final_answer = ""
    for resp in reversed(history):
        match = re.search(r'<answer>(.*?)</answer>', resp, re.DOTALL)
        if match:
            final_answer = match.group(1).strip()
            break

    return {
        'final_answer': final_answer,
        'full_history': history,
        'rounds': len(history),
        'code_executions': code_executions,
    }



