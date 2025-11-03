#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_deepseek_verbose.py
更详细的调试脚本，用于验证 Ollama 上的模型（例如 llama3.2:3b / deepseek-r1）
是否能输出 function_call 风格的 JSON 并与本地 map API 交互。

特点:
 - 打印 stdout/stderr 的原始 repr（以便看隐藏字符）
 - 打印 subprocess returncode
 - 优先解析 JSON function_call（严格解析）
 - fallback: 从任意自然语言文本中提取 token 列表并打印全部候选
 - 在获取 map 查询结果后把结果回传给模型做最终的自然语言总结（闭环）
 - 每一步都输出足够的调试信息，方便定位问题

使用:
    cd ~/robocon-llm/repo/scripts
    python3 verify_deepseek_verbose.py
"""

import json
import re
import subprocess
import traceback
import time
from pathlib import Path

# ---------- 配置 ----------
MODEL = "llama3.2:1b"  # 改成你实际在 Ollama 中的模型名，例如 "deepseek-r1:latest" 或 "llama3.2:3b"
MAP_PATH = Path(__file__).resolve().parents[1] / "data" / "initial" / "map.json"

# ---------- 工具函数 ----------
def log(msg: str):
    """简单时间戳化日志"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"🚀[{ts}] {msg}")

def debug_print(title, content):
    """结构化打印，自动格式化 JSON / 列表"""
    print("\n\n" + "🧩" +"="*8 + f" {title} " + "="*8)
    if isinstance(content, (dict, list)):
        try:
            print(json.dumps(content, indent=2, ensure_ascii=False))
        except Exception:
            print(repr(content))
    else:
        print(content)

def get_zone_info(zone_name: str):
    """从 map.json 查表并返回 dict。保证 deterministic、100% 准确（取自 map.json）"""
    try:
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            map_data = json.load(f)
    except Exception as e:
        return {"error": f"无法读取 map.json: {e}"}

    # 查找名字完全匹配的区域
    for zone in map_data.get("zones", []):
        if zone.get("name") == zone_name:
            return zone
    # 如果没找到，尝试不区分大小写匹配（更宽容）
    for zone in map_data.get("zones", []):
        if zone.get("name", "").lower() == zone_name.lower():
            return zone
    return {"error": f"未找到区域: {zone_name}"}

def run_ollama(prompt: str, timeout: int = 120):
    """
    通过 subprocess 调用 ollama run 并把 prompt 发到 stdin。
    返回 (stdout_str, stderr_str, returncode).
    stdout/stderr 使用 'replace' 解码，保留尽可能多的信息。
    """
    try:
        proc = subprocess.run(
            ["ollama", "run", MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        return stdout, stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        return "", f"TIMEOUT after {timeout}s", -1
    except Exception as e:
        return "", f"CALL ERROR: {e}\n{traceback.format_exc()}", -2

def extract_first_json(text: str):
    """
    尝试从 text 中提取第一个 {...} JSON 片段并解析。
    返回 (obj_or_None, raw_json_text_or_None, error_or_None)
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, None, "no_json_fragment"
    raw = match.group(0)
    try:
        obj = json.loads(raw)
        return obj, raw, None
    except Exception as e:
        return None, raw, f"json_parse_error: {e}"

def extract_all_tokens(text: str):
    """
    从自然语言中抽取所有类似 token 的词（字母数字和下划线为一组），
    返回一个 candidate list（按出现顺序）。
    """
    tokens = re.findall(r"[A-Za-z0-9_]+", text)
    return tokens

# ----------------------------------------
# NEW/MODIFIED TOOL FUNCTION
# ----------------------------------------
def extract_first_json_robust(text: str):
    """
    尝试从 text 中提取第一个 {...} JSON 片段并解析。
    如果解析失败，尝试修复缺失的末尾大括号。
    返回 (obj_or_None, raw_json_text_or_None, error_or_None)
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, None, "no_json_fragment"
    
    raw = match.group(0).strip()
    
    # 1. 尝试原始解析
    try:
        obj = json.loads(raw)
        return obj, raw, None
    except Exception as e:
        original_error = str(e)

    # 2. 尝试 JSON 修复（处理模型截断尾部括号的问题）
    if original_error and ("Expecting" in original_error or "Unexpected" in original_error):
        # 统计原始 raw 中未闭合的括号数量
        open_braces = raw.count('{')
        close_braces = raw.count('}')
        missing_braces = open_braces - close_braces
        
        if missing_braces > 0:
            # 假设缺失的都是右大括号
            fixed_raw = raw + '}' * missing_braces
            
            try:
                fixed_obj = json.loads(fixed_raw)
                return fixed_obj, fixed_raw, f"repaired_missing_braces: {missing_braces}"
            except Exception:
                pass  # 修复失败，继续

    return None, raw, original_error # 返回原始失败信息

# ---------- 主逻辑 ----------
def main():
    log("开始验证脚本（verbose 模式）")

    # Step0: 检查 map.json 可读性（先行）
    try:
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            map_preview = json.load(f)
        debug_print("map.json 加载成功，zones 数量", len(map_preview.get("zones", [])))
    except Exception as e:
        debug_print("ERROR: 读取 map.json 失败", str(e))
        return

    # Step1: 构建合并 prompt（system + tool-spec + user question）
    combined_prompt = f"""
系统说明:
你是一个 Robocon 比赛地图推理助手，你可以调用以下工具函数：
[
  {{
    "name": "get_zone_info",
    "description": "查询 map.json 中指定区域的详细信息（坐标、邻接、R2_access 等）",
    "parameters": {{
      "type": "object",
      "properties": {{
        "zone_name": {{
          "type": "string",
          "description": "区域名，例如 MC_idle2, staff_rack, F6 等"
        }}
      }},
      "required": ["zone_name"]
    }}
  }}
]

行为规范:
- 当你需要查询某个区域时，请仅输出一个 JSON 片段 (单独一段文本)：
  {{"function_call": {{"name": "get_zone_info", "arguments": {{"zone_name": "区域名"}}}}}}
- 不要在 JSON 外写多余的解释或文本（在本次测试中我们希望严格解析 JSON）。
- 如果你无法确定区域名，请输出一个占位参数，例如 "unknown"。
- 你可以使用中文或英文，但是返回的 JSON 必须是合法的 JSON。

测试问题:
现在请你查询区域 'ramp' 的详细信息，包括它的可进入区域。
"""
    log("发送 prompt 到模型（一次性上下文）")
    stdout, stderr, rc = run_ollama(combined_prompt)

    # Step2: 打印原始响应（详尽）
    debug_print("模型 raw stdout (repr)", repr(stdout))
    debug_print("模型 stdout (raw)", stdout)
    if stderr and stderr.strip():
        debug_print("模型 stderr", stderr)
    debug_print("Subprocess returncode", rc)

    # Step3: 尝试严格解析 JSON（优先）
    # Step3: 尝试严格解析 JSON（优先）
    # **替换为新的函数**
    json_obj, raw_json, err = extract_first_json_robust(stdout) 
    
    if json_obj is not None:
        debug_print("找到 JSON 片段 raw", raw_json)
        # ... 如果修复成功，err 会包含 "repaired_missing_braces" ...
        if "repaired_missing_braces" in str(err):
            log(f"JSON 修复成功: {err}") 
        # ... 保持后续逻辑不变 ...

        # 检查 function_call 结构
        fc = json_obj.get("function_call")
        if not fc:
            debug_print("JSON 中未包含 function_call 键", json_obj.keys())
        else:
            func_name = fc.get("name")
            args = fc.get("arguments", {})
            debug_print("解析到的 function name", func_name)
            debug_print("解析到的 arguments", args)

            if func_name == "get_zone_info":
                zone_name = args.get("zone_name")
                debug_print("将执行 map 查询的 zone_name", zone_name)

                # Step4: 执行 map 查询
                result = get_zone_info(zone_name or "")
                debug_print(f"get_zone_info('{zone_name}') 返回", result)

                # Step5: 把结果回传给模型做最终总结（闭环）
                final_prompt = (
                    f"这是 get_zone_info('{zone_name}') 的结果: {json.dumps(result, ensure_ascii=False)}\n"
                    "请基于上面的结果，用中文简要总结：这个区域的要点（位置、是否R2可进入、相邻区域）。"
                )
                log("把 map 查询结果回传给模型，要求模型做自然语言总结")
                stdout2, stderr2, rc2 = run_ollama(final_prompt)
                debug_print("模型对 map 查询结果的总结 stdout (repr)", repr(stdout2))
                debug_print("模型对 map 查询结果的总结 stdout (raw)", stdout2)
                if stderr2 and stderr2.strip():
                    debug_print("模型对 map 查询结果的总结 stderr", stderr2)
                debug_print("Subprocess returncode (第二次调用)", rc2)
                return  # 只做一次完整闭环测试

    # Step6: 如果没能提取 JSON 或解析失败 -> fallback 分析
    debug_print("JSON 提取情况", err)
    log("进入 fallback 模式：打印模型输出、提取所有 candidate tokens 并尝试 fallback 查询")

    tokens = extract_all_tokens(stdout)
    debug_print("从模型输出提取到的 token 列表（candidate）", tokens)

    # 打印常见候选和上下文，便于人工判断
    if not tokens:
        debug_print("模型输出为空或无可提取 token", stdout)
        log("结束：未能从模型输出识别区域名")
        return

    # 尝试用第一个合理 token 做为区域名进行查询（非常宽容的 fallback）
    candidate = tokens[0]
    debug_print("选择第一个 candidate 作为区名尝试查询", candidate)
    result = get_zone_info(candidate)
    debug_print(f"get_zone_info('{candidate}') 返回", result)

    # 把 fallback 结果也发回模型，请求模型做解释/更正（便于观察模型如何反应）
    followup_prompt = (
        f"模型原始回复是: {stdout}\n\n"
        f"我基于你的回复尝试查询，并将 '{candidate}' 当作区域名查询，返回结果为: {json.dumps(result, ensure_ascii=False)}。\n"
        "如果我误解了你的意图，请直接用 JSON 格式的 function_call 返回正确的 zone_name："
        '{"function_call": {"name": "get_zone_info", "arguments": {"zone_name": "正确的区域名"}}}'
    )
    log("将 fallback 查询结果和原始模型回复一起回传，询问模型纠正（如果可能）")
    stdout3, stderr3, rc3 = run_ollama(followup_prompt)
    debug_print("模型对 fallback 情况的回复 stdout (repr)", repr(stdout3))
    debug_print("模型对 fallback 情况的回复 stdout (raw)", stdout3)
    if stderr3 and stderr3.strip():
        debug_print("模型对 fallback 情况的回复 stderr", stderr3)
    debug_print("Subprocess returncode (fallback call)", rc3)
    log("结束（verbose 验证）")

print("==============================\n"*3)
if __name__ == "__main__":
    main()
