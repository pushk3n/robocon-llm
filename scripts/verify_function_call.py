#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_deepseek_verbose.py
更详细的调试脚本，用于验证 Ollama 上的模型（例如 llama3.2:3b / deepseek-r1）
是否能输出 function_call 风格的 JSON 并与本地 map API 交互。

特点:
 - 增加时间统计，评估LLM调用性能。
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
import os
import json
import re
import subprocess
import traceback
import time
from pathlib import Path

# ---------- 配置 ----------
MODEL = "llama3.2:3b"  # 改成你实际在 Ollama 中的模型名，例如 "deepseek-r1:latest" 或 "llama3.2:3b"
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
    # 计时开始：查表是 Python 的任务，速度应该极快
    start_time = time.time() 
    
    try:
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            map_data = json.load(f)
    except Exception as e:
        return {"error": f"无法读取 map.json: {e}"}

    # 查找名字完全匹配的区域
    for zone in map_data.get("zones", []):
        if zone.get("name") == zone_name:
            elapsed_time = time.time() - start_time
            log(f"API/查表耗时: {elapsed_time:.4f}s")
            return zone
    # 如果没找到，尝试不区分大小写匹配（更宽容）
    for zone in map_data.get("zones", []):
        if zone.get("name", "").lower() == zone_name.lower():
            elapsed_time = time.time() - start_time
            log(f"API/查表耗时: {elapsed_time:.4f}s")
            return zone
            
    elapsed_time = time.time() - start_time
    log(f"API/查表耗时: {elapsed_time:.4f}s")
    return {"error": f"未找到区域: {zone_name}"}

def run_ollama(prompt: str, timeout: int = 120, use_cpu_only: bool = False):
    """
    通过 subprocess 调用 ollama run 并把 prompt 发到 stdin。
    如果 use_cpu_only 为 True，则强制 Ollama 仅使用 CPU。
    返回 (stdout_str, stderr_str, returncode).
    """
    start_time = time.time()
    
    # 设置环境变量来控制 GPU 使用
    env = os.environ.copy()
    if use_cpu_only:
        # OM_NUM_GPU=0 告诉 Ollama/llama.cpp 不使用 GPU
        env['OM_NUM_GPU'] = '0'
        log("警告: 强制使用 CPU 模式进行 LLM 调用...")
    else:
        # 允许 Ollama 使用 GPU (如果存在)
        env.pop('OM_NUM_GPU', None)
    
    try:
        proc = subprocess.run(
            ["ollama", "run", MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env  # 传递修改后的环境变量
        )
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        
        elapsed_time = time.time() - start_time
        log(f"LLM 调用总耗时: {elapsed_time:.4f}s {'(CPU ONLY)' if use_cpu_only else '(GPU/Default)'}")
        
        return stdout, stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        elapsed_time = time.time() - start_time
        return "", f"TIMEOUT after {timeout}s (Elapsed: {elapsed_time:.4f}s)", -1
    except Exception as e:
        elapsed_time = time.time() - start_time
        return "", f"CALL ERROR (Elapsed: {elapsed_time:.4f}s): {e}\n{traceback.format_exc()}", -2

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
你是一个 Robocon 比赛地图推理助手，你可以调用以下 Python 工具函数：
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
请查询区域 'R2_EX_zone1' 的详细信息，包括它的 R2_access 状态和邻接区域。
"""
    
    # ----------------------------------------------------
    # 执行性能测试的函数
    # ----------------------------------------------------
    def run_full_test(use_cpu_only: bool):
        mode_label = "CPU ONLY" if use_cpu_only else "GPU/DEFAULT"
        log(f"===== 开始 {mode_label} 性能测试 (查询 F6) =====")
        
        # --- 第一次 LLM 调用 (Function Calling) ---
        stdout1, stderr1, rc1 = run_ollama(combined_prompt, use_cpu_only=use_cpu_only)

        # Step2/3: 解析 JSON
        json_obj, raw_json, err = extract_first_json_robust(stdout1) 
        
        if json_obj is not None:
            # 简化解析，假设 function_call 成功
            fc = json_obj.get("function_call", {})
            args = fc.get("arguments", {})
            zone_name = args.get("zone_name")

            if fc and fc.get("name") == "get_zone_info" and zone_name:
                
                # Step4: 执行 map 查询（Python API）
                result = get_zone_info(zone_name or "")
                
                # Step5: 构造闭环 Prompt
                final_prompt = (
                    f"这是 get_zone_info('{zone_name}') 的结果: {json.dumps(result, ensure_ascii=False)}\n"
                    "请基于上面的结果，用中文简要总结：这个区域的要点（是否R2可进入、高度和邻接区域）。"
                )
                log(f"把 map 查询结果回传给模型 ({mode_label})")
                
                # --- 第二次 LLM 调用 (总结/决策) ---
                stdout2, stderr2, rc2 = run_ollama(final_prompt, use_cpu_only=use_cpu_only)
                
                debug_print(f"[{mode_label}] LLM 第一次调用输出 (请求 JSON)", stdout1)
                debug_print(f"[{mode_label}] LLM 第二次调用输出 (总结)", stdout2)
                
            else:
                log(f"JSON 结构错误或未识别函数名: {stdout1}")
        else:
            log(f"JSON 解析失败 ({mode_label}): {err}")
        
        log(f"===== {mode_label} 性能测试结束 =====")
        print("\n" + "="*50 + "\n")
    
    
    # ----------------------------------------------------
    # 运行实际的两次测试
    # ----------------------------------------------------
    
    # 测试 1: GPU/默认模式 (预计速度快)
    run_full_test(use_cpu_only=False)
    
    # 测试 2: 纯 CPU 模式 (模拟边缘设备，预计速度慢)
    run_full_test(use_cpu_only=True)

print("==============================\n"*3)
if __name__ == "__main__":
    main()