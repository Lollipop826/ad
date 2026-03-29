"""
视觉评估工具 - 使用视觉大模型评估患者动作/图形

支持视频和图片两种输入模式：
- 视频模式（推荐）：录制 5-10 秒短视频，VL 模型逐帧分析动作
- 图片模式（降级）：单帧截图分析

支持的 MMSE 视觉任务：
1. language_reading_close_eyes - 判断患者是否闭眼
2. language_3step_action - 判断患者是否执行三步动作
3. copy_pentagons - 判断患者临摹的五边形是否正确
"""

import os
import json
import re
import time
import base64
from typing import Optional, Dict, Any

from src.llm.http_client_pool import get_shared_httpx_client


# 各视觉任务的评估 prompt（视频模式）
VISION_TASK_PROMPTS: Dict[str, Dict[str, str]] = {
    "language_reading_close_eyes": {
        "system": "你是一位神经内科医生的助手，正在通过摄像头视频辅助评估老年患者的认知能力。",
        "prompt": """请观察这段视频中的人物，判断他/她是否**闭上了眼睛**。

注意观察整个视频过程中眼部的变化：
- 视频中有明确的闭眼动作（从睁眼到闭眼） → is_correct: true, quality: "excellent"
- 视频中大部分时间眼睛闭合或半闭 → is_correct: true, quality: "good"
- 视频中眼睛始终睁开，没有闭眼动作 → is_correct: false, quality: "poor"
- 看不清人脸/无人 → is_correct: null, quality: "unknown"

只输出JSON：{"is_correct": true/false/null, "quality": "excellent/good/poor/unknown", "detail": "简短描述"}""",
    },
    "language_3step_action": {
        "system": "你是一位神经内科医生的助手，正在通过摄像头视频辅助评估老年患者的认知能力。",
        "prompt": """请观察这段视频，判断患者是否在胸口摆出了"5"的手势（五指张开）。

重点观察：
- 手是否放在胸口附近
- 五指是否张开呈"5"的手势

评分标准：
- 在胸口正确摆出5的手势 → is_correct: true, quality: "excellent"
- 手势大致正确但位置不太准 → is_correct: true, quality: "good"
- 没有做出手势或手势明显错误 → is_correct: false, quality: "poor"
- 看不清/无人 → is_correct: null, quality: "unknown"

只输出JSON：{"is_correct": true/false/null, "quality": "excellent/good/fair/poor/unknown", "detail": "简短描述"}""",
    },
    "copy_pentagons": {
        "system": "你是一位神经内科医生的助手，正在通过视频评估患者临摹的五边形图形。",
        "prompt": """请观察这段视频中患者画的图形，判断是否正确临摹了**两个相交的五边形**。

MMSE 评分标准：
- 两个五边形都是五条边，且有一个交叠区域 → is_correct: true, quality: "excellent"（1分）
- 基本能看出两个五边形且有交叠，但形状不够标准 → is_correct: true, quality: "good"（1分）
- 只画了一个图形，或两个图形没有交叠 → is_correct: false, quality: "fair"（0分）
- 完全无法辨认 → is_correct: false, quality: "poor"（0分）
- 看不到画作 → is_correct: null, quality: "unknown"

只输出JSON：{"is_correct": true/false/null, "quality": "excellent/good/fair/poor/unknown", "detail": "简短描述"}""",
    },
}


def _messages_to_ark_input(messages: list) -> list:
    """
    将 OpenAI chat messages 格式转换为 Volcengine Responses API 的 input 格式。
    system role 合并进第一条 user 消息的文本前缀。
    """
    system_text = ""
    ark_input = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_text = content if isinstance(content, str) else ""
            continue
        # user / assistant
        if isinstance(content, str):
            parts = []
            if system_text:
                parts.append({"type": "input_text", "text": system_text})
                system_text = ""
            parts.append({"type": "input_text", "text": content})
            ark_input.append({"role": role, "content": parts})
        elif isinstance(content, list):
            parts = []
            if system_text:
                parts.append({"type": "input_text", "text": system_text})
                system_text = ""
            for item in content:
                t = item.get("type", "")
                if t in ("text",):
                    parts.append({"type": "input_text", "text": item["text"]})
                elif t == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    parts.append({"type": "input_image", "image_url": url})
                elif t == "video_url":
                    # Volcengine Responses API 不支持 video_url，跳过（由调用方降级处理）
                    raise ValueError("Volcengine Responses API 不支持 video_url，请使用 SiliconFlow")
            ark_input.append({"role": role, "content": parts})
    return ark_input


def _call_vlm_ark(messages: list, timeout: float = 60.0) -> str:
    """
    调用 Volcengine ARK Responses API（/api/v3/responses）
    """
    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    model = os.getenv("VISION_EVAL_MODEL")  # 必须是 ep-xxx 视觉接入点
    if not api_key:
        raise ValueError("未配置 ARK_API_KEY")
    if not model:
        raise ValueError("未配置 VISION_EVAL_MODEL (需要 ep-xxx 视觉接入点)")

    ark_input = _messages_to_ark_input(messages)
    payload = {
        "model": model,
        "input": ark_input,
        "max_output_tokens": 600,
        "temperature": 0.05,
    }

    client = get_shared_httpx_client()
    resp = client.post(
        f"{base_url}/responses",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        print(f"[VisionEval][ARK] ⚠️ API 返回 {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    data = resp.json()
    # Responses API: output[].content[].text
    # 先尝试提取完整或部分输出（incomplete_details=length 时仍可能有内容）
    for item in data.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and part.get("text", "").strip():
                return part["text"].strip()
    # 输出被截断（reason=length）且无有效内容 → 告知上层回退
    incomplete = data.get("incomplete_details", {})
    if incomplete.get("reason") == "length":
        raise ValueError(f"ARK 输出被截断(max_output_tokens不足)，回退SiliconFlow")
    raise ValueError(f"ARK Responses API 返回格式异常: {str(data)[:300]}")


def _call_vlm_siliconflow(messages: list, timeout: float = 60.0) -> str:
    """
    调用 SiliconFlow VLM API（/v1/chat/completions）
    """
    api_key = os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    model = os.getenv("SILICONFLOW_VISION_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")

    if not api_key:
        raise ValueError("未配置 SILICONFLOW_API_KEY")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.05,
    }

    client = get_shared_httpx_client()
    resp = client.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        print(f"[VisionEval][SF] ⚠️ API 返回 {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_vlm(messages: list, timeout: float = 60.0, force_siliconflow: bool = False) -> str:
    """
    统一 VLM 调用入口：优先 Volcengine ARK，回退 SiliconFlow。
    force_siliconflow=True 时跳过 ARK（如 video_url 不被 ARK 支持时）。
    """
    vision_model = os.getenv("VISION_EVAL_MODEL", "")
    use_ark = bool(os.getenv("ARK_API_KEY")) and vision_model.startswith("ep-") and not force_siliconflow

    if use_ark:
        try:
            result = _call_vlm_ark(messages, timeout=timeout)
            print(f"[VisionEval] ✅ ARK Responses API 调用成功")
            return result
        except ValueError as e:
            # video_url 不支持等明确错误 → 回退
            print(f"[VisionEval] ⚠️ ARK 不支持，回退 SiliconFlow: {e}")
        except Exception as e:
            print(f"[VisionEval] ⚠️ ARK 调用失败，回退 SiliconFlow: {e}")

    return _call_vlm_siliconflow(messages, timeout=timeout)


def _parse_vlm_json(content: str) -> Dict[str, Any]:
    """从 VLM 返回中提取 JSON"""
    if "```" in content:
        match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        return json.loads(json_match.group())
    return {"is_correct": None, "quality": "unknown", "detail": content}


def _build_result(result: Dict, task_id: str, elapsed_ms: float, source: str) -> Dict[str, Any]:
    """标准化输出格式"""
    is_correct = result.get("is_correct")
    quality = result.get("quality", result.get("quality_level", "unknown"))
    detail = result.get("detail", "")
    quality_to_cognitive = {
        "excellent": "正常", "good": "正常",
        "fair": "轻度异常", "poor": "异常", "unknown": "无法判断",
    }
    return {
        "success": True,
        "is_correct": is_correct,
        "quality_level": quality,
        "cognitive_performance": quality_to_cognitive.get(quality, "无法判断"),
        "is_complete": True,
        "evaluation_detail": f"{source}评估: {detail}",
        "need_followup": False,
        "confidence": 0.85 if is_correct is not None else 0.0,
        "source": source,
        "steps_completed": result.get("steps_completed"),
        "elapsed_ms": elapsed_ms,
    }


def evaluate_video_with_vlm(
    video_base64: str,
    task_id: str,
    mime_type: str = "video/webm",
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    使用 SiliconFlow video_url 格式评估视频（最准确）

    Args:
        video_base64: base64 编码的视频（可含 data:video/... 前缀）
        task_id: MMSE 任务 ID
        mime_type: 视频 MIME 类型（浏览器录的一般是 video/webm）
        extra_context: 额外上下文信息
    """
    start_time = time.time()

    task_config = VISION_TASK_PROMPTS.get(task_id)
    if not task_config:
        return {"success": False, "error": f"不支持的视觉任务: {task_id}",
                "is_correct": None, "quality_level": "unknown"}

    raw_b64 = video_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    user_prompt = task_config["prompt"]
    if extra_context:
        user_prompt += f"\n\n补充信息：{extra_context}"

    data_uri = f"data:{mime_type};base64,{raw_b64}"
    messages = [
        {"role": "system", "content": task_config["system"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": data_uri,
                        "detail": "high",
                        "max_frames": 16,
                        "fps": 2,
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    content = _call_vlm(messages, timeout=90.0, force_siliconflow=True)
    print(f"[VisionEval] 🎬 VLM 视频返回: {content[:200]}")
    result = _parse_vlm_json(content)
    elapsed = (time.time() - start_time) * 1000
    print(f"[VisionEval] ✅ 视频评估完成: task={task_id}, result={result} ({elapsed:.0f}ms)")
    return _build_result(result, task_id, elapsed, "视频")


def evaluate_frames_with_vlm(
    frames_base64: list,
    task_id: str,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    多帧评估（降级方案）：将多张摄像头截图作为时序图片序列发送给 VL 模型
    """
    start_time = time.time()

    task_config = VISION_TASK_PROMPTS.get(task_id)
    if not task_config:
        return {"success": False, "error": f"不支持的视觉任务: {task_id}",
                "is_correct": None, "quality_level": "unknown"}

    if not frames_base64 or len(frames_base64) == 0:
        return {"success": False, "error": "未收到任何帧数据",
                "is_correct": None, "quality_level": "unknown"}

    n_frames = len(frames_base64)
    print(f"[VisionEval] 🎞️ 多帧降级: {n_frames} 帧, task={task_id}")

    user_prompt = task_config["prompt"]
    if extra_context:
        user_prompt += f"\n\n补充信息：{extra_context}"

    content_parts = []
    for i, frame_b64 in enumerate(frames_base64):
        raw = frame_b64
        if "," in raw:
            raw = raw.split(",", 1)[1]
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{raw}",
                "detail": "low",
            },
        })

    time_hint = f"以上是按时间顺序拍摄的 {n_frames} 张连续截图，请综合所有图片判断患者的动作变化。\n\n"
    content_parts.append({"type": "text", "text": time_hint + user_prompt})

    messages = [
        {"role": "system", "content": task_config["system"]},
        {"role": "user", "content": content_parts},
    ]

    content = _call_vlm(messages, timeout=60.0)
    print(f"[VisionEval] 🎞️ VLM 多帧返回: {content[:200]}")
    result = _parse_vlm_json(content)
    elapsed = (time.time() - start_time) * 1000
    print(f"[VisionEval] ✅ 多帧评估完成: task={task_id}, {n_frames}帧, result={result} ({elapsed:.0f}ms)")
    return _build_result(result, task_id, elapsed, f"多帧({n_frames}帧)")


def evaluate_hybrid(
    task_id: str,
    video_base64: str = "",
    mime_type: str = "video/webm",
    frames_base64: list = None,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    混合评估：优先用 video_url（最准确），失败则自动降级到多帧

    Args:
        task_id: MMSE 任务 ID
        video_base64: 视频 base64（可选）
        mime_type: 视频 MIME 类型
        frames_base64: 多帧截图列表（可选，作为降级）
        extra_context: 额外上下文
    """
    # 策略1: 优先视频
    video_error = None
    if video_base64:
        try:
            print(f"[VisionEval] 🎬 尝试视频模式 (video_url)...")
            return evaluate_video_with_vlm(video_base64, task_id, mime_type, extra_context)
        except Exception as e:
            video_error = str(e)
            print(f"[VisionEval] ⚠️ 视频模式失败，降级到多帧: {e}")

    # 策略2: 多帧降级
    if frames_base64 and len(frames_base64) > 0:
        try:
            result = evaluate_frames_with_vlm(frames_base64, task_id, extra_context)
            if video_error:
                result["video_fallback_reason"] = video_error
            return result
        except Exception as e:
            print(f"[VisionEval] ❌ 多帧模式也失败: {e}")
            return {
                "success": False, "error": str(e),
                "is_correct": None, "quality_level": "unknown",
                "cognitive_performance": "无法判断", "is_complete": False,
                "evaluation_detail": f"视频和多帧评估均失败: {e}",
                "confidence": 0.0, "source": "hybrid",
            }

    return {"success": False, "error": "未收到视频或帧数据",
            "is_correct": None, "quality_level": "unknown"}


def evaluate_image_with_vlm(
    image_base64: str,
    task_id: str,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    调用 SiliconFlow 视觉大模型评估图片（降级模式）
    """
    start_time = time.time()

    task_config = VISION_TASK_PROMPTS.get(task_id)
    if not task_config:
        return {"success": False, "error": f"不支持的视觉任务: {task_id}",
                "is_correct": None, "quality_level": "unknown"}

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    user_prompt = task_config["prompt"].replace("这段视频", "这张图片").replace("视频中", "图片中")
    if extra_context:
        user_prompt += f"\n\n补充信息：{extra_context}"

    messages = [
        {"role": "system", "content": task_config["system"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    try:
        content = _call_vlm(messages, timeout=30.0)
        print(f"[VisionEval] 🔍 VLM 图片返回: {content[:200]}")
        result = _parse_vlm_json(content)
        elapsed = (time.time() - start_time) * 1000
        print(f"[VisionEval] ✅ 图片评估完成: task={task_id}, result={result} ({elapsed:.0f}ms)")
        return _build_result(result, task_id, elapsed, "图片")

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        print(f"[VisionEval] ❌ 图片评估失败 ({elapsed:.0f}ms): {e}")
        return {
            "success": False, "error": str(e),
            "is_correct": None, "quality_level": "unknown",
            "cognitive_performance": "无法判断", "is_complete": False,
            "evaluation_detail": f"图片评估失败: {e}",
            "confidence": 0.0, "source": "vision_model",
        }
