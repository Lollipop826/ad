"""
火山引擎 BigASR 大模型流式语音识别（替换 SenseVoice 本地模型）

协议文档: wss://openspeech.bytedance.com/api/v3/sauc/bigmodel
注意：BigASR 协议与 TTS 完全不同，使用 sequence number 而非 event number。

环境变量：
  VOLC_APP_ID        - APP ID（控制台数字ID）
  VOLC_ACCESS_TOKEN  - Access Token（与 TTS 共用同一套凭证）

接口与 SenseVoice 返回格式保持一致：
  {"text": str, "emotion": "neutral", "language": "zh", "event": "Speech"}
"""
import asyncio
import gzip
import json
import os
import struct
import uuid
import time
import numpy as np
import websockets

_WSS_URL     = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
_RESOURCE_ID = "volc.bigasr.sauc.duration"

# ── Binary protocol ───────────────────────────────────────────────────────────
# Byte 1: (msg_type << 4) | flags
#   msg_type: 0b0001=FullClientReq, 0b0010=AudioOnlyReq, 0b1001=FullServerResp, 0b1111=Error
#   flags:    0b0000=no-seq, 0b0001=pos-seq, 0b0010=last-no-seq, 0b0011=last-with-seq
# Byte 2: (serialization << 4) | compression
#   serialization: 0b0000=raw, 0b0001=JSON
#   compression:   0b0000=none, 0b0001=gzip


def _full_client_request(payload_json: bytes) -> bytes:
    """配置帧：Full client request, flags=0b0000(no seq), JSON, no-compress"""
    hdr = bytes([0x11, 0x10, 0x10, 0x00])
    return hdr + struct.pack(">I", len(payload_json)) + payload_json


def _audio_request(audio: bytes, last: bool = False) -> bytes:
    """音频帧：Audio-only request, raw, no-compress"""
    flags = 0x02 if last else 0x00   # 0b0010=last, 0b0000=normal
    hdr = bytes([0x11, 0x20 | flags, 0x00, 0x00])
    return hdr + struct.pack(">I", len(audio)) + audio


def _parse_server_response(data: bytes) -> dict:
    """解析服务端响应帧"""
    msg_type = (data[1] >> 4) & 0x0F
    flags    = data[1] & 0x0F
    serializ = (data[2] >> 4) & 0x0F
    compress = data[2] & 0x0F

    # Error frame: [hdr][4B error_code][4B msg_len][msg]
    if msg_type == 0x0F:
        error_code = struct.unpack(">I", data[4:8])[0]
        msg_len    = struct.unpack(">I", data[8:12])[0]
        msg = data[12:12+msg_len].decode("utf-8", errors="replace")
        return {"is_last": True, "payload": None, "error": f"code={error_code}: {msg}"}

    pos = 4
    # flags bit 0: has sequence number
    seq = None
    if flags & 0x01:
        seq = struct.unpack(">i", data[pos:pos+4])[0]
        pos += 4

    # flags bit 1: is last packet
    is_last = bool(flags & 0x02) or (seq is not None and seq < 0)

    payload_size = struct.unpack(">I", data[pos:pos+4])[0]
    pos += 4
    raw = data[pos:pos+payload_size]

    if compress == 0x01 and raw:
        raw = gzip.decompress(raw)

    payload = None
    if serializ == 0x01 and raw:
        try:
            payload = json.loads(raw)
        except Exception:
            pass

    return {"is_last": is_last, "payload": payload, "error": None, "seq": seq}


def _to_pcm_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


async def ark_asr_recognize(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> dict:
    """
    调用火山引擎 BigASR (WebSocket) 识别一段完整音频。

    Returns:
        {"text": str, "emotion": "neutral", "language": "zh", "event": "Speech"}
    """
    app_id = os.getenv("VOLC_APP_ID", "")
    token  = os.getenv("VOLC_ACCESS_TOKEN", "")
    if not app_id or not token:
        raise ValueError("未配置 VOLC_APP_ID / VOLC_ACCESS_TOKEN")

    t0 = time.time()
    pcm = _to_pcm_bytes(audio)

    ws_headers = {
        "X-Api-App-Key":     app_id,
        "X-Api-Access-Key":  token,
        "X-Api-Resource-Id": _RESOURCE_ID,
        "X-Api-Connect-Id":  str(uuid.uuid4()),
    }

    config = {
        "user": {"uid": "mmse_screening"},
        "audio": {
            "format": "pcm",
            "rate":   sample_rate,
            "bits":   16,
            "channel": 1,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_punc": True,
        },
    }

    text = ""
    chunk_bytes = sample_rate // 5 * 2   # 200ms per chunk (推荐值)
    chunks = [pcm[i:i+chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
    total_frames = 1 + len(chunks)   # 1 config frame + N audio frames

    async with websockets.connect(_WSS_URL, additional_headers=ws_headers, open_timeout=10) as ws:

        async def _send_all():
            await ws.send(_full_client_request(json.dumps(config, ensure_ascii=False).encode()))
            for i, chunk in enumerate(chunks):
                await ws.send(_audio_request(chunk, last=(i == len(chunks) - 1)))

        async def _recv_all():
            nonlocal text
            for _ in range(total_frames):
                data = await asyncio.wait_for(ws.recv(), timeout=15)
                resp = _parse_server_response(data)
                if resp["error"]:
                    raise RuntimeError(f"[ArkASR] 错误: {resp['error']}")
                if resp["payload"]:
                    t = (resp["payload"].get("result") or {}).get("text", "")
                    if t:
                        text = t

        await asyncio.gather(_send_all(), _recv_all())

    elapsed = time.time() - t0
    print(f"[ArkASR] ✅ '{text}' ({elapsed:.2f}s)")
    return {"text": text, "emotion": "neutral", "language": "zh", "event": "Speech"}
