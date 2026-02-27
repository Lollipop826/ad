"""
共享 HTTP 客户端池

优化 API 调用性能：
1. 单例模式共享客户端
2. 增大连接池大小
3. 启用 Keep-Alive
"""
import httpx
from typing import Optional, Any
from functools import lru_cache


# 全局共享的 httpx 客户端配置
_shared_async_client: Optional[httpx.AsyncClient] = None
_shared_sync_client: Optional[httpx.Client] = None


def get_shared_httpx_client() -> httpx.Client:
    """
    获取共享的同步 httpx 客户端
    
    - 连接池大小: 100 (默认10)
    - Keep-Alive 超时: 60s
    - 连接超时: 10s
    """
    global _shared_sync_client
    
    if _shared_sync_client is None:
        # 创建带大连接池的 httpx 客户端
        limits = httpx.Limits(
            max_keepalive_connections=100,  # Keep-Alive 连接数（默认20）
            max_connections=200,            # 最大连接数（默认100）
            keepalive_expiry=60.0,          # Keep-Alive 超时（秒）
        )
        
        _shared_sync_client = httpx.Client(
            limits=limits,
            timeout=httpx.Timeout(
                connect=10.0,   # 连接超时
                read=60.0,      # 读取超时
                write=30.0,     # 写入超时
                pool=10.0,      # 连接池等待超时
            ),
            http2=False,  # 禁用 HTTP/2（需要额外安装 h2 包）
        )
        print("[HTTPClient] ✅ 创建共享 HTTP 客户端池 (max=200, keepalive=100)")
    
    return _shared_sync_client


def get_shared_async_httpx_client() -> httpx.AsyncClient:
    """
    获取共享的异步 httpx 客户端
    """
    global _shared_async_client
    
    if _shared_async_client is None:
        limits = httpx.Limits(
            max_keepalive_connections=100,
            max_connections=200,
            keepalive_expiry=60.0,
        )
        
        _shared_async_client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=30.0,
                pool=10.0,
            ),
            http2=False,  # 禁用 HTTP/2
        )
        print("[HTTPClient] ✅ 创建共享异步 HTTP 客户端池")
    
    return _shared_async_client


@lru_cache(maxsize=1)
def get_siliconflow_chatmodel():
    """
    获取共享的 SiliconFlow ChatOpenAI 实例
    
    使用单例模式，所有工具共享同一个 LLM 客户端
    """
    # 向后兼容：原先固定返回 72B 的单例
    return get_siliconflow_chat_openai(
        model="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.7,
        max_tokens=200,
        timeout=30,
    )


@lru_cache(maxsize=32)
def get_siliconflow_chat_openai(
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Any = 30,
    max_retries: int = 1,
    streaming: bool = False,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    获取共享的 SiliconFlow ChatOpenAI 实例（按配置缓存）。
    
    目的：
    - 复用同一个 ChatOpenAI 对象（避免重复初始化）
    - 复用同一个 httpx.Client / httpx.AsyncClient（连接池 + Keep-Alive）
    
    注意：
    - 不同参数（model/temperature/max_tokens/timeout/streaming）会生成不同缓存实例
    """
    import os
    from langchain_openai import ChatOpenAI

    http_client = get_shared_httpx_client()
    http_async_client = get_shared_async_httpx_client()

    _base_url = base_url or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    _api_key = api_key or os.getenv("SILICONFLOW_API_KEY")

    kwargs = dict(
        model=model,
        base_url=_base_url,
        api_key=_api_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        streaming=streaming,
        http_client=http_client,
        http_async_client=http_async_client,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    llm = ChatOpenAI(**kwargs)
    print(
        f"[HTTPClient] ✅ 创建共享 ChatOpenAI: model={model} temp={temperature} "
        f"max_tokens={max_tokens} streaming={streaming} timeout={timeout}"
    )
    return llm


def cleanup_clients():
    """清理客户端资源（应用关闭时调用）"""
    global _shared_sync_client, _shared_async_client
    
    if _shared_sync_client:
        _shared_sync_client.close()
        _shared_sync_client = None
        
    if _shared_async_client:
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(_shared_async_client.aclose())
        except:
            pass
        _shared_async_client = None
    
    print("[HTTPClient] 🔒 已清理共享客户端")
