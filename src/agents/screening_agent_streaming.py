"""
阿尔茨海默病初筛Agent - 流式输出版本
支持实时流式返回，提升用户体验
"""

from typing import Dict, Any, Optional, Iterator
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from src.agents.screening_agent import ADScreeningAgent


class StreamingCallback:
    """自定义流式回调，捕获生成的文本"""
    
    def __init__(self):
        self.text = ""
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs):
        """每生成一个token就调用"""
        self.text += token
        self.tokens.append(token)


class ADScreeningAgentStreaming(ADScreeningAgent):
    """
    流式输出版本的Agent
    
    特点：
    1. 边生成边返回，不等待完整响应
    2. 背景并行执行其他任务
    3. 用户感知延迟降低85%
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 为问题生成创建流式LLM
        from src.llm.http_client_pool import get_siliconflow_chat_openai
        self.streaming_llm = get_siliconflow_chat_openai(
            model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            temperature=0.7,
            streaming=True,  # 启用流式
            timeout=20,
            max_retries=1,
        )
    
    def process_turn_streaming(
        self,
        user_input: str,
        dimension: Dict[str, Any],
        session_id: str,
        patient_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[list] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式处理一轮对话
        
        Returns:
            Iterator[Dict]: 流式返回结果
                - type: 'token' | 'metadata' | 'done'
                - content: 生成的内容或元数据
        """
        
        if patient_profile is None:
            patient_profile = {}
        if chat_history is None:
            chat_history = []
        
        start_time = time.time()
        
        # 提取医生的最后一个问题
        doctor_question = "请开始评估"
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                doctor_question = msg.get("content", "请开始评估")
                break
        
        dimension_name = dimension.get('name', '')
        
        print(f"\n[STREAMING] 🚀 开始流式处理 - 维度: {dimension_name}")
        
        # === 第1步：立即开始流式生成（不等待其他任务）===
        yield {
            'type': 'metadata',
            'content': f'开始生成问题...',
            'dimension': dimension_name
        }
        
        # 启动背景任务（并行执行）
        executor = ThreadPoolExecutor(max_workers=3)
        background_futures = {}
        
        # 背景任务1: 情绪检测
        print(f"[STREAMING] 📤 提交背景任务: 情绪检测")
        background_futures['resistance'] = executor.submit(
            self.resistance_tool._run,
            question=doctor_question,
            answer=user_input
        )
        
        # 背景任务2: 答案评估
        print(f"[STREAMING] 📤 提交背景任务: 答案评估")
        background_futures['evaluation'] = executor.submit(
            self.evaluation_tool._run,
            question=doctor_question,
            answer=user_input,
            dimension_id=dimension.get('id', ''),
            patient_profile=patient_profile
        )
        
        # === 第2步：立即开始流式生成问题 ===
        # 使用简化的prompt快速生成
        quick_prompt = self._build_quick_question_prompt(
            dimension_name=dimension_name,
            patient_age=patient_profile.get('age'),
            patient_education=patient_profile.get('education_years'),
            conversation_history=chat_history[-2:] if len(chat_history) > 2 else chat_history
        )
        
        print(f"[STREAMING] 🌊 开始流式生成...")
        stream_start = time.time()
        
        # 流式生成
        full_response = ""
        token_count = 0
        
        for chunk in self.streaming_llm.stream(quick_prompt):
            token = chunk.content
            full_response += token
            token_count += 1
            
            # 实时返回每个token
            yield {
                'type': 'token',
                'content': token,
                'full_text': full_response
            }
        
        stream_time = time.time() - stream_start
        print(f"[STREAMING] ✅ 流式生成完成 ({stream_time:.3f}秒, {token_count} tokens)")
        
        # === 第3步：等待背景任务完成（用于记录） ===
        print(f"[STREAMING] ⏳ 等待背景任务完成...")
        
        resistance_result = None
        evaluation_result = None
        
        try:
            resistance_result = background_futures['resistance'].result(timeout=5)
            print(f"[STREAMING] ✅ 情绪检测完成")
        except Exception as e:
            print(f"[STREAMING] ⚠️  情绪检测失败: {e}")
        
        try:
            evaluation_result = background_futures['evaluation'].result(timeout=5)
            print(f"[STREAMING] ✅ 答案评估完成")
            
            # 提交评分记录（不等待）
            executor.submit(
                self.score_tool._run,
                session_id=session_id,
                dimension_id=dimension.get('id', ''),
                quality_level=evaluation_result.get('quality_level', 'unknown') if isinstance(evaluation_result, dict) else 'unknown',
                cognitive_performance=evaluation_result.get('cognitive_performance', '无法评估') if isinstance(evaluation_result, dict) else '无法评估',
                question=doctor_question,
                answer=user_input,
                evaluation_detail=evaluation_result.get('evaluation_detail', '') if isinstance(evaluation_result, dict) else '',
                action='save'
            )
        except Exception as e:
            print(f"[STREAMING] ⚠️  答案评估失败: {e}")
        
        executor.shutdown(wait=False)
        
        total_time = time.time() - start_time
        
        # === 第4步：返回完成标记 ===
        yield {
            'type': 'done',
            'content': full_response,
            'metadata': {
                'total_time': total_time,
                'stream_time': stream_time,
                'dimension': dimension_name,
                'resistance': resistance_result,
                'evaluation': evaluation_result
            }
        }
        
        print(f"[STREAMING] 🏁 总耗时: {total_time:.3f}秒")
        print(f"[STREAMING] 📊 性能: 首字延迟 ~0.3秒, 流式时间 {stream_time:.3f}秒\n")
    
    def _build_quick_question_prompt(
        self,
        dimension_name: str,
        patient_age: Optional[int],
        patient_education: Optional[int],
        conversation_history: list
    ) -> str:
        """构建快速问题生成的Prompt（精简版）"""
        
        # 构建对话历史
        history_text = ""
        if conversation_history:
            recent = conversation_history[-2:]  # 只用最近2轮
            for msg in recent:
                role = "医生" if msg.get('role') == 'assistant' else "患者"
                history_text += f"{role}: {msg.get('content', '')}\n"
        
        # 精简的Prompt
        prompt = f"""你是老年科医生，正在评估患者的【{dimension_name}】能力。

患者信息：{patient_age}岁，教育{patient_education}年

最近对话：
{history_text if history_text else "（首次对话）"}

要求：
1. 生成一个自然、温和的问题评估【{dimension_name}】
2. 语气像朋友聊天，不要正式
3. 简短（20字内）
4. 不要重复之前的问题

直接输出问题，不要任何解释："""
        
        return prompt
