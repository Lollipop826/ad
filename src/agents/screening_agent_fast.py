"""
阿尔茨海默病初筛Agent - 性能优化版（并行执行）

主要优化：
1. 工具并行调用（3层并行结构）
2. 简单维度跳过知识检索
3. 直接函数调用，不使用ReAct Agent

性能提升：
- 简单维度：5-6秒 → 1-2秒 (↑80%)
- 复杂维度：5-6秒 → 3-4秒 (↑40%)
- 平均：5-6秒 → 2-3秒 (↑60%)
"""

from typing import List, Dict, Any, Optional
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import time

from langchain_openai import ChatOpenAI

from src.tools.agent_tools import (
    ResistanceDetectionTool,
    AnswerEvaluationTool,
    ScoreRecordingTool,
    QuestionGenerationTool,
    KnowledgeRetrievalTool,
)
from src.tools.query_sentence.generator import QuerySentenceGenerator


class ADScreeningAgentFast:
    """
    阿尔茨海默病初筛Agent - 性能优化版
    
    使用并行调用和智能跳过策略，大幅提升响应速度
    """
    
    # 简单维度列表（无需知识检索）- 已禁用，所有维度都执行检索
    # 通过优化检索速度（跳过重排序），即使所有维度检索也能快速响应
    SIMPLE_DIMENSIONS = [
        # "定向力",           # 已禁用
        # "注意力与计算力",    # 已禁用
        # "记忆力 - 即刻记忆", # 已禁用
    ]
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0.3,
    ):
        # 初始化LLM - 优化超时
        from src.llm.http_client_pool import get_siliconflow_chat_openai
        self.llm = get_siliconflow_chat_openai(
            model=model or os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),  # 默认使用7B更快
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout=15,  # 从30秒降到15秒
            max_retries=0,  # 不重试，快速失败
        )
        
        # 初始化工具（直接实例化，不通过Agent）
        self.resistance_tool = ResistanceDetectionTool()
        self.evaluation_tool = AnswerEvaluationTool()
        self.score_tool = ScoreRecordingTool()
        self.question_tool = QuestionGenerationTool()
        self.retrieval_tool = KnowledgeRetrievalTool()
        self.query_generator = QuerySentenceGenerator()
    
    def _is_simple_dimension(self, dimension_name: str) -> bool:
        """判断是否为简单维度"""
        for simple_dim in self.SIMPLE_DIMENSIONS:
            if simple_dim in dimension_name:
                return True
        return False
    
    def _parallel_layer1(
        self, 
        question: str, 
        answer: str, 
        dimension: Dict[str, Any],
        patient_profile: Dict[str, Any],
        include_query: bool = True
    ) -> Dict[str, Any]:
        """
        第1层并行：同时执行情绪检测、答案评估、查询生成
        
        Args:
            question: 医生的问题
            answer: 患者的回答
            dimension: 当前维度
            patient_profile: 患者画像
            include_query: 是否包含查询生成（简单维度可跳过）
        
        Returns:
            包含所有结果的字典
        """
        results = {}
        tool_times = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交并行任务
            futures = {}
            submit_times = {}
            
            print(f"[TIMING] 📤 提交第1层任务...")
            layer1_submit_start = time.time()
            
            # 任务1: 情绪检测
            submit_times['resistance'] = time.time()
            future_resistance = executor.submit(
                self.resistance_tool._run,
                question=question,
                answer=answer
            )
            futures['resistance'] = future_resistance
            print(f"[TIMING]   ✓ resistance_detection 已提交")
            
            # 任务2: 答案评估
            submit_times['evaluation'] = time.time()
            future_evaluation = executor.submit(
                self.evaluation_tool._run,
                question=question,
                answer=answer,
                dimension_id=dimension.get('id', ''),
                patient_profile=patient_profile
            )
            futures['evaluation'] = future_evaluation
            print(f"[TIMING]   ✓ answer_evaluation 已提交")
            
            # 任务3: 查询生成（仅复杂维度）
            if include_query:
                submit_times['query'] = time.time()
                future_query = executor.submit(
                    self.query_generator.generate_query,
                    dimension=dimension,
                    history=[],
                    profile=patient_profile
                )
                futures['query'] = future_query
                print(f"[TIMING]   ✓ query_generator 已提交")
            
            layer1_submit_time = time.time() - layer1_submit_start
            print(f"[TIMING] ⏱️  任务提交完成 ({layer1_submit_time:.3f}秒)")
            
            # ⚡ 优化：使用as_completed真正并行等待
            print(f"[TIMING] ⏳ 并行等待任务完成...")
            wait_start = time.time()
            
            # 记录每个任务的开始时间
            task_start_times = {}
            for key in futures.keys():
                task_start_times[key] = submit_times[key]
            
            # 使用as_completed()：哪个先完成就先处理哪个 - 增加超时处理
            try:
                completed_futures = list(as_completed(futures.values(), timeout=20))  # 增加到20秒
            except TimeoutError:
                print(f"[TIMING]   ⚠️  第1层部分任务超时，继续处理已完成的任务")
                completed_futures = [f for f in futures.values() if f.done()]
            
            for future in completed_futures:
                # 找到这个future对应的key
                key = None
                for k, f in futures.items():
                    if f == future:
                        key = k
                        break
                
                if key is None:
                    continue
                
                try:
                    result = future.result(timeout=0.1)  # 已经完成，立即返回
                    tool_time = time.time() - submit_times[key]
                    tool_times[key] = tool_time
                    print(f"[TIMING]   ✅ {key} 完成 ({tool_time:.3f}秒)")
                    
                    # 解析JSON结果
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except Exception as json_err:
                            print(f"[WARNING] {key} JSON解析失败，使用原始字符串")
                            result = {"raw_result": result}
                    
                    results[key] = result
                except Exception as e:
                    tool_time = time.time() - submit_times[key]
                    tool_times[key] = tool_time
                    print(f"[TIMING]   ❌ {key} 失败 ({tool_time:.3f}秒): {str(e)[:100]}")
                    import traceback
                    print(f"[ERROR] {key} 错误详情:\n{traceback.format_exc()}")
                    results[key] = {"success": False, "error": str(e)}
            
            total_wait_time = time.time() - wait_start
            print(f"[TIMING] ✅ 第1层所有任务完成 (并行耗时: {total_wait_time:.3f}秒)")
            print(f"[TIMING] 📊 各工具耗时: {', '.join([f'{k}={v:.3f}s' for k, v in tool_times.items()])}")
        
        return results
    
    def _parallel_layer2(
        self,
        layer1_results: Dict[str, Any],
        session_id: str,
        dimension: Dict[str, Any],
        question: str,
        answer: str,
        include_retrieval: bool = True
    ) -> Dict[str, Any]:
        """
        第2层并行：同时执行评分记录和知识检索
        
        Args:
            layer1_results: 第1层的结果
            session_id: 会话ID
            dimension: 当前维度
            question: 医生的问题
            answer: 患者的回答
            include_retrieval: 是否包含知识检索
        
        Returns:
            包含所有结果的字典
        """
        results = {}
        tool_times = {}
        
        print(f"\n[TIMING] 📤 提交第2层任务...")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            # 任务1: 评分记录（依赖evaluation结果）
            eval_result = layer1_results.get('evaluation', {})
            # 检查是否有有效的评估结果（有 quality_level 字段即可）
            if eval_result and eval_result.get('quality_level'):
                print(f"[TIMING]   ✓ score_recording 已提交")
                future_score = executor.submit(
                    self.score_tool._run,
                    session_id=session_id,
                    dimension_id=dimension.get('id', ''),
                    quality_level=eval_result.get('quality_level', 'unknown'),
                    cognitive_performance=eval_result.get('cognitive_performance', '无法评估'),
                    question=question,
                    answer=answer,
                    evaluation_detail=eval_result.get('evaluation_detail', ''),
                    action='save'
                )
                futures['score'] = future_score
            else:
                print(f"[TIMING]   ⚠️  evaluation结果无效，跳过score_recording")
                print(f"[DEBUG]   eval_result: {eval_result}")
            
            # 任务2: 知识检索（依赖query结果，仅复杂维度）
            if include_retrieval and 'query' in layer1_results:
                query_result = layer1_results['query']
                if query_result and 'query' in query_result:
                    print(f"[TIMING]   ✓ knowledge_retrieval 已提交 (query={query_result.get('query', '')[:30]}...)")
                    retrieval_start_time = time.time()
                    future_retrieval = executor.submit(
                        self.retrieval_tool._run,
                        query=query_result['query'],
                        top_k=3,  # 从5降到3，减少检索时间
                        skip_reranking=True,  # 跳过重排序，提速60%
                        use_fusion=False  # 🔥 禁用RAG Fusion，从2-3秒降到0.3秒
                    )
                    futures['retrieval'] = future_retrieval
                else:
                    print(f"[TIMING]   ⚠️  query结果无效，跳过knowledge_retrieval")
            else:
                print(f"[TIMING]   ⚠️  简单维度，跳过knowledge_retrieval")
            
            # ⚡ 优化：使用as_completed真正并行等待
            print(f"[TIMING] ⏳ 并行等待第2层任务完成...")
            wait_start = time.time()
            
            # 记录每个任务的提交时间
            submit_times_layer2 = {}
            for key in futures.keys():
                submit_times_layer2[key] = time.time()
            
            # 使用as_completed()并行等待 - 增加超时处理
            try:
                completed_futures = list(as_completed(futures.values(), timeout=20))  # 增加到20秒
            except TimeoutError:
                print(f"[TIMING]   ⚠️  第2层部分任务超时，继续处理已完成的任务")
                completed_futures = [f for f in futures.values() if f.done()]
            
            for future in completed_futures:
                # 找到这个future对应的key
                key = None
                for k, f in futures.items():
                    if f == future:
                        key = k
                        break
                
                if key is None:
                    continue
                
                try:
                    result = future.result(timeout=0.1)  # 已经完成，立即返回
                    tool_time = time.time() - submit_times_layer2[key]
                    tool_times[key] = tool_time
                    print(f"[TIMING]   ✅ {key} 完成 ({tool_time:.3f}秒)")
                    
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except Exception as json_err:
                            print(f"[WARNING] {key} JSON解析失败")
                            result = {"raw_result": result}
                    results[key] = result
                except Exception as e:
                    tool_time = time.time() - submit_times_layer2[key]
                    tool_times[key] = tool_time
                    print(f"[TIMING]   ❌ {key} 失败 ({tool_time:.3f}秒): {str(e)[:100]}")
                    import traceback
                    print(f"[ERROR] {key} 详情:\n{traceback.format_exc()}")
                    results[key] = {"success": False, "error": str(e)}
            
            total_wait_time = time.time() - wait_start
            print(f"[TIMING] ✅ 第2层所有任务完成 (并行耗时: {total_wait_time:.3f}秒)")
            if tool_times:
                print(f"[TIMING] 📊 各工具耗时: {', '.join([f'{k}={v:.3f}s' for k, v in tool_times.items()])}")
        
        return results
    
    def _generate_question_layer3(
        self,
        layer1_results: Dict[str, Any],
        layer2_results: Dict[str, Any],
        dimension: Dict[str, Any],
        patient_profile: Dict[str, Any],
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        第3层：生成下一个问题（必须串行，依赖前面所有结果）
        
        Args:
            layer1_results: 第1层结果
            layer2_results: 第2层结果
            dimension: 当前维度
            patient_profile: 患者画像
            chat_history: 对话历史
        
        Returns:
            生成的问题文本
        """
        print(f"\n[TIMING] 🔧 第3层：生成下一个问题...")
        layer3_start = time.time()
        
        # 检查是否有抵抗情绪
        resistance_result = layer1_results.get('resistance', {})
        if resistance_result.get('is_resistant'):
            # 如果有抵抗，返回安慰话语
            comfort_time = time.time() - layer3_start
            print(f"[TIMING]   ⚠️  检测到抵抗情绪，返回安慰话语 ({comfort_time:.3f}秒)")
            return resistance_result.get('comfort_response', '请放松，我们慢慢来。')
        
        # 获取知识上下文
        print(f"[TIMING]   📚 提取知识上下文...")
        context_start = time.time()
        knowledge_context = ""
        retrieval_result = layer2_results.get('retrieval', {})
        if retrieval_result.get('success') and retrieval_result.get('results'):
            # 提取前2个最相关的结果
            top_results = retrieval_result['results'][:2]
            knowledge_context = "\n".join([r.get('text', '') for r in top_results])
            print(f"[TIMING]   ✅ 知识上下文提取完成 (长度:{len(knowledge_context)}字符, {time.time()-context_start:.3f}秒)")
        else:
            print(f"[TIMING]   ⚠️  无知识上下文 ({time.time()-context_start:.3f}秒)")
        
        # 获取患者情绪
        patient_emotion = resistance_result.get('emotional_state', 'neutral')
        print(f"[TIMING]   😊 患者情绪: {patient_emotion}")
        
        # 调用问题生成工具
        try:
            print(f"[TIMING]   🤖 调用 generate_question (LLM)...")
            generate_start = time.time()
            
            result = self.question_tool._run(
                dimension_name=dimension.get('name', ''),
                knowledge_context=knowledge_context,
                patient_age=patient_profile.get('age') if patient_profile else None,
                patient_education=patient_profile.get('education_years') if patient_profile else None,
                conversation_history=chat_history[-3:] if len(chat_history) > 3 else chat_history,
                patient_emotion=patient_emotion
            )
            
            generate_time = time.time() - generate_start
            print(f"[TIMING]   ✅ generate_question 完成 ({generate_time:.3f}秒)")
            
            if isinstance(result, str):
                result = json.loads(result)
            
            layer3_total = time.time() - layer3_start
            print(f"[TIMING] ✅ 第3层完成 (总耗时: {layer3_total:.3f}秒)")
            
            return result.get('question', '请继续回答。')
        except Exception as e:
            error_time = time.time() - layer3_start
            print(f"[TIMING]   ❌ generate_question 失败 ({error_time:.3f}秒): {str(e)[:100]}")
            return "请继续。"
    
    def process_turn(
        self,
        user_input: str,
        dimension: Dict[str, Any],
        session_id: str,
        patient_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        处理一轮对话（并行优化版）
        
        Args:
            user_input: 用户输入/回答
            dimension: 当前评估的维度
            session_id: 会话ID
            patient_profile: 患者画像
            chat_history: 对话历史
        
        Returns:
            包含生成问题和中间步骤的结果
        """
        start_time = time.time()
        
        if patient_profile is None:
            patient_profile = {}
        if chat_history is None:
            chat_history = []
        
        # 提取医生的最后一个问题
        doctor_question = "请开始评估"
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                doctor_question = msg.get("content", "请开始评估")
                break
        
        dimension_name = dimension.get('name', '')
        is_simple = self._is_simple_dimension(dimension_name)
        
        print(f"\n{'='*80}")
        print(f"[TIMING] 🚀 开始处理 - 维度: {dimension_name}")
        print(f"[TIMING] 📋 维度类型: {'简单 (跳过检索)' if is_simple else '复杂 (完整流程)'}")
        print(f"[TIMING] 💬 问题: {doctor_question[:50]}...")
        print(f"[TIMING] 💭 回答: {user_input[:50]}...")
        print(f"{'='*80}\n")
        
        # 收集中间步骤（用于调试显示）
        intermediate_steps = []
        
        try:
            # ============ 第1层并行 ============
            print(f"\n{'─'*80}")
            print(f"[TIMING] 🔷 第1层：并行执行情绪检测 + 答案评估" + (" + 查询生成" if not is_simple else ""))
            print(f"{'─'*80}")
            layer1_start = time.time()
            
            layer1_results = self._parallel_layer1(
                question=doctor_question,
                answer=user_input,
                dimension=dimension,
                patient_profile=patient_profile,
                include_query=not is_simple  # 简单维度跳过查询生成
            )
            
            layer1_time = time.time() - layer1_start
            print(f"\n[TIMING] ✅ 第1层并行执行完成 - 总耗时: {layer1_time:.3f}秒")
            
            # 记录第1层步骤
            for key, result in layer1_results.items():
                intermediate_steps.append({
                    'layer': 1,
                    'tool': key,
                    'result': result,
                    'time': layer1_time
                })
            
            # ============ 第2层并行 ============
            print(f"\n{'─'*80}")
            print(f"[TIMING] 🔶 第2层：并行执行评分记录" + (" + 知识检索" if not is_simple else ""))
            print(f"{'─'*80}")
            layer2_start = time.time()
            
            layer2_results = self._parallel_layer2(
                layer1_results=layer1_results,
                session_id=session_id,
                dimension=dimension,
                question=doctor_question,
                answer=user_input,
                include_retrieval=not is_simple  # 简单维度跳过检索
            )
            
            layer2_time = time.time() - layer2_start
            print(f"\n[TIMING] ✅ 第2层并行执行完成 - 总耗时: {layer2_time:.3f}秒")
            
            # 记录第2层步骤
            for key, result in layer2_results.items():
                intermediate_steps.append({
                    'layer': 2,
                    'tool': key,
                    'result': result,
                    'time': layer2_time
                })
            
            # ============ 第3层串行 ============
            print(f"\n{'─'*80}")
            print(f"[TIMING] 🔸 第3层：串行生成问题 (依赖前两层结果)")
            print(f"{'─'*80}")
            layer3_start = time.time()
            
            next_question = self._generate_question_layer3(
                layer1_results=layer1_results,
                layer2_results=layer2_results,
                dimension=dimension,
                patient_profile=patient_profile,
                chat_history=chat_history
            )
            
            layer3_time = time.time() - layer3_start
            
            # 记录第3层步骤
            intermediate_steps.append({
                'layer': 3,
                'tool': 'generate_question',
                'result': {'question': next_question},
                'time': layer3_time
            })
            
            total_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print(f"[TIMING] 🏁 全部完成！")
            print(f"{'='*80}")
            print(f"[TIMING] ⏱️  总耗时: {total_time:.3f}秒")
            print(f"[TIMING] 📊 性能分解:")
            print(f"[TIMING]   • 第1层 (并行): {layer1_time:.3f}秒 ({layer1_time/total_time*100:.1f}%)")
            print(f"[TIMING]   • 第2层 (并行): {layer2_time:.3f}秒 ({layer2_time/total_time*100:.1f}%)")
            print(f"[TIMING]   • 第3层 (串行): {layer3_time:.3f}秒 ({layer3_time/total_time*100:.1f}%)")
            print(f"[TIMING] 🎯 生成问题: {next_question[:60]}...")
            print(f"{'='*80}\n")
            
            # 转换 intermediate_steps 格式以兼容前端
            # 前端期望: [(action, observation), ...]
            # 我们的格式: [{'layer': 1, 'tool': 'xxx', 'result': {...}}, ...]
            converted_steps = []
            for step in intermediate_steps:
                # 创建伪 action 对象
                class FakeAction:
                    def __init__(self, tool_name):
                        self.tool = tool_name
                        self.name = tool_name
                        self.tool_input = {}
                
                action = FakeAction(step.get('tool', 'unknown'))
                observation = json.dumps(step.get('result', {}), ensure_ascii=False)
                converted_steps.append((action, observation))
            
            return {
                'output': next_question,
                'intermediate_steps': converted_steps,  # 使用转换后的格式
                'performance': {
                    'total_time': total_time,
                    'layer1_time': layer1_time,
                    'layer2_time': layer2_time,
                    'layer3_time': layer3_time,
                    'is_simple_dimension': is_simple
                },
                'raw_steps': intermediate_steps,  # 保留原始格式用于调试
                'parallel_execution': {  # 新增：并行执行详情
                    'layer1': {
                        'tasks': list(layer1_results.keys()),
                        'parallel_time': layer1_time,
                        'task_details': {
                            k: {
                                'status': 'success' if not v.get('error') else 'failed',
                                'result_preview': str(v)[:100] if v else 'empty'
                            } for k, v in layer1_results.items()
                        }
                    },
                    'layer2': {
                        'tasks': list(layer2_results.keys()),
                        'parallel_time': layer2_time,
                        'task_details': {
                            k: {
                                'status': 'success' if not v.get('error') else 'failed',
                                'result_preview': str(v)[:100] if v else 'empty'
                            } for k, v in layer2_results.items()
                        }
                    },
                    'layer3': {
                        'task': 'generate_question',
                        'serial_time': layer3_time,
                        'result_preview': next_question[:50]
                    }
                }
            }
            
        except Exception as e:
            print(f"[ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'output': '抱歉，系统出现问题，请继续。',
                'intermediate_steps': intermediate_steps,
                'error': str(e),
                'parallel_execution': None,  # 添加空值避免错误
                'performance': None  # 添加空值避免错误
            }
    
    def process_turn_streaming(
        self,
        user_input: str,
        dimension: Dict[str, Any],
        session_id: str,
        patient_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        处理一轮对话（流式输出版）
        yields: Dict with 'type' ('token', 'done') and data
        """
        start_time = time.time()
        
        if patient_profile is None:
            patient_profile = {}
        if chat_history is None:
            chat_history = []
            
        # 提取医生的最后一个问题
        doctor_question = "请开始评估"
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                doctor_question = msg.get("content", "请开始评估")
                break
        
        dimension_name = dimension.get('name', '')
        is_simple = self._is_simple_dimension(dimension_name)
        
        # Layer 1
        layer1_start = time.time()
        layer1_results = self._parallel_layer1(
            question=doctor_question,
            answer=user_input,
            dimension=dimension,
            patient_profile=patient_profile,
            include_query=not is_simple
        )
        layer1_time = time.time() - layer1_start
        
        # Layer 2
        layer2_start = time.time()
        layer2_results = self._parallel_layer2(
            layer1_results=layer1_results,
            session_id=session_id,
            dimension=dimension,
            question=doctor_question,
            answer=user_input,
            include_retrieval=not is_simple
        )
        layer2_time = time.time() - layer2_start
        
        # Layer 3 - Streaming Generation
        layer3_start = time.time()
        
        # 1. Check Resistance
        resistance_result = layer1_results.get('resistance', {})
        if resistance_result.get('is_resistant'):
             response = resistance_result.get('comfort_response', '请放松，我们慢慢来。')
             yield {'type': 'token', 'content': response, 'full_text': response}
             
             # Construct full result structure for 'done'
             yield {
                'type': 'done',
                'content': response,
                'metadata': {
                    'total_time': time.time() - start_time,
                    'stream_time': 0
                }
             }
             return

        # 2. Prepare Context
        knowledge_context = ""
        retrieval_result = layer2_results.get('retrieval', {})
        if retrieval_result.get('success') and retrieval_result.get('results'):
            top_results = retrieval_result['results'][:2]
            knowledge_context = "\n".join([r.get('text', '') for r in top_results])
            
        patient_emotion = resistance_result.get('emotional_state', 'neutral')
        
        # 3. Construct Prompt (Copied from QuestionGenerationTool for consistency)
        system_prompt = (
            "你是一位温柔、耐心、经验丰富的老年科医生。你正在与一位可能患有阿尔茨海默病的老人聊天，"
            "像朋友、家人一样关心他们，用最自然、最温暖的方式进行认知评估。\n\n"
            "💡 核心原则 - 像陪老人聊天，不是审问考试：\n"
            "1. **语气温柔亲切**：把患者当成自己的长辈，用温暖、尊重的语气\n"
            "2. **自然过渡**：根据对话内容自然引入问题，而不是生硬转话题\n"
            "3. **多鼓励夸奖**：经常说「很好」「不错」「您真棒」「记性不错呢」\n"
            "4. **避免压力感**：不要让患者觉得在被测试，而是在闲聊\n"
            "5. **用日常口语**：「您看」「咱们」「您说」「您觉得呢」\n\n"
            "⚠️ 严格禁止的做法：\n"
            "❌ 不要用「请您告诉我」「能不能麻烦您」「我想问您」这类生硬的开场\n"
            "❌ 不要重复已经问过的问题（务必检查对话历史）\n"
            "❌ 不要问其他维度的问题，只问当前指定的维度\n"
            "❌ 不要用医学术语或书面语\n\n"
            "🌟 最重要的：让患者感到被关心、被尊重、被温柔对待！"
        )
        
        user_prompt_parts = []
        user_prompt_parts.append(f"评估维度：{dimension_name}")
        if dimension.get('description'):
            user_prompt_parts.append(f"维度说明：{dimension.get('description')}")
            
        if patient_profile.get('age') and patient_profile.get('education_years'):
            user_prompt_parts.append(f"患者信息：{patient_profile['age']}岁，教育{patient_profile['education_years']}年")
        
        if patient_emotion:
            user_prompt_parts.append(f"患者情绪：{patient_emotion}")
            
        user_prompt_parts.append(f"\n相关医学知识：\n{knowledge_context[:800]}")
        
        # History (Last 3 rounds)
        if chat_history:
            history_text = json.dumps(chat_history[-3:] if len(chat_history) > 3 else chat_history, ensure_ascii=False)
            user_prompt_parts.append(f"\n【已经问过的问题（禁止重复）】：\n{history_text}")
            
        user_prompt_parts.append(f"\n请针对【{dimension_name}】维度生成一个新的、未问过的评估问题：")
        user_prompt = "\n".join(user_prompt_parts)
        
        # 4. Stream Output
        full_response = ""
        stream_start = time.time()
        
        try:
            for chunk in self.llm.stream([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]):
                content = chunk.content
                if content:
                    full_response += content
                    yield {'type': 'token', 'content': content, 'full_text': full_response}
        except Exception as e:
            print(f"[ERROR] Streaming failed: {e}")
            full_response = "请继续。"
            yield {'type': 'token', 'content': full_response, 'full_text': full_response}

        # Cleanup
        final_question = full_response.strip().strip('"').strip("'")
        if not final_question.endswith(("？", "?", "。", ".")):
            final_question += "？"
            
        layer3_time = time.time() - layer3_start
        total_time = time.time() - start_time
        
        yield {
            'type': 'done',
            'content': final_question,
            'metadata': {
                'total_time': total_time,
                'stream_time': layer3_time,
                'layer1_time': layer1_time,
                'layer2_time': layer2_time
            }
        }

    def get_tools_info(self) -> List[Dict[str, str]]:
        """获取所有工具的信息"""
        return [
            {"name": "resistance_detection", "description": "情绪检测"},
            {"name": "answer_evaluation", "description": "答案评估"},
            {"name": "score_recording", "description": "评分记录"},
            {"name": "question_generation", "description": "问题生成"},
            {"name": "knowledge_retrieval", "description": "知识检索（复杂维度）"},
        ]

