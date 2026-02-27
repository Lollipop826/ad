"""
阿尔茨海默病初筛Agent

使用LLM作为Agent，自主决定调用哪些工具来完成对话任务
"""

from typing import List, Dict, Any, Optional
import os

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

# 本地模型支持
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

from src.tools.agent_tools import (
    QueryTool,
    KnowledgeRetrievalTool,
    DimensionDetectionTool,
    ResistanceDetectionTool,
    QuestionGenerationTool,
    ConversationStorageTool,
    AnswerEvaluationTool,
    ScoreRecordingTool,
)
from src.tools.query_sentence.generator import QuerySentenceGenerator


class ADScreeningAgent:
    """
    阿尔茨海默病初筛Agent
    
    自主决定何时：
    - 生成检索查询
    - 检索知识
    - 分析用户回答
    - 生成下一个问题
    - 保存对话记录
    """
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0.3,
        use_local: bool = None,  # 是否使用本地模型
    ):
        # 决定是否使用本地模型
        use_local = use_local if use_local is not None else USE_LOCAL_MODEL
        
        if use_local:
            # 使用本地 Qwen 模型（从模型池获取，复用实例）
            print("[Agent] 🚀 使用本地 Qwen2.5-7B 模型（模型池）")
            from src.llm.model_pool import get_pooled_llm
            self.llm = get_pooled_llm(pool_key='agent')  # 使用池化的 Agent 实例
            self.streaming_llm = self.llm  # 本地模型暂不支持流式
        else:
            # 使用 API
            print("[Agent] 🌐 使用 API 模型")
            from src.llm.http_client_pool import get_siliconflow_chat_openai
            self.llm = get_siliconflow_chat_openai(
                model=model or os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                timeout=15,
                max_retries=1,
            )
            
            # 流式LLM（用于流式输出）
            self.streaming_llm = get_siliconflow_chat_openai(
                model=model or os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
                base_url=base_url,
                api_key=api_key,
                temperature=0.7,
                streaming=True,
                timeout=15,
                max_retries=1,
            )
        
        # 延迟初始化工具（提高启动速度）
        self._tools = None
        self._agent = None
        
        # 是否启用 RAG（需要下载 embedding 模型）
        self.enable_rag = os.getenv("ENABLE_RAG", "false").lower() == "true"
        
        # 创建Agent（延迟加载工具）
        self.agent = self._create_agent()
    
    @property
    def tools(self) -> List[BaseTool]:
        """延迟加载工具"""
        if self._tools is None:
            print("[Agent] 正在初始化工具...")
            
            # 判断是否使用本地模型
            use_local = USE_LOCAL_MODEL
            
            self._tools = [
                QueryTool(use_local=use_local),                      # 查询生成
                KnowledgeRetrievalTool(),                            # 知识检索 (不需要LLM)
                DimensionDetectionTool(use_local=use_local),         # 维度检测
                ResistanceDetectionTool(use_local=use_local),        # 抵抗检测
                QuestionGenerationTool(use_local=use_local),         # 问题生成
                ConversationStorageTool(),                           # 对话存储 (不需要LLM)
                AnswerEvaluationTool(use_local=use_local),           # 回答评估
                ScoreRecordingTool(),                                # 评分记录 (不需要LLM)
            ]
            print(f"[Agent] ✅ 工具初始化完成 (本地模式: {use_local})")
            
            # 预热工具（避免首次调用延迟）
            if use_local:
                self._warmup_tools()
        return self._tools
    
    def _warmup_tools(self):
        """预热工具（首次调用初始化Python解释器、导入等）"""
        print("[Agent] 🔥 开始预热工具...")
        import time
        start_time = time.time()
        
        try:
            # 找到已创建的工具实例并预热
            for tool in self._tools:
                tool_name = tool.name
                
                # 预热 ResistanceTool
                if tool_name == "resistance_detection_tool":
                    _ = tool._run(question="测试", answer="好的")
                
                # 预热 AnswerEvalTool
                elif tool_name == "answer_evaluation_tool":
                    _ = tool._run(question="测试", answer="好的", dimension_id="orientation")
            
            warmup_time = time.time() - start_time
            print(f"[Agent] ✅ 工具预热完成 (耗时 {warmup_time:.2f}秒)\n")
        except Exception as e:
            print(f"[Agent] ⚠️  工具预热失败: {e}\n")
    
    def _create_agent(self) -> AgentExecutor:
        """创建Agent执行器（ReAct模式）"""
        
        # ReAct Agent Prompt模板
        template = """你是一位专业的阿尔茨海默病初筛评估医生的AI助手。

你可以使用以下工具：
{tools}

使用以下格式进行推理：

Question: 需要处理的问题
Thought: 你应该思考要做什么
Action: 要使用的工具，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入（必须是有效的JSON对象）
Observation: 工具的执行结果
... (这个 Thought/Action/Action Input/Observation 可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 最终的回复内容（可以是安慰话语或评估问题）

【参数格式示例】：
- dimension_detection_tool: {{"question": "医生的问题", "answer": "患者的回答"}}
- resistance_detection_tool: {{"question": "医生的问题", "answer": "患者的回答"}}
- query_sentence_generator: {{"dimension": {{"name": "定向力", "id": "orientation"}}, "history": [], "profile": {{}}}}
- knowledge_retrieval: {{"query": "检索查询文本", "top_k": 5}}
- generate_question: {{"dimension_name": "定向力", "knowledge_context": "...", "patient_profile": {{}}, "conversation_history": [], "patient_emotion": "neutral"}}
- conversation_storage: {{"action": "save_turn", "session_id": "s_xxx", "user_input": "...", "generated_question": "..."}}
- answer_evaluation_tool: {{"question": "医生的问题", "answer": "患者的回答", "dimension_id": "orientation", "patient_profile": {{}}}}
- score_recording_tool: {{"session_id": "s_xxx", "dimension_id": "orientation", "quality_level": "good", "cognitive_performance": "正常", "question": "...", "answer": "...", "evaluation_detail": "...", "action": "save"}}

【重要规则】：
1. Action Input 必须是有效的JSON格式
2. 所有字符串必须用双引号
3. 必须传递所有必需参数
4. Final Answer 只返回回复内容本身，不要任何前缀

【关键工作流程 - 必须严格按顺序执行】：

==================== 第1步：情绪检测 ====================
Action: resistance_detection_tool
参数：question=【医生的问题】, answer=【患者的回答】

判断结果：
  ★ 如果 is_resistant = true 且 confidence >= 0.8（检测到抵抗情绪）：
    → Thought: 检测到抵抗情绪，调用安慰工具
    → Action: comfort_response_tool
    → Action Input: {{
        "patient_emotion": "抵抗/拒绝/焦虑",
        "conversation_context": "当前对话内容",
        "dimension_name": "当前维度"
      }}
    → Observation: 得到安慰话语
    → Final Answer: [安慰话语]
    → 流程结束
  
  ★ 如果 is_resistant = false 或 confidence < 0.8（情绪正常）：
    → 继续第2步

==================== 第2步：评估回答 ====================
Action: answer_evaluation_tool
参数：question=【医生的问题】, answer=【患者的回答】, dimension_id, patient_profile
功能：评估回答是否符合问题，更新维度评估

==================== 第3步：记录评分（必须执行）====================
Action: score_recording_tool
参数：session_id, dimension_id, quality_level, cognitive_performance, question, answer, evaluation_detail, action="save"
功能：记录评估结果到数据库

==================== 第4步：生成下一个问题（必须走检索流程）====================
� 关键：以下3个步骤必须全部执行，缺一不可！

4.1) Thought: 生成检索查询语句
     Action: query_sentence_generator
     Action Input: {{
       "dimension": {{
         "name": "当前维度名称",
         "id": "当前维度ID"
       }},
       "history": [历史记录],
       "profile": {{患者画像}}
     }}
     Observation: 得到检索查询query

4.2) Thought: 使用query检索知识库
     Action: knowledge_retrieval
     Action Input: {{"query": "[上一步得到的query]", "top_k": 5}}
     Observation: 得到相关医学知识

4.3) Thought: 使用检索到的知识生成问题
     Action: generate_question
     Action Input: {{
       "dimension_name": "维度名称",
       "knowledge_context": "[检索到的知识]",
       "patient_profile": {{患者画像}},
       "conversation_history": [历史记录],
       "patient_emotion": "情绪类别"
     }}
     Observation: 得到生成的问题

4.4) Thought: 完成，返回生成的问题
     Final Answer: [工具返回的question]

【严格禁止】：
❌ 禁止跳过 score_recording_tool
❌ 禁止跳过 knowledge_retrieval
❌ 禁止手动编造问题，必须使用工具生成

【完整示例 - 情绍正常】：

Thought: 第1步：检测患者情绪
Action: resistance_detection_tool
Action Input: {{"question": "您还记得今天是几月几号吗？", "answer": "10月4号"}}
Observation: {{"is_resistant": false, "category": "none", "confidence": 1.0}}

Thought: 第2步：情绪正常，评估回答质量
Action: answer_evaluation_tool
Action Input: {{"question": "您还记得今天是几月几号吗？", "answer": "10月4号", "dimension_id": "orientation", "patient_profile": {{"age": 70, "education": 6}}}}
Observation: {{"is_correct": true, "quality_level": "good", "cognitive_performance": "正常", "is_complete": true, "need_followup": false}}

Thought: 第3步：记录认知表现（必须执行）
Action: score_recording_tool
Action Input: {{"session_id": "s_20241004", "dimension_id": "orientation", "quality_level": "good", "cognitive_performance": "正常", "question": "您还记得今天是几月几号吗？", "answer": "10月4号", "evaluation_detail": "回答正确完整", "action": "save"}}
Observation: {{"success": true}}

Thought: 第4步：生成检索查询
Action: query_sentence_generator
Action Input: {{"dimension": {{"name": "定向力", "id": "orientation"}}, "history": [], "profile": {{"age": 70, "education_years": 6}}}}
Observation: {{"query": "阿尔茨海默病 定向力 时间评估"}}

Thought: 使用query检索知识库
Action: knowledge_retrieval
Action Input: {{"query": "阿尔茨海默病 定向力 时间评估", "top_k": 5}}
Observation: {{"success": true, "results": [...]}}

Thought: 使用检索到的知识生成问题
Action: generate_question
Action Input: {{"dimension_name": "定向力", "knowledge_context": "...", "patient_profile": {{"age": 70}}, "conversation_history": [], "patient_emotion": "neutral"}}
Observation: {{"success": true, "question": "很好！您知道今天是星期几吗？"}}

Thought: 完成，返回生成的问题
Final Answer: 很好！您知道今天是星期几吗？

【完整示例 - 检测到抵抗情绍】：

Thought: 第1步：检测患者情绍
Action: resistance_detection_tool
Action Input: {{"question": "请您帮我算一下100减7等于多少？", "answer": "我不想算了，这有什么意义呢？"}}
Observation: {{"is_resistant": true, "category": "avoidance", "confidence": 0.85, "rationale": "患者表现出回避和质疑态度"}}

Thought: 检测到明确的抵抗情绪（confidence=0.85>=0.8），暂停评估，直接生成安慰话语
Final Answer: 我完全理解您的感受，这些问题确实可能让人觉得有些累。我们不着急，您想休息一下吗？要不我们先聊聊别的，您今天心情怎么样？

开始！

Question: {input}
Thought: {agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,  # 关闭默认的verbose输出
            max_iterations=15,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )
    
    def process_turn(
        self,
        user_input: str,
        dimension: Dict[str, Any],
        session_id: str,
        patient_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        处理一轮对话
        
        Args:
            user_input: 用户输入/回答
            dimension: 当前评估的维度
            session_id: 会话ID
            patient_profile: 患者画像
            chat_history: 对话历史
        
        Returns:
            Agent的执行结果，包含生成的问题等
        """
        import time
        import json
        
        start_time = time.time()
        
        # 从历史记录中提取医生的最后一个问题
        doctor_question = "请开始评估"
        if chat_history:
            print(f"[DEBUG] chat_history 长度: {len(chat_history)}")
            for i, msg in enumerate(chat_history[-3:]):  # 显示最后3条
                print(f"[DEBUG]   [{i}] role={msg.get('role')}, content={msg.get('content', '')[:50]}...")
            
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant":
                    doctor_question = msg.get("content", "请开始评估")
                    print(f"[DEBUG] 找到上一轮医生问题: {doctor_question[:50]}...")
                    break
        else:
            print(f"[DEBUG] chat_history 为空，使用默认问题")
        
        print(f"\n{'='*60}")
        print(f"[TURN] 🚀 开始新一轮对话处理")
        print(f"[TURN] 📅 维度: {dimension.get('name')} ({dimension.get('id')})")
        print(f"[TURN] 👤 医生: {doctor_question}")
        print(f"[TURN] 🗣️ 患者: {user_input}")
        print(f"{'='*60}\n")
        
        # 构建输入 - 明确标注问题和回答
        input_parts = [
            f"会话ID: {session_id}",
            f"当前评估维度: {dimension.get('name')} ({dimension.get('description')})",
            f"维度ID: {dimension.get('id')}",
            f"",
            f"==================== 本轮对话 ====================",
            f"【医生的问题】（本轮）: {doctor_question}",
            f"【患者的回答】（本轮）: {user_input}",
            f"================================================",
            f"",
            f"⚠️ 重要：调用工具时，question参数必须使用【医生的问题】（本轮），answer参数必须使用【患者的回答】（本轮）！",
            f"",
        ]
        
        if patient_profile:
            input_parts.append(f"【患者信息】: 年龄{patient_profile.get('age')}岁，教育{patient_profile.get('education_years')}年")
        
        # 添加对话历史（如果有）
        if chat_history and len(chat_history) > 2:
            input_parts.append("【对话历史】（仅供参考）：")
            for msg in chat_history[-6:-1]:  # 最近5轮，不包括最后一轮
                role = "医生" if msg.get("role") == "assistant" else "患者"
                input_parts.append(f"  {role}: {msg.get('content', '')}")
            input_parts.append("")
        
        input_parts.append("请严格按照【关键工作流程】的5个步骤执行，不得跳过任何步骤！")
        
        input_text = "\n".join(input_parts)
        
        # 执行Agent
        print(f"[AGENT] ⏳ 正在调用 ReAct Agent 思考链...")
        agent_start = time.time()
        
        result = self.agent.invoke({
            "input": input_text
        })
        
        agent_time = time.time() - agent_start
        total_time = time.time() - start_time
        
        # ========== 检查并补充缺失的工具调用 ==========
        called_tools = []
        has_resistance = False
        
        if 'intermediate_steps' in result:
            for step in result['intermediate_steps']:
                if isinstance(step, tuple) and len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    tool_name = action.tool if hasattr(action, 'tool') else 'unknown'
                    called_tools.append(tool_name)
                    
                    # 检查是否有抵抗情绍
                    if tool_name == 'resistance_detection_tool':
                        try:
                            obs_str = str(observation)
                            if 'is_resistant' in obs_str:
                                eval_result = json.loads(obs_str)
                                if eval_result.get('is_resistant') and eval_result.get('confidence', 0) >= 0.8:
                                    has_resistance = True
                                    print(f"[检查] 检测到抵抗情绍，不需要检索")
                        except:
                            pass
        
        # 如果没有抵抗情绍且没有调用检索工具，强制补充
        if not has_resistance and 'knowledge_retrieval' not in called_tools:
            print(f"[检查] ⚠️ 未调用检索工具，自动补充执行完整检索流程...")
            
            try:
                # 1. 调用query_sentence_generator
                query_tool = self.tools[0]  # QueryTool
                query_input = {
                    "dimension": dimension,
                    "history": chat_history[-6:] if chat_history else [],
                    "profile": patient_profile
                }
                query_result = query_tool._run(**query_input)
                print(f"[补充] ✅ 生成检索查询: {query_result[:100]}")
                
                # 2. 调用knowledge_retrieval
                retrieval_tool = self.tools[1]  # KnowledgeRetrievalTool
                try:
                    query_data = json.loads(query_result) if isinstance(query_result, str) else query_result
                    search_query = query_data.get('query', '')
                except:
                    search_query = query_result
                
                retrieval_result = retrieval_tool._run(query=search_query, top_k=5, use_fusion=False, skip_reranking=True)
                print(f"[补充] ✅ 知识检索完成")
                
                # 3. 调用generate_question
                question_tool = self.tools[4]  # QuestionGenerationTool
                question_input = {
                    "dimension_name": dimension.get('name'),
                    "dimension_description": dimension.get('description', ''),
                    "knowledge_context": retrieval_result[:500] if len(str(retrieval_result)) > 500 else retrieval_result,
                    "patient_age": patient_profile.get('age') if patient_profile else None,
                    "patient_education": patient_profile.get('education_years') if patient_profile else None,
                    "conversation_history": json.dumps(chat_history[-6:]) if chat_history else None,
                    "patient_emotion": "neutral"
                }
                question_result = question_tool._run(**question_input)
                print(f"[补充] ✅ 生成新问题")
                
                # 更新result的output
                try:
                    q_data = json.loads(question_result) if isinstance(question_result, str) else question_result
                    if isinstance(q_data, dict) and 'question' in q_data:
                        result['output'] = q_data['question']
                        print(f"[补充] ✅ 已更新输出为工具生成的问题")
                except:
                    pass
            except Exception as e:
                print(f"[补充] ❌ 补充执行失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 分析工具调用 - 结构化输出
        if 'intermediate_steps' in result:
            steps_count = len(result['intermediate_steps'])
            print(f"\n{'='*80}")
            print(f"[AGENT] 📊 工具调用详情 (共 {steps_count} 步)")
            print(f"{'='*80}\n")
            
            for idx, step in enumerate(result['intermediate_steps'], 1):
                if isinstance(step, tuple) and len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    
                    tool_name = action.tool if hasattr(action, 'tool') else 'unknown'
                    tool_input = action.tool_input if hasattr(action, 'tool_input') else {}
                    thought = action.log if hasattr(action, 'log') else ''
                    
                    # 打印步骤标题
                    print(f"┌─ Step {idx}: 🔧 {tool_name}")
                    print(f"│")
                    
                    # 打印思考过程（如果有）
                    if thought and 'Thought:' in thought:
                        # 提取Thought部分
                        thought_lines = thought.split('\n')
                        for line in thought_lines:
                            if line.strip().startswith('Thought:'):
                                thought_text = line.replace('Thought:', '').strip()
                                if thought_text and len(thought_text) > 5:
                                    # 截断过长的思考
                                    if len(thought_text) > 150:
                                        thought_text = thought_text[:147] + "..."
                                    print(f"│  💭 思考: {thought_text}")
                                    print(f"│")
                                break
                    
                    # 打印输入
                    print(f"│  📥 输入:")
                    if isinstance(tool_input, dict):
                        for key, value in tool_input.items():
                            # 截断过长的值
                            value_str = str(value)
                            if len(value_str) > 100:
                                value_str = value_str[:97] + "..."
                            print(f"│     • {key}: {value_str}")
                    else:
                        print(f"│     {str(tool_input)[:100]}")
                    
                    print(f"│")
                    
                    # 打印输出
                    print(f"│  📤 输出:")
                    observation_str = str(observation)
                    if len(observation_str) > 200:
                        observation_str = observation_str[:197] + "..."
                    # 多行输出时每行缩进
                    for line in observation_str.split('\n'):
                        print(f"│     {line}")
                    
                    print(f"└{'─'*78}\n")
        
        print(f"\n{'='*60}")
        print(f"[TURN] 🏁 处理完成 (耗时: {total_time:.2f}s)")
        print(f"[TURN] 🎯 生成回复: {result.get('output', 'N/A')}")
        print(f"{'='*60}\n")
        
        return result
    
    def process_turn_streaming(self, user_input: str, dimension: Dict[str, Any], session_id: str, 
                              patient_profile: Optional[Dict[str, Any]] = None,
                              chat_history: Optional[List[Dict[str, str]]] = None):
        """流式处理一轮对话，边生成边返回"""
        from concurrent.futures import ThreadPoolExecutor
        import time
        import json
        
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
        
        print(f"\n[STREAMING] 🌊 开始流式处理 - 维度: {dimension.get('name')}")
        
        # 背景任务：并行执行评估和情绪检测
        executor = ThreadPoolExecutor(max_workers=2)
        background_futures = {}
        
        print(f"[STREAMING] 📤 提交背景任务...")
        
        # 背景任务1: 情绪检测
        resistance_tool = self.tools[3]  # ResistanceDetectionTool
        background_futures['resistance'] = executor.submit(
            resistance_tool._run,
            question=doctor_question,
            answer=user_input
        )
        
        # 背景任务2: 答案评估
        evaluation_tool = self.tools[6]  # AnswerEvaluationTool
        background_futures['evaluation'] = executor.submit(
            evaluation_tool._run,
            question=doctor_question,
            answer=user_input,
            dimension_id=dimension.get('id', ''),
            patient_profile=patient_profile
        )
        
        # 立即开始流式生成问题
        prompt = f"""你是老年科医生，正在评估患者的【{dimension.get('name')}】能力。

患者信息：{patient_profile.get('age')}岁，教育{patient_profile.get('education_years')}年

要求：
1. 生成一个自然、温和的问题评估【{dimension.get('name')}】
2. 语气像朋友聊天
3. 简短（20字内）
4. 直接输出问题，不要任何解释

问题："""
        
        print(f"[STREAMING] 🌊 开始流式生成...")
        stream_start = time.time()
        
        # 流式生成
        full_response = ""
        for chunk in self.streaming_llm.stream(prompt):
            token = chunk.content
            full_response += token
            
            # 实时返回每个token
            yield {
                'type': 'token',
                'content': token,
                'full_text': full_response
            }
        
        stream_time = time.time() - stream_start
        print(f"[STREAMING] ✅ 流式生成完成 ({stream_time:.3f}秒)")
        
        # 等待背景任务完成
        print(f"[STREAMING] ⏳ 等待背景任务...")
        
        resistance_result = None
        evaluation_result = None
        
        try:
            resistance_result = background_futures['resistance'].result(timeout=5)
            if isinstance(resistance_result, str):
                resistance_result = json.loads(resistance_result)
        except Exception as e:
            print(f"[STREAMING] ⚠️  情绪检测失败: {e}")
        
        try:
            evaluation_result = background_futures['evaluation'].result(timeout=5)
            if isinstance(evaluation_result, str):
                evaluation_result = json.loads(evaluation_result)
            
            # 提交评分记录（不等待）
            score_tool = self.tools[7]  # ScoreRecordingTool
            executor.submit(
                score_tool._run,
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
        
        # 返回完成标记
        yield {
            'type': 'done',
            'content': full_response,
            'metadata': {
                'total_time': total_time,
                'stream_time': stream_time,
                'dimension': dimension.get('name'),
                'resistance': resistance_result,
                'evaluation': evaluation_result
            }
        }
        
        print(f"[STREAMING] 🏁 总耗时: {total_time:.3f}秒\n")
    
    def get_tools_info(self) -> List[Dict[str, str]]:
        """获取所有工具的信息"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self.tools
        ]

