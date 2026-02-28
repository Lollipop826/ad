"""
阿尔茨海默病初筛Agent - Function Calling 版本（高性能）

使用 Function Calling 代替 ReAct，减少 LLM 调用次数，提升速度
核心优化：
1. 一次性决定所有工具调用（而不是逐个思考）
2. 支持并行工具调用
3. 减少思考链开销
"""

from typing import List, Dict, Any, Optional, Union, Generator
import os
import json
import time
import re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.tools.agent_tools import (
    QueryTool,
    KnowledgeRetrievalTool,
    DimensionDetectionTool,
    ResistanceDetectionTool,
    QuestionGenerationTool,
    ConversationStorageTool,
    AnswerEvaluationTool,
    ScoreRecordingTool,
    MMSEScoringTool,
    ComfortResponseTool,
    ImageDisplayTool,
    StandardQuestionTool,
)
from src.domain.dimensions import MMSE_DIMENSIONS


class ADScreeningAgentFunctionCalling:
    # 🔥 任务配置：添加 min_turns（最少对话轮数）
    TASK_CONFIG: Dict[str, Dict[str, Any]] = {
        # 缓冲任务：至少聊2轮
        "persona_collect_1": {"dimension_id": None, "max_points": 0, "type": "buffer", "min_turns": 1},
        "persona_collect_2": {"dimension_id": None, "max_points": 0, "type": "buffer", "min_turns": 1},
        "buffer_chat": {"dimension_id": None, "max_points": 0, "type": "buffer", "min_turns": 1},
        "buffer_consent": {"dimension_id": None, "max_points": 0, "type": "buffer", "min_turns": 1},
        # 定向力任务：1轮即可
        "orientation_time_weekday": {"dimension_id": "orientation", "max_points": 1, "type": "task", "min_turns": 1},
        "orientation_time_date_month_season": {"dimension_id": "orientation", "max_points": 2, "type": "task", "min_turns": 1},
        "orientation_place_city_district": {"dimension_id": "orientation", "max_points": 2, "type": "task", "min_turns": 1},
        # 记忆任务
        "registration_3words": {"dimension_id": "registration", "max_points": 3, "type": "task", "min_turns": 1},
        # 注意力计算（连续减5次）
        "attention_calc_life_math": {"dimension_id": "attention_calculation", "max_points": 5, "type": "task", "min_turns": 5},
        # 语言任务
        "language_naming_watch": {"dimension_id": "language", "max_points": 1, "type": "task", "min_turns": 1},
        "language_naming_pencil": {"dimension_id": "language", "max_points": 1, "type": "task", "min_turns": 1},
        "language_repetition_sentence": {"dimension_id": "language", "max_points": 1, "type": "task", "min_turns": 1},
        "language_reading_close_eyes": {"dimension_id": "language", "max_points": 1, "type": "task", "min_turns": 1},
        "language_3step_action": {"dimension_id": "language", "max_points": 3, "type": "task", "min_turns": 1},
        # 延迟回忆
        "recall_3words": {"dimension_id": "recall", "max_points": 3, "type": "task", "min_turns": 1},
    }

    REQUIRED_TASKS: List[str] = [
        "persona_collect_1",
        "persona_collect_2",
        "orientation_time_weekday",
        "orientation_time_date_month_season",
        "orientation_place_city_district",
        "registration_3words",
        "attention_calc_life_math",
        "language_naming_watch",
        "language_naming_pencil",
        "language_repetition_sentence",
        "language_reading_close_eyes",
        "language_3step_action",
        "recall_3words",
    ]

    BUFFER_TASKS: set = {"persona_collect_1", "persona_collect_2", "buffer_chat", "buffer_consent", "buffer_answer_question"}

    """
    阿尔茨海默病初筛Agent - Function Calling 版本
    
    直接按预定义流程调用工具，避免 ReAct 的多次 LLM 思考
    """
    
    def __init__(self, use_local: bool = True):
        self.use_local = use_local
        self.session_id = None
        self.current_dimension = MMSE_DIMENSIONS[0]  # 从定向力开始
        self.dimension_index = 0
        
        
        # 闲聊模式管理
        self.is_in_comfort_mode = False  # 是否在闲聊模式
        self.comfort_turn_count = 0      # 连续闲聊轮次（不是抵抗次数）
        
        
        # ⭐ 特殊维度数据存储（用于 recall 时获取 registration 的记忆词）
        self.session_data = {
            'memory_words': None,  # registration 阶段的记忆词
            'calculation_config': None,  # 计算配置
        }

        self._dimension_map = {d.get('id'): d for d in MMSE_DIMENSIONS}
        self._last_task_id: Optional[str] = None
        self._task_done: set = set()
        self._task_attempts: Dict[str, int] = {}
        self._task_best: Dict[str, Dict[str, Any]] = {}
        self._registration_ts: Optional[float] = None
        self._turn_counter: int = 0
        self._session_start_ts: float = time.time()
        self._asked_to_continue: bool = False  # 是否已问过过渡语
        self._asked_questions: List[str] = []  # 已问过的问题列表（用于去重）
        self._used_chat_topics: List[str] = []  # 🔥 已聊过的闲聊话题（用于安慰去重）
        self._used_bridge_topics: List[str] = []  # 🔥 已用过的过渡话题（用于评估模式过渡去重）
        self._last_bridge_hint: Optional[str] = None  # 🔥 当前任务的过渡提示（传给QuestionGenTool）
        self._consecutive_free_chat: int = 0  # 🔥 连续自由闲聊轮数（超过3轮需引导回任务）
        self._pending_consent_task_id: Optional[str] = None
        self._consent_granted_task_id: Optional[str] = None
        self._task_cooldown_until: Dict[str, int] = {}
        self._last_generated_question: str = "请开始评估"  # 🔥 保存上一次生成的问题
        
        # 🔥 异步问题分类相关
        self._pending_classification_task = None  # 后台分类任务
        self._last_classification_result: Optional[str] = None  # 上一轮分类结果
        self._available_candidates: List[str] = []  # 当前可用的候选任务
        
        # 初始化工具
        print("[AgentFC] 🚀 初始化 Function Calling Agent...")
        self._init_tools()
        
        # ⭐ 暴露 llm 属性（用于全双工打断检测的语义判断）
        if use_local:
            from src.llm.model_pool import get_pooled_llm
            self.llm = get_pooled_llm(pool_key='small_classify')  # 使用快速小模型
            print("[AgentFC] ✅ 已配置打断检测 LLM (small_classify)")
        else:
            # API模式，可以使用工具的LLM
            self.llm = None
        
        print("[AgentFC] ✅ Agent 初始化完成\n")
    
    def _init_tools(self):
        """初始化所有工具"""
        print("[AgentFC] 📦 初始化工具...")
        
        # ⭐ 所有工具使用本地模式，避免网络延迟
        print(f"[AgentFC] 💡 工具模式: use_local={self.use_local}")
        # 🔥 这些工具强制使用 API（可与本地模型并行执行）
        self.resistance_tool = ResistanceDetectionTool(use_local=False)
        self.comfort_tool = ComfortResponseTool(use_local=False)
        self.question_tool = QuestionGenerationTool(use_local=False)
        print("[AgentFC] 🌐 ResistanceTool/ComfortTool/QuestionGenTool 使用 API 模式（支持并行）")
        
        # 这些工具跟随 use_local 配置（需要本地 GPU）
        self.eval_tool = AnswerEvaluationTool(use_local=self.use_local)
        self.query_tool = QueryTool(use_local=self.use_local)
        
        # 其他工具
        self.score_tool = ScoreRecordingTool()  # 定性评估记录
        self.mmse_tool = MMSEScoringTool()  # ⭐ MMSE标准评分（30分制）
        self.retrieval_tool = KnowledgeRetrievalTool()
        self.storage_tool = ConversationStorageTool()
        self.image_tool = ImageDisplayTool()  # 📋 图片展示工具
        self.standard_question_tool = StandardQuestionTool(use_local=self.use_local)  # 📝 特殊维度问题工具
        
        # ⭐ 追踪连续错误次数（用于 attention_calculation 提前终止）
        self.consecutive_failures = 0
        
        # ⭐ 连续减法状态跟踪（100-7 的当前值）
        self._calculation_current_value = 100  # 初始值
        self._calculation_step = 7  # 每次减去的数
        
        print("[AgentFC] ✅ 工具初始化完成")
        
        # 预热工具
        if self.use_local:
            self._warmup_tools()
    
    def _warmup_tools(self):
        """预热关键工具"""
        print("[AgentFC] 🔥 预热工具...")
        start_time = time.time()
        
        try:
            # 预热 Resistance + Eval（最常用）
            _ = self.resistance_tool._run(question="测试", answer="好的")
            _ = self.eval_tool._run(question="测试", answer="好的", task_id="orientation_time_weekday")
            
            warmup_time = time.time() - start_time
            print(f"[AgentFC] ✅ 工具预热完成 (耗时 {warmup_time:.2f}秒)")
        except Exception as e:
            print(f"[AgentFC] ⚠️  工具预热失败: {e}")
    
    def process_turn(
        self,
        user_input: str,
        dimension: Optional[Dict[str, Any]] = None,  # 改为可选参数
        session_id: str = None,
        patient_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        current_emotion: str = 'neutral'  # 新增参数：当前语音情绪
    ):
        """
        处理一轮对话（Function Calling 版本 - 快速）
        
        参数与 screening_agent.py 保持一致，方便切换
        
        固定流程：
        1. 抵抗检测 (并行)
        2. 回答评估 (并行)
        3. 记录评分
        4. 生成问题（检索 + 生成）
        5. 存储对话
        """
        start_time = time.time()
        
        # 处理默认的session_id
        if session_id is None:
            import uuid
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        self.session_id = session_id

        # 🔍 调试：打印当前同意状态
        print(f"[AgentFC] 🔍 DEBUG session_id检查: active={getattr(self, '_active_session_id', None)}, new={session_id}")
        print(f"[AgentFC] 🔍 DEBUG pending_consent={self._pending_consent_task_id}, granted={self._consent_granted_task_id}")

        if getattr(self, "_active_session_id", None) != session_id:
            print(f"[AgentFC] ⚠️ session_id 变化，重置任务池！")
            self._reset_task_pool(session_id)
        
        # 处理默认参数
        if patient_profile is None:
            patient_profile = {}
        if chat_history is None:
            chat_history = []

        self.session_data['chat_history'] = chat_history
        
        # 设置当前维度
        if dimension is not None:
            # 只有显式传入dimension时才覆盖
            self.current_dimension = dimension
        
        dimension_id = self.current_dimension.get('id', 'orientation')
        dimension_name = self.current_dimension.get('name', '定向力')
        
        # 从历史中提取医生的最后一个问题
        # 🔥 优先使用 Agent 保存的上一次生成的问题（更可靠）
        doctor_question = self._last_generated_question or "请开始评估"
        # 备用：从 chat_history 获取
        if doctor_question == "请开始评估" and chat_history:
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant":
                    doctor_question = msg.get("content", "请开始评估")
                    break
        
        user_answer = user_input  # 统一变量名

        self._turn_counter += 1
        
        from datetime import datetime
        _now = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n{'='*60}")
        print(f"⏰ [{_now}] [AgentFC] 🗣️  收到用户回复 (Turn {self._turn_counter})")
        print(f"  ├── 👤 医生: {doctor_question[:50]}...")
        print(f"  ├── 🗣️ 患者: {user_answer}")
        print(f"  ├── 📅 维度: {dimension_name} ({dimension_id})")
        print(f"  └── 😊 情绪: {current_emotion}")
        print(f"{'='*60}\n")
        
        image_display_command = None  # Initialize at start of function to prevent UnboundLocalError
        
        try:
            # 🔥 关键修复：如果上一轮是认知任务，必须退出闲聊模式来正确评分
            last_task_id = self._last_task_id
            if self.is_in_comfort_mode and last_task_id and last_task_id not in self.BUFFER_TASKS:
                print(f"[AgentFC] 🔄 上轮执行了认知任务 [{last_task_id}]，自动退出闲聊模式以进行评分")
                self.is_in_comfort_mode = False
                self.comfort_turn_count = 0
        
            # ⭐ 根据模式决定处理流程
            if self.is_in_comfort_mode:
                # ==================== 闲聊模式：不检测抵抗，只管理闲聊轮次 ====================
                print(f"[AgentFC] 💤 当前处于 [闲聊模式] (第 {self.comfort_turn_count + 1}/3 轮)")
                self.comfort_turn_count += 1
                
                # 🔔 检测用户是否要结束对话（goodbye检测）
                goodbye_keywords = ['结束', '再见', '拜拜', '不聊了', '就这样', '到这儿', '先这样', '回头聊']
                is_goodbye = any(kw in user_answer for kw in goodbye_keywords)
                
                if is_goodbye:  #关键词监测到执行
                    # 🔍 检查 MMSE 是否已完成所有维度
                    mmse_complete = self._check_mmse_complete(session_id)
                    
                    if mmse_complete:
                        # ✅ MMSE 已完成，可以正常告别
                        print(f"[AgentFC] 👋 检测到用户想结束，MMSE已完成，正常告别")
                        farewell = f"{patient_profile.get('name', '')}，好嘞，那今天就到这儿，您好好休息。想聊的时候随时叫我！"

                        
                        return {
                            'output': farewell,
                            'response': farewell,
                            'is_comfort_mode': True,
                            'is_goodbye': True,
                            'total_time': time.time() - start_time
                        }
                    else:
                        # ⚠️ MMSE 未完成，不能真正结束，用 LLM 生成自然的过渡
                        print(f"[AgentFC] ⚠️ 用户想结束但MMSE未完成，温柔地继续对话")
                        soft_continue = self._generate_soft_continuation(
                            patient_profile, user_answer, chat_history
                        )

                        
                        return {
                            'output': soft_continue,
                            'response': soft_continue,
                            'is_comfort_mode': True,
                            'is_goodbye': False,
                            'total_time': time.time() - start_time
                        }
                
                # 判断是否该尝试引导回评估：必须同时满足轮数足够 AND 情绪积极
                ask_to_continue = False
                positive_emotions = ['happy', 'excited', 'joy', 'positive', 'neutral']
                is_positive = current_emotion in positive_emotions
                
                if self.comfort_turn_count >= 3 and is_positive:
                    print(f"[AgentFC] 💡 已闲聊 {self.comfort_turn_count} 轮且情绪积极({current_emotion})，尝试引导回评估")
                    ask_to_continue = True
                elif self.comfort_turn_count >= 3:
                    print(f"[AgentFC] 💬 已闲聊 {self.comfort_turn_count} 轮，但情绪不佳({current_emotion})，继续闲聊")
                else:
                    print(f"[AgentFC] 💬 闲聊第 {self.comfort_turn_count} 轮，继续保持闲聊")
                
                # 🔥 修复：满足退出条件时，恢复被打断的任务
                if ask_to_continue:
                    print(f"[AgentFC] 💡 满足退出条件，自动恢复评估模式")
                    self.is_in_comfort_mode = False
                    self.comfort_turn_count = 0
                    
                    # 🔥 检查被打断的任务是否已完成
                    interrupted_task = getattr(self, '_comfort_interrupted_task_id', None)
                    if interrupted_task and interrupted_task not in self._task_done:
                        # 继续被打断的任务
                        print(f"[AgentFC] 🔄 恢复被打断的任务: {interrupted_task}")
                        self._comfort_interrupted_task_id = None  # 清除记录
                        
                        # 获取任务配置
                        task_cfg = self.TASK_CONFIG.get(interrupted_task, {})
                        task_dim_id = task_cfg.get('dimension_id', dimension_id)
                        task_dim_name = self._dimension_map.get(task_dim_id, {}).get('name', dimension_name)
                        
                        return self._generate_assessment_question(
                            task_dim_name, patient_profile, chat_history, start_time,
                            task_id=interrupted_task  # 🔥 传入任务ID
                        )
                    else:
                        # 被打断的任务已完成或不存在，选新任务
                        print(f"[AgentFC] 📋 被打断的任务已完成，选择新任务")
                        self._comfort_interrupted_task_id = None
                        next_task_id = self._select_next_task()
                        
                        if next_task_id is None:
                            # 所有任务完成，生成完成消息
                            summary_json = self.mmse_tool._run(session_id=session_id, dimension_id="", action="summary")
                            summary = json.loads(summary_json)
                            scaled_score = summary.get('scaled_total_score', summary.get('total_score', 0))
                            risk_assessment = self._calculate_alzheimers_risk(int(scaled_score))
                            completion_message = self._generate_completion_message(int(scaled_score), risk_assessment, patient_profile)

                            
                            return {
                                'output': completion_message,
                                'response': completion_message,
                                'assessment_complete': True,
                                'total_score': summary.get('total_score', 0),
                                'scaled_score': scaled_score,
                                'total_time': time.time() - start_time
                            }
                        
                        # 获取新任务的维度
                        task_cfg = self.TASK_CONFIG.get(next_task_id, {})
                        task_dim_id = task_cfg.get('dimension_id', dimension_id)
                        task_dim_name = self._dimension_map.get(task_dim_id, {}).get('name', dimension_name)
                        
                        return self._generate_assessment_question(
                            task_dim_name, patient_profile, chat_history, start_time,
                            task_id=next_task_id  # 🔥 传入任务ID
                        )
                
                # 使用原有comfort工具生成闲聊回复
                # 映射情绪到抵抗类别
                emotion_to_category = {
                    'sad': 'fatigue',
                    'angry': 'hostility',
                    'fear': 'avoidance',
                    'neutral': 'normal',
                    'happy': 'normal',
                    'excited': 'normal',
                }
                category = emotion_to_category.get(current_emotion, 'normal')
                comfort_result = self.comfort_tool._run(
                    resistance_category=category,
                    patient_answer=user_answer,
                    patient_name=patient_profile.get('name'),
                    patient_age=patient_profile.get('age'),
                    patient_gender=patient_profile.get('gender'),  # 🔥 新增：性别参数
                    used_topics=self._used_chat_topics,  # 🔥 传入已聊话题
                    chat_history=chat_history,  # 🔥 新增：传入聊天记录
                    use_template=False,
                )
                
                # 更新已聊话题
                try:
                    comfort_data = json.loads(comfort_result)
                    if topic := comfort_data.get('selected_topic'):
                        self._used_chat_topics.append(topic)
                        print(f"[AgentFC] 📌 记录新话题: {topic} (总计: {len(self._used_chat_topics)})")
                except:
                    pass
                comfort_data = json.loads(comfort_result)
                comfort_message = comfort_data.get('comfort_message', '您说得对，咱们继续聊聊。')
                
                # 🔥 记录闲聊问题到 _asked_questions（防止退出闲聊后重复）
                if comfort_message and len(comfort_message) > 5:
                    self._asked_questions.append(comfort_message[:80])
                    if len(self._asked_questions) > 20:
                        self._asked_questions = self._asked_questions[-20:]
                
                return {
                    'output': comfort_message,
                    'response': comfort_message,
                    'is_comfort_mode': True,
                    'comfort_turn': self.comfort_turn_count,
                    'total_time': time.time() - start_time
                }
            
            else:
                # ==================== 评估模式：正常流程 ====================
                step1_start = time.time()
                
                # 🔥 标记本轮是否运行了 Step 1（用于后续判断话题是否新鲜）
                self._current_turn_topic_set = False
                
                # 🔥 获取后台分类结果（仅供参考，不覆盖 last_task_id）
                classification_result = self.get_classification_result(timeout=0.5)
                
                # 🆕 任务池：获取上一个任务的信息（这是实际执行的任务）
                last_task_id = self._last_task_id
                
                # 🔥 修复：分类结果只在特定场景使用（如评估任务时确认问题类型）
                # 不应该覆盖 last_task_id，因为 last_task_id 代表"实际执行的任务"
                # 而 classification_result 代表"AI上一轮提问的类型"
                if classification_result and classification_result != "buffer_chat":
                    if last_task_id != classification_result:
                        print(f"[AgentFC] 📋 分类参考: {classification_result} (实际任务: {last_task_id})")
                
                last_task_is_buffer = (last_task_id in self.BUFFER_TASKS) if last_task_id else True
                last_task_cfg = self.TASK_CONFIG.get(last_task_id, {}) if last_task_id else {}
                task_dimension_id = last_task_cfg.get('dimension_id') or dimension_id
                
                # 如果上一轮是缓冲任务，跳过评分但仍检测抵抗
                if last_task_is_buffer:
                    print(f"[AgentFC] 💬 上轮是缓冲任务({last_task_id})，跳过评分但检测抵抗")
                    
                    # 🔥 性能优化：buffer 路径也并行执行（ResistanceTool + TaskSelection + Retrieval 预取）
                    # 原来是串行：ResistanceTool(~4s) → Phase2 TaskSelection(~2s) = 6s
                    # 现在并行：max(ResistanceTool, TaskSelection, Retrieval) ≈ 2s
                    print("[AgentFC] ⚡ buffer路径: 并行执行抵抗检测(API)+任务选择(API)+知识预取...")
                    
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        # 1. 抵抗检测
                        future_resist = executor.submit(self._call_resistance_detection, doctor_question, user_answer)
                        
                        # 2. 🔥 提前启动任务选择（不用等 ResistanceTool 完成）
                        future_task_select = None
                        if not self._pending_consent_task_id:
                            future_task_select = executor.submit(self._select_next_task)
                        
                        # 3. 🔥 预取知识检索（Phase 3 可能需要）
                        _prefetch_query = self._call_query_generation(dimension_name, chat_history)
                        future_retrieval = executor.submit(self._call_knowledge_retrieval, _prefetch_query)
                        print(f"[AgentFC] ⚡ buffer路径: 预取知识检索已启动 (维度: {dimension_name})")
                        
                        # 获取抵抗检测结果
                        resistance_result = future_resist.result()
                    
                    # 🔥 处理请求重复：直接重复上一个问题
                    if resistance_result.get('category') == 'repeat_request':
                        print(f"[AgentFC] 🔁 检测到请求重复，直接重复上一个问题")
                        repeat_prefixes = [
                            "好的，我再说一遍：",
                            "没关系，我再问一次：",
                            "好啊，我重复一下：",
                            "您说的对，我再说一次：",
                        ]
                        import random
                        prefix = random.choice(repeat_prefixes)
                        repeated_question = f"{prefix}{doctor_question}"
                        
                        return {
                            'output': repeated_question,
                            'response': repeated_question,
                            'is_repeat': True,
                            'dimension': dimension_name,
                            'dimension_id': dimension_id,
                            'total_time': time.time() - start_time
                        }
                    
                    # 🔥 修复：buffer 任务检测到抵抗也需要安慰 -> 直接中断返回
                    if resistance_result.get('is_resistant'):
                        category = resistance_result.get('category', 'unknown')
                        print(f"[AgentFC] 🛑 缓冲任务中检测到抵抗: {category} -> 进入安抚模式")
                        self.is_in_comfort_mode = True
                        self.comfort_turn_count = 0
                        
                        # 调用 comfort 工具生成安抚话术
                        comfort_result = self.comfort_tool._run(
                            resistance_category=category,
                            patient_answer=user_answer,
                            patient_name=patient_profile.get('name'),
                            patient_age=patient_profile.get('age'),
                            patient_gender=patient_profile.get('gender'),
                            used_topics=self._used_chat_topics,
                            chat_history=chat_history,
                            use_template=False,
                        )
                        
                        # 解析结果
                        comfort_data = json.loads(comfort_result)
                        if topic := comfort_data.get('selected_topic'):
                            self._used_chat_topics.append(topic)
                            self._last_bridge_hint = topic
                        
                        comfort_message = comfort_data.get('comfort_message', '您说得对，咱们继续聊聊。')
                        
                        return {
                            'output': comfort_message,
                            'response': comfort_message,
                            'has_resistance': True,
                            'resistance_category': category,
                            'is_comfort_mode': True,
                                'dimension': dimension_name,
                            'dimension_id': 'buffer',
                            'total_time': time.time() - start_time
                        }

                    # 🔥 无抵抗：收集预取的知识检索结果（Phase 3 可直接用）
                    try:
                        self._prefetched_retrieval = future_retrieval.result(timeout=5)
                        print(f"[AgentFC] ⚡ buffer路径: 预取知识检索完成")
                    except Exception as e:
                        self._prefetched_retrieval = None
                        print(f"[AgentFC] ⚠️ buffer路径: 预取检索失败: {e}")

                    # 🔥 收集任务选择结果（关键！Phase 2 直接用，省掉 ~2s LLM 调用）
                    if future_task_select:
                        self._precomputed_next_task = future_task_select.result()
                        topic_info = getattr(self, '_last_bridge_hint', 'N/A')
                        print(f"╔════════════════════════════════════════════════════════════╗")
                        print(f"║ ⏰ [{time.strftime('%H:%M:%S')}] 🔮 Phase 1 (buffer): Task Pre-Selection              ║")
                        print(f"╠════════════════════════════════════════════════════════════╣")
                        print(f"  ✨ 预选任务: {self._precomputed_next_task}")
                        if topic_info:
                            print(f"  🌉 话题过渡: {topic_info}")
                        print(f"╚════════════════════════════════════════════════════════════╝")
                    
                    eval_result = {
                        'is_correct': True, 'quality_level': 'good', 'cognitive_performance': '正常',
                        'is_complete': True, 'evaluation_detail': '缓冲闲聊轮不计分',
                        'need_followup': False, 'confidence': 1.0
                    }
                else:
                    # 获取期望答案（用于精准评估）
                    expected_answer = self._get_expected_answer_for_task(last_task_id, patient_profile)
                    
                    # 🔥 优化：先做抵抗检测（~40ms），如果有抵抗就跳过回答评估（~4秒）
                    simple_answers = ["好", "好的", "嗯", "对", "是", "是的", "知道", "明白", "行", "可以"]
                    is_simple_answer = user_answer.strip() in simple_answers or len(user_answer.strip()) <= 3
                    
                    if is_simple_answer:
                        # 简单回答：只做回答评估，跳过抵抗检测
                        print(f"[AgentFC] ⚡ 快速路径: 简单回答'{user_answer[:10]}'，跳过抵抗检测")
                        resistance_result = {'is_resistant': False, 'confidence': 1.0, 'category': 'none'}
                        self._prefetched_retrieval = None  # 快速路径无预取
                        eval_result = self._call_answer_evaluation(
                            doctor_question, user_answer, last_task_id, patient_profile, expected_answer
                        )
                    else:
                        # 🔥 优化：ResistanceTool (API) 与 AnswerEval (Local) 与 TaskSelection (API) 并行执行
                        print("[AgentFC] ⚡ 阶段一: 并行执行抵抗检测(API)+回答评估(Local)+任务选择(API)...")
                        
                        # 🔥 预计算：本轮结束后任务是否会完成？（用于决定是否需要启动任务选择）
                        # 轮数+1（用户回答了这一轮）
                        predicted_turns = self._task_turns.get(last_task_id, 0) + 1
                        min_turns = self.TASK_CONFIG.get(last_task_id, {}).get('min_turns', 1)
                        task_will_complete = (predicted_turns >= min_turns)
                        
                        # 🔥 预更新 _task_done（供任务选择使用，之后会正式更新）
                        if task_will_complete and last_task_id:
                            self._task_done.add(last_task_id)
                            print(f"[AgentFC] 🔮 预测任务完成: {last_task_id} (轮数: {predicted_turns}/{min_turns})")
                        
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            # 提交任务
                            future_resist = executor.submit(self._call_resistance_detection, doctor_question, user_answer)
                            future_eval = executor.submit(
                                self._call_answer_evaluation, doctor_question, user_answer,
                                last_task_id, patient_profile, expected_answer
                            )
                            
                            # 🔥 如果任务会完成，并行启动任务选择
                            future_task_select = None
                            if task_will_complete and not self._pending_consent_task_id:
                                future_task_select = executor.submit(self._select_next_task)
                            
                            # 🔥 性能优化：预取知识检索（Phase 3 可能需要）
                            # query_generation 已规则化(~0ms)，所以可以立即启动检索
                            _prefetch_query = self._call_query_generation(dimension_name, chat_history)
                            future_retrieval = executor.submit(self._call_knowledge_retrieval, _prefetch_query)
                            print(f"[AgentFC] ⚡ 预取知识检索已启动 (维度: {dimension_name})")
                            
                            # 获取抵抗检测结果
                            resistance_result = future_resist.result()
                            
                            # 🔥 处理请求重复：直接重复上一个问题，跳过所有其他处理
                            if resistance_result.get('category') == 'repeat_request':
                                print(f"[AgentFC] 🔁 检测到请求重复，直接重复上一个问题")
                                # 生成自然的重复前缀
                                repeat_prefixes = [
                                    "好的，我再说一遍：",
                                    "没关系，我再问一次：",
                                    "好啊，我重复一下：",
                                    "您说的对，我再说一次：",
                                ]
                                import random
                                prefix = random.choice(repeat_prefixes)
                                repeated_question = f"{prefix}{doctor_question}"
                                
                                return {
                                    'output': repeated_question,
                                    'response': repeated_question,
                                    'is_repeat': True,
                                    'dimension': dimension_name,
                                    'dimension_id': dimension_id,
                                    'total_time': time.time() - start_time
                                }
                            
                            # 🔥 检测到抵抗 -> 直接进入闲聊模式（工具内部已做LLM仲裁，结果可信）
                            if resistance_result.get('is_resistant'):
                                category = resistance_result.get('category', 'unknown')
                                print(f"[AgentFC] 🛑 触发抵抗中断: {category} -> 进入安抚模式")
                                
                                # 进入闲聊模式
                                self.is_in_comfort_mode = True
                                self.comfort_turn_count = 0
                                # 调用 comfort 工具生成安抚话术
                                comfort_result = self.comfort_tool._run(
                                    resistance_category=category,
                                    patient_answer=user_answer,
                                    patient_name=patient_profile.get('name'),
                                    patient_age=patient_profile.get('age'),
                                    patient_gender=patient_profile.get('gender'),
                                    used_topics=self._used_chat_topics,
                                    chat_history=chat_history,  # 🔥 新增：传入聊天记录
                                    use_template=False,
                                )
                                
                                # 解析结果并记录话题
                                comfort_data = json.loads(comfort_result)
                                if topic := comfort_data.get('selected_topic'):
                                    self._used_chat_topics.append(topic)
                                    print(f"[AgentFC] 📌 记录新话题: {topic}")
                                
                                comfort_message = comfort_data.get('comfort_message', '您说得对，咱们继续聊聊。')
                                print(f"[AgentFC] ✅ 已切换至闲聊模式 (耗时 {time.time() - step1_start:.2f}秒)\n")
                                
                                return {
                                    'output': comfort_message,
                                    'response': comfort_message,
                                    'has_resistance': True,
                                    'resistance_category': category,
                                    'is_comfort_mode': True,
                                    'dimension': dimension_name,
                                    'dimension_id': dimension_id,
                                    'total_time': time.time() - start_time
                                }
                            else:
                                # 无抵抗或轻微抵抗，获取评估结果
                                eval_result = future_eval.result()
                                
                                # 🔥 获取预取的知识检索结果
                                try:
                                    self._prefetched_retrieval = future_retrieval.result(timeout=5)
                                    print(f"[AgentFC] ⚡ 预取知识检索完成")
                                except Exception as e:
                                    self._prefetched_retrieval = None
                                    print(f"[AgentFC] ⚠️ 预取检索失败: {e}")
                                
                                # 🔥 如果启动了任务选择，获取结果
                                if future_task_select:
                                    self._precomputed_next_task = future_task_select.result()
                                    
                                    # 获取话题提示
                                    topic_info = getattr(self, '_last_bridge_hint', 'N/A')
                                    
                                    #  样式化输出
                                    print(f"╔════════════════════════════════════════════════════════════╗")
                                    print(f"║ ⏰ [{time.strftime('%H:%M:%S')}] 🔮 Phase 1: Task Selection                        ║")
                                    print(f"╠════════════════════════════════════════════════════════════╣")
                                    print(f"  ✨ 预选任务: {self._precomputed_next_task}")
                                    if topic_info:
                                        print(f"  🌉 话题过渡: {topic_info}")
                                    print(f"╚════════════════════════════════════════════════════════════╝")
                
                step1_time = time.time() - step1_start
                print(f"[AgentFC] ✅ 阶段一完成 (耗时 {step1_time:.2f}秒)\n")
            
            # ==================== 步骤2: 记录评分 + 任务池更新 ====================
            print("[AgentFC] 📊 阶段二: 评分与进度管理...")
            step2_start = time.time()
            
            mmse_result = {'total_score': 0}
            
            # 🆕 任务池：记录评分 + 检查是否达到最小轮数
            if last_task_id:
                # 增加该任务的轮数计数
                if not hasattr(self, '_task_turns'):
                    self._task_turns = {}
                self._task_turns[last_task_id] = self._task_turns.get(last_task_id, 0) + 1
                current_turns = self._task_turns[last_task_id]
                
                # 获取任务配置
                task_cfg = self.TASK_CONFIG.get(last_task_id, {})
                min_turns = task_cfg.get('min_turns', 1)
                
                if last_task_id not in self.BUFFER_TASKS:
                    # 非缓冲任务：记录评分
                    task_dim_id = task_cfg.get('dimension_id', dimension_id)
                    task_max_points = task_cfg.get('max_points')
                    mmse_result = self._call_score_recording(
                        session_id, task_dim_id, eval_result, doctor_question, user_answer, task_max_points
                    )
                
                # 🔥 检查是否达到最小轮数
                if current_turns >= min_turns:
                    self._task_done.add(last_task_id)
                    if last_task_id in self.BUFFER_TASKS:
                        print(f"[AgentFC] (buffer) 缓冲任务完成: {last_task_id} (轮数: {current_turns}/{min_turns})")
                    else:
                        print(f"[AgentFC] ✅ 任务完成: {last_task_id} (轮数: {current_turns}/{min_turns})")
                    
                    # 如果是 registration，记录时间戳
                    if last_task_id == "registration_3words":
                        self._registration_ts = time.time()
                        print(f"[AgentFC] ⏰ 记录 registration 时间戳，recall 需等待2分钟")
                else:
                    print(f"[AgentFC] 🔄 任务继续: {last_task_id} (轮数: {current_turns}/{min_turns})")
            
            # 🆕 检测用户是否在提问，如果是则先回答问题
            user_is_asking = self._is_user_asking_question(user_answer)
            if user_is_asking and last_task_id not in {"buffer_consent"}:
                print(f"[AgentFC] 💬 检测到用户提问，先回答用户问题")
            
            # 🆕 任务池：选择下一个任务
            if self._pending_consent_task_id:
                pending = self._pending_consent_task_id
                print(f"[AgentFC] 🔔 等待同意: pending={pending}, 用户回答='{user_answer}'")
                user_agreed = self._check_user_willing_to_continue(user_answer)
                print(f"[AgentFC] 🔔 用户同意判断: {user_agreed}")
                if user_agreed:
                    self._consent_granted_task_id = pending
                    self._pending_consent_task_id = None
                    next_task_id = pending
                    print(f"[AgentFC] ✅ 用户同意！设置 consent_granted={pending}")
                else:
                    self._task_cooldown_until[pending] = self._turn_counter + 3
                    self._pending_consent_task_id = None
                    next_task_id = "buffer_chat"
                    print(f"[AgentFC] ❌ 用户拒绝，进入闲聊")
            elif user_is_asking:
                # 用户在提问，先回答问题再继续评估
                next_task_id = "buffer_answer_question"
                # 🔥 清空预计算的任务，避免下一轮捡到"遗产"
                if hasattr(self, '_precomputed_next_task'):
                    self._precomputed_next_task = None
                print(f"[AgentFC] 🔍 用户提问检测: '{user_answer[:20]}...' → 是提问")
            else:
                # 🔥 如果当前任务还没完成，继续执行它
                if last_task_id and last_task_id not in self._task_done:
                    next_task_id = last_task_id
                    print(f"[AgentFC] 🔄 任务继续: {last_task_id} (轮数: {self._task_turns.get(last_task_id, 0)}/{self.TASK_CONFIG.get(last_task_id, {}).get('min_turns', 1)})")
                else:
                    # 🔥 优先使用预计算的结果（阶段一并行计算的）
                    if hasattr(self, '_precomputed_next_task') and self._precomputed_next_task is not None:
                        next_task_id = self._precomputed_next_task
                        self._precomputed_next_task = None  # 用完清空
                        
                        # 🔥 检查话题是否新鲜（本轮 Step1 是否运行）
                        topic_info = ""
                        if not getattr(self, '_current_turn_topic_set', False):
                            # 本轮没有运行 Step 1，话题可能是陈旧的
                            # 同步运行 Step 1 获取新鲜话题
                            print(f"[AgentFC] ⚠️ 预计算任务来自上轮，同步刷新话题...")
                            _ = self._select_next_task()  # 这会更新 _last_bridge_hint
                        
                        if hasattr(self, '_last_bridge_hint') and self._last_bridge_hint:
                            topic_info = f" (话题: {self._last_bridge_hint})"
                        
                        # 🔥 检查 recall_3words 的时间限制
                        is_valid_precomputed = True
                        if next_task_id == "recall_3words" and self._registration_ts:
                            elapsed = time.time() - self._registration_ts
                            if elapsed < 120:
                                print(f"[AgentFC] ⏳ 预计算任务 '{next_task_id}' 未满足时间限制 (还需 {120 - elapsed:.0f}s)，回退到 buffer_chat")
                                is_valid_precomputed = False
                                next_task_id = "buffer_chat"
                        
                        if is_valid_precomputed:
                            print(f"[AgentFC] ⚡ 使用预计算任务: {next_task_id}{topic_info}")
                        else:
                            print(f"[AgentFC] 🔄 回退到 buffer_chat{topic_info}")
                    else:
                        next_task_id = self._select_next_task()
            assessment_complete = (next_task_id is None)


            
            if assessment_complete:
                print(f"[AgentFC] ✅ 所有MMSE任务已完成！")
            else:
                print(f"[AgentFC] 下一个任务: {next_task_id}")
                # 更新 current_dimension 以兼容旧代码
                next_task_cfg = self.TASK_CONFIG.get(next_task_id, {})
                next_dim_id = next_task_cfg.get('dimension_id')
                if next_dim_id and next_dim_id in self._dimension_map:
                    self.current_dimension = self._dimension_map[next_dim_id]
                    dimension_id = next_dim_id
                    dimension_name = self.current_dimension.get('name', '未知')
            
            step2_time = time.time() - step2_start
            print(f"[AgentFC] ✅ 阶段二完成 (耗时 {step2_time:.2f}秒)\n")
            
            # ⭐ 如果所有任务已完成，返回评估结果
            if assessment_complete:
                # 获取 MMSE 汇总
                summary_json = self.mmse_tool._run(session_id=session_id, dimension_id="", action="summary")
                summary = json.loads(summary_json)
                
                total_score = summary.get('total_score', 0)
                scaled_score = summary.get('scaled_total_score', total_score)
                coverage = summary.get('coverage', 1.0)
                
                risk_assessment = self._calculate_alzheimers_risk(int(scaled_score))

                print(f"[AgentFC] 🏆 MMSE原始分: {total_score}, 折算分: {scaled_score:.1f}/30 (覆盖率: {coverage:.1%})")

                completion_message = self._generate_completion_message(
                    int(scaled_score),
                    risk_assessment,
                    patient_profile
                )
                

                
                return {
                    'output': completion_message,
                    'response': completion_message,
                    'assessment_complete': True,
                    'total_score': total_score,
                    'scaled_score': scaled_score,
                    'coverage': coverage,
                    'max_score': 30,
                    'risk_assessment': risk_assessment,
                    'dimension': '完成',
                    'total_time': time.time() - start_time
                }
            
            # ==================== 步骤3: 根据任务池生成下一个问题 ====================
            print(f"[AgentFC] 🎯 阶段三: 策略生成 (Next: {next_task_id})...")
            step3_start = time.time()
            
            image_display_command = None
            effective_task_id_for_last = next_task_id

            if next_task_id and self._needs_consent_for_task(next_task_id):
                # 🔥 检查任务状态，如果是进行中则无需再次同意
                task_status = self.session_data.get('task_progress', {}).get(next_task_id, {}).get('status')
                is_granted = (next_task_id == self._consent_granted_task_id) or (task_status == 'in_progress')
                
                if not is_granted:
                    self._pending_consent_task_id = next_task_id
                    effective_task_id_for_last = "buffer_consent"
                    next_question = self._build_consent_prompt(next_task_id, patient_profile, user_answer)
                else:
                    # 已经是已授予或进行中，继续生成问题（fall through）
                    if next_task_id == self._consent_granted_task_id:
                        self._consent_granted_task_id = None
                    # 不设置 next_question，让下面的代码块处理
                    next_question = None  # Will be set below
            else:
                next_question = None  # Will be set below

            # 🆕 如果 next_question 还没设置（不是 consent 请求），则生成问题
            if next_question is None:
                task_instruction = self._get_task_instruction(next_task_id)
                
                # 🔥 注入话题过渡提示（针对非 buffer 任务）
                if hasattr(self, '_last_bridge_hint') and self._last_bridge_hint:
                    task_instruction += f"\n\n【必须执行的过渡策略】\n1. 先顺着话题「{self._last_bridge_hint}」回应一句。\n2. 然后**必须**转折到本任务的核心问题（即使话题不顺也要转折）。\n3. ❌ 禁止一直停留在闲聊话题上，你的核心目标是评估！"
                    # self._last_bridge_hint = None # 暂不清除，以防重试
                
                persona_hooks = self._extract_persona_hooks(patient_profile, chat_history)
                must_include = self._get_must_include_for_task(next_task_id)
                
                # 🆕 根据任务类型决定问题生成策略
                special_tasks = {
                    "registration_3words", "recall_3words", "attention_calc_life_math",
                    "language_naming_watch", "language_naming_pencil", 
                    "language_repetition_sentence", "language_reading_close_eyes", "language_3step_action"
                }
                
                if next_task_id in special_tasks:
                    if next_task_id in {"language_naming_watch", "language_naming_pencil"}:
                        image_id = 'watch' if next_task_id == "language_naming_watch" else 'pencil'
                        patient_name = patient_profile.get('name', '')
                        greeting = f"{patient_name}，" if patient_name else ""
                        if last_task_id in {"language_naming_watch", "language_naming_pencil"}:
                            title = "爷爷，那再看一张，您说这是什么呀？"
                        else:
                            title = "我给您看张图，您说这是什么呀？"
                        next_question = f"{greeting}{title}"
                        img_result = self.image_tool._run(
                            image_id=image_id,
                            title=title,
                            action='show'
                        )
                        img_data = json.loads(img_result)
                        if img_data.get('success'):
                            image_display_command = img_data.get('display_command')
                    elif next_task_id == "language_reading_close_eyes":
                        patient_name = patient_profile.get('name', '')
                        greeting = f"{patient_name}，" if patient_name else ""
                        title = "我给您看一句话，您读出来，然后照着做就行。"
                        next_question = f"{greeting}{title}"
                        img_result = self.image_tool._run(
                            image_id='close_eyes',
                            title="请读一下图片上的文字，并照着做",
                            action='show'
                        )
                        img_data = json.loads(img_result)
                        if img_data.get('success'):
                            image_display_command = img_data.get('display_command')
                    elif next_task_id == "language_3step_action":
                        patient_name = patient_profile.get('name', '')
                        greeting = f"{patient_name}，" if patient_name else ""
                        next_question = (
                            f"{greeting}那咱们开始：请您先抬起右手，"
                            "再摸一下左耳朵，最后把手放回腿上。您做完跟我说一声。"
                        )
                    else:
                        # 🔥 修复：正确判断是否是维度切换
                        # 获取上一个任务的 dimension_id
                        last_dim_id = self.TASK_CONFIG.get(last_task_id, {}).get('dimension_id') if last_task_id else None
                        # 获取当前任务的 dimension_id
                        current_dim_id = self.TASK_CONFIG.get(next_task_id, {}).get('dimension_id')
                        # 维度切换条件：上一个任务是 buffer 或者 维度 ID 不同
                        is_dim_switch = (last_task_id is None or 
                                        last_task_id in self.BUFFER_TASKS or 
                                        last_dim_id != current_dim_id)
                        
                        # 🔥 任务ID到特殊维度ID的映射（StandardQuestionTool需要）
                        TASK_TO_SPECIAL_DIM = {
                            "registration_3words": "registration",
                            "recall_3words": "recall",
                            "attention_calc_life_math": "attention_calculation",
                            "language_repetition_sentence": "language_repetition",  # 🔥 复述句子
                        }
                        special_dim_id = TASK_TO_SPECIAL_DIM.get(next_task_id, dimension_id)
                        
                        print(f"[AgentFC] 🔍 调用 StandardQuestionTool: dimension_id={special_dim_id}, last_dim={last_dim_id}, current_dim={current_dim_id}, is_dimension_switch={is_dim_switch}")
                        
                        # 🔥 对于 attention_calculation，传入当前计算值
                        calc_current = None
                        calc_step = 7
                        if dimension_id == "attention_calculation":
                            # 如果是第一轮（刚切换到这个任务），使用 None（从100开始）
                            # 如果是后续轮次，使用之前保存的期望答案
                            if not is_dim_switch and hasattr(self, '_calculation_current_value'):
                                calc_current = self._calculation_current_value
                                calc_step = self._calculation_step
                            print(f"[AgentFC] 🧮 计算任务: current_value={calc_current}, step={calc_step}")
                        
                        standard_result = self._call_standard_question(
                            special_dim_id,  # 🔥 使用映射后的特殊维度ID
                            is_dimension_switch=is_dim_switch,
                            memory_words=self.session_data.get('memory_words'),
                            patient_name=patient_profile.get('name'),
                            calculation_current_value=calc_current,
                            calculation_step=calc_step
                        )
                        
                        print(f"[AgentFC] 🔍 StandardQuestionTool 返回: has_standard_question={standard_result.get('has_standard_question')}, message={standard_result.get('message', 'N/A')}")
                        
                        if standard_result.get('has_standard_question'):
                            next_question = standard_result['question']
                            print(f"[AgentFC] 📝 特殊任务问题: {next_task_id}")
                            
                            if standard_result.get('memory_words'):
                                self.session_data['memory_words'] = standard_result['memory_words']
                                print(f"[AgentFC] 💾 保存记忆词: {standard_result['memory_words']}")
                            if standard_result.get('calculation_config'):
                                self.session_data['calculation_config'] = standard_result['calculation_config']
                                # 🔥 更新计算状态：保存期望答案作为下一轮的当前值
                                expected_ans = standard_result['calculation_config'].get('expected_answer')
                                if expected_ans is not None:
                                    self._calculation_current_value = expected_ans
                                    print(f"[AgentFC] 🧮 更新计算状态: 下一轮从 {expected_ans} 开始")
                            
                            if standard_result.get('requires_image'):
                                image_config = standard_result.get('image_config', {})
                                img_result = self.image_tool._run(
                                    image_id=image_config.get('image_id', 'pentagons'),
                                    title=image_config.get('title', '请看下面的图片'),
                                    action='show'
                                )
                                img_data = json.loads(img_result)
                                if img_data.get('success'):
                                    image_display_command = img_data.get('display_command')
                        else:
                            next_question = self._call_question_generation(
                                dimension_name, "", patient_profile, chat_history, False,
                                is_dimension_switch=True,
                                needs_encouragement=False,
                                resistance_info=None,
                                task_instruction=task_instruction,
                                persona_hooks=persona_hooks,
                                must_include=must_include,
                                patient_emotion=current_emotion
                            )

                elif next_task_id in self.BUFFER_TASKS:
                    # 缓冲任务：生成自然闲聊或回答用户问题
                    if next_task_id == "buffer_answer_question":
                        next_question = self._generate_answer_to_user_question(user_answer, patient_profile, chat_history)
                    else:
                        next_question = self._generate_buffer_question(next_task_id, patient_profile, chat_history)
                
                else:
                    # 通用任务（orientation 等）：使用 QuestionGenerationTool + task_instruction
                    # 🔥 性能优化：复用 Phase 1 预取的检索结果（省去串行等待）
                    if hasattr(self, '_prefetched_retrieval') and self._prefetched_retrieval:
                        retrieval_result = self._prefetched_retrieval
                        self._prefetched_retrieval = None  # 用完清除
                        print(f"[AgentFC] ⚡ 使用预取检索结果（节省 ~0.3-0.5s）")
                    else:
                        query_result = self._call_query_generation(dimension_name, chat_history)
                        retrieval_result = self._call_knowledge_retrieval(query_result)
                    
                    next_question = self._call_question_generation(
                        dimension_name,
                        retrieval_result.get('knowledge_context', ''),
                        patient_profile,
                        chat_history,
                        eval_result.get('need_followup', False),
                        is_dimension_switch=(last_task_id is None or last_task_id in self.BUFFER_TASKS),
                        needs_encouragement=eval_result.get('needs_encouragement', False),
                        resistance_info=eval_result.get('resistance_info'),
                        task_instruction=task_instruction,
                        persona_hooks=persona_hooks,
                        must_include=must_include,
                        patient_emotion=current_emotion
                    )
            
            # 🆕 去重仅用于缓冲闲聊任务；评估任务题目不能被替换为闲聊
            if next_question and effective_task_id_for_last != "buffer_consent":
                next_question = self._ensure_question_not_repeated(
                    next_question, patient_profile, chat_history, task_id=effective_task_id_for_last
                )

            # 🆕 更新 _last_task_id
            self._last_task_id = effective_task_id_for_last
            
            # 🆕 记录已问问题（用于去重）
            if next_question and len(next_question) > 5:
                self._asked_questions.append(next_question[:80])
                if len(self._asked_questions) > 20:
                    self._asked_questions = self._asked_questions[-20:]
            
            step3_time = time.time() - step3_start
            print(f"[AgentFC] ✅ 阶段三完成 (任务: {next_task_id}，耗时 {step3_time:.2f}秒)\n")
            
            # 检测 language 维度是否需要展示图片
            # 🔥 跳过 consent 请求，避免邀请语中的关键词误触发图片展示
            if dimension_id == 'language' and not image_display_command and effective_task_id_for_last != "buffer_consent":
                image_result = self._check_and_display_image(next_question, session_id)
                if image_result.get('should_display'):
                    image_display_command = image_result.get('display_command')
                    print(f"[AgentFC] 📋 需要展示图片: {image_result.get('image_id')}")
            
            # ==================== 步骤5: 存储对话 ====================
            self._call_conversation_storage(session_id, user_answer, next_question)
            
            total_time = time.time() - start_time
            print(f"[AgentFC] 🏁 处理完成 (总耗时 {total_time:.2f}秒)\n")
            
            # 🔥 保存本次生成的问题（供下一轮使用）
            self._last_generated_question = next_question
            
            result = {
                'output': next_question,  # ⭐ voice_server.py 期望的字段
                'response': next_question,
                'has_resistance': False,
                'resistance_category': 'normal',  # 🎯 情绪类别（无抵抗时为normal）
                'dimension': dimension_name,
                'dimension_id': dimension_id,
                'evaluation': eval_result,
                'mmse_score': mmse_result,  # ⭐ 添加MMSE评分信息
                'selected_topic': self._last_bridge_hint,  # ⭐ 传递选中话题给后台映射
                'total_time': total_time,
                'step_times': {
                    'detection_eval': step1_time,
                    'score_recording': step2_time,
                    'question_gen': step3_time
                }
            }
            
            # 添加图片展示指令
            if image_display_command:
                result['image_display'] = image_display_command
            

            
            return result
            
        except Exception as e:
            print(f"[AgentFC] ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            error_message = "抱歉，我需要重新理解一下您的回答。能否再说一遍？"
            return {
                'output': error_message,  # ⭐ voice_server.py 期望的字段
                'response': error_message,
                'error': str(e),
            }
    
            

        

    def _call_standard_question(
        self, 
        dimension_id: str, 
        is_dimension_switch: bool,
        memory_words: Optional[List[str]] = None,
        patient_name: Optional[str] = None,
        calculation_current_value: Optional[int] = None,
        calculation_step: Optional[int] = 7
    ) -> Dict:
        """
        调用特殊维度问题工具
        
        Args:
            dimension_id: 维度ID
            is_dimension_switch: 是否刚切换到此维度
            memory_words: 之前的记忆词（用于 recall 维度）
            patient_name: 患者姓名（用于个性化称呼）
            calculation_current_value: 连续减法的当前值（用于 attention_calculation）
            calculation_step: 连续减法的步长（默认7）
            
        Returns:
            包含 has_standard_question, question, memory_words 等字段的字典
        """
        result_json = self.standard_question_tool._run(
            dimension_id=dimension_id,
            is_dimension_switch=is_dimension_switch,
            memory_words=memory_words,
            patient_name=patient_name,
            calculation_current_value=calculation_current_value,
            calculation_step=calculation_step
        )
        return json.loads(result_json)
    
    # 🔥 性能优化：规则化问题分类（替代 LLM API 调用，省 1-2 秒）
    _TASK_KEYWORD_RULES = [
        ("orientation_time_weekday", ['星期', '周几', '礼拜']),
        ("orientation_time_date_month_season", ['几月', '季节', '几号', '日期', '月份']),
        ("orientation_place_city_district", ['住哪', '城市', '哪个区', '什么地方', '哪里', '地址', '省份']),
        ("persona_collect_1", ['爱好', '兴趣', '喜欢做', '喜欢什么', '娱乐', '消遣']),
        ("persona_collect_2", ['起床', '睡觉', '作息', '吃什么', '早饭', '晚饭', '习惯']),
        ("registration_3words", ['记住', '记下', '这几个词', '三个词']),
        ("recall_3words", ['刚才', '记的词', '还记得', '之前的词']),
        ("language_naming_watch", ['手表', '看图']),
        ("language_naming_pencil", ['铅笔']),
        ("language_repetition_sentence", ['复述', '跟我说', '再说一遍']),
        ("language_reading_close_eyes", ['闭眼', '读字', '闭上眼']),
        ("language_3step_action", ['三步', '左手', '右手', '放在', '指令']),
        ("attention_calc_life_math", ['算', '减7', '减去', '计算', '100']),
    ]

    def classify_question_sync(self, question: str, candidates: List[str]) -> str:
        """规则化问题分类（替代 LLM API 调用，省 1-2 秒）"""
        if not candidates:
            return "buffer_chat"
        
        q = question.strip()
        for task_id, keywords in self._TASK_KEYWORD_RULES:
            if task_id in candidates and any(kw in q for kw in keywords):
                print(f"[Classification] ✅ 规则分类: {task_id} (命中关键词)")
                return task_id
            
        print(f"[Classification] ℹ️ 规则未命中，归为闲聊")
        return "buffer_chat"
    
    def start_classification_async(self, question: str):
        """
        启动后台分类任务（非阻塞）
        """
        import concurrent.futures
        
        # 🔥 关闭之前的 executor（如果有）
        if hasattr(self, '_classification_executor') and self._classification_executor:
            try:
                self._classification_executor.shutdown(wait=False)
            except:
                pass
        
        # 使用当前可用的候选任务
        candidates = self._available_candidates.copy()
        
        # 🔥 增强日志：显示问题和候选任务
        question_preview = question[:50] + "..." if len(question) > 50 else question
        print(f"[Classification] 🚀 后台分类任务启动")
        print(f"[Classification]    ├── 问题: {question_preview}")
        print(f"[Classification]    └── 候选任务: {candidates[:5]}{'...' if len(candidates) > 5 else ''}")
        
        def run_classification():
            return self.classify_question_sync(question, candidates)
        
        # 使用线程池执行（避免阻塞事件循环）
        self._classification_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = self._classification_executor.submit(run_classification)
        
        # 保存 future 供下一轮检查
        self._pending_classification_task = future
    
    def get_classification_result(self, timeout: float = 1.0) -> Optional[str]:
        """
        获取分类结果（最多等待指定时间）
        
        Args:
            timeout: 最大等待时间（秒）
            
        Returns:
            任务ID 或 None（如果超时或未完成）
        """
        import concurrent.futures
        
        if self._pending_classification_task is None:
            return self._last_classification_result
        
        try:
            result = self._pending_classification_task.result(timeout=timeout)
            self._last_classification_result = result
            self._pending_classification_task = None
            return result
        except concurrent.futures.TimeoutError:
            print(f"[Classification] ⏱️ 分类超时，使用上次结果")
            return self._last_classification_result
        except Exception as e:
            print(f"[Classification] ❌ 获取结果失败: {e}")
            self._pending_classification_task = None
            return "buffer_chat"
    
    def _call_resistance_detection(self, question: str, answer: str) -> Dict:
        """调用抵抗检测工具"""
        result_json = self.resistance_tool._run(question=question, answer=answer)
        return json.loads(result_json)
    
    def _call_comfort_response(
        self, resistance_result: Dict, patient_answer: str, patient_profile: Dict,
        chat_relaxed: bool = False  # 保留参数但不传递给工具
    ) -> Dict:
        """
        调用安慰话语生成工具
        
        Args:
            resistance_result: 抵抗检测结果
            patient_answer: 患者回答
            patient_profile: 患者画像
            chat_relaxed: 是否聊轻松话题（目前工具不使用此参数）
        """
        result_json = self.comfort_tool._run(
            resistance_category=resistance_result.get('category', 'refusal'),
            patient_answer=patient_answer,
            resistance_reason=resistance_result.get('rationale'),
            patient_age=patient_profile.get('age'),
            patient_name=patient_profile.get('name'),
            patient_gender=patient_profile.get('gender'),
            used_topics=self._used_chat_topics,
            chat_history=self.session_data.get('chat_history')
        )
        
        # 更新已聊话题
        try:
            data = json.loads(result_json)
            if topic := data.get('selected_topic'):
                self._used_chat_topics.append(topic)
                print(f"[AgentFC] 📌 记录新话题: {topic} (总计: {len(self._used_chat_topics)})")
        except:
            pass
            
        return json.loads(result_json)
    
    def _call_answer_evaluation(
        self,
        question: str,
        answer: str,
        task_id: str,
        patient_profile: Dict,
        expected_answer: Optional[str] = None,
    ) -> Dict:
        """调用回答评估工具"""
        result_json = self.eval_tool._run(
            question=question,
            answer=answer,
            task_id=task_id,
            expected_answer=expected_answer,
            patient_profile=patient_profile
        )
        return json.loads(result_json)

    def _reset_task_pool(self, session_id: str) -> None:
        self._active_session_id = session_id
        self._last_task_id = None
        self._last_generated_question = "请开始评估"
        self._task_done = set()
        self._task_attempts = {}
        self._task_best = {}
        self._task_turns = {}  # 🔥 每个任务的对话轮数计数
        self._registration_ts = None
        self._turn_counter = 0
        self._session_start_ts = time.time()
        self.is_in_comfort_mode = False
        self.comfort_turn_count = 0
        self._comfort_interrupted_task_id = None
        self._asked_to_continue = False  # 重置过渡语标记
        self._asked_questions = []  # 重置已问问题列表
        self._used_chat_topics = []  # 🔥 重置安抚阶段话题记忆
        self._used_bridge_topics = []  # 🔥 重置过渡话题记忆
        self._last_bridge_hint = None  # 🔥 重置当前过渡提示
        self._consecutive_free_chat = 0  # 🔥 重置自由闲聊计数
        self._consecutive_buffer_count = 0  # 🔥 重置连续缓冲计数
        self._precomputed_next_task = None  # 🔥 重置预计算任务
        self._prefetched_retrieval = None  # 🔥 重置预取检索结果
        self._current_turn_topic_set = False
        self._pending_consent_task_id = None
        self._consent_granted_task_id = None
        self._task_cooldown_until = {}
        self.session_data['memory_words'] = None
        self.session_data['calculation_config'] = None
        self.dimension_index = 0
        self.current_dimension = MMSE_DIMENSIONS[0]
        self.consecutive_failures = 0

    def _get_max_consecutive_buffer_chat(self) -> int:
        """读取连续 buffer 上限，默认 1（最多连续一轮闲聊）。"""
        try:
            limit = int(os.getenv("MAX_CONSECUTIVE_BUFFER_CHAT", "1"))
        except ValueError:
            limit = 1
        return max(0, limit)

    def _select_next_task(self) -> Optional[str]:
        """
        任务池调度：选择下一个要执行的任务
        
        硬约束：
        1. 已完成的任务不再重复
        2. recall_3words 必须在 registration_3words 完成后至少2分钟才能执行
        3. registration_3words 必须在 persona_collect 之后
        
        软约束（优先级）：
        1. 优先完成 persona_collect 收集用户信息
        2. orientation 任务优先（自然话题）
        3. registration 尽早执行以留出 recall 时间
        4. language 任务可穿插
        5. recall 在时间足够时执行
        6. attention_calculation 放在中后期
        """
        now = time.time()
        candidates = []
        
        for task_id in self.REQUIRED_TASKS:
            if task_id in self._task_done:
                continue

            until_turn = self._task_cooldown_until.get(task_id)
            if until_turn is not None and self._turn_counter < until_turn:
                continue
            
            cfg = self.TASK_CONFIG.get(task_id, {})
            
            # 硬约束：recall 必须等 registration 完成后 >= 2分钟
            if task_id == "recall_3words":
                if "registration_3words" not in self._task_done:
                    continue
                if self._registration_ts is None:
                    continue
                elapsed = now - self._registration_ts
                if elapsed < 120:  # 2分钟 = 120秒
                    print(f"[TaskPool] ⏳ recall_3words 需等待 {120 - elapsed:.0f}秒")
                    continue
            
            # 硬约束：registration 必须在 persona 收集后
            if task_id == "registration_3words":
                if "persona_collect_1" not in self._task_done or "persona_collect_2" not in self._task_done:
                    continue
            
            candidates.append(task_id)
        
        # 🔥 保存候选任务供异步分类使用
        self._available_candidates = candidates.copy()
        
        if not candidates:
            # 检查是否所有必需任务已完成
            remaining = set(self.REQUIRED_TASKS) - self._task_done
            if remaining:
                # 可能是 recall 还在等待
                if "recall_3words" in remaining and self._registration_ts:
                    elapsed = now - self._registration_ts
                    wait_time = 120 - elapsed
                    if wait_time > 0:
                        print(f"[TaskPool] ⏳ 等待 recall，插入 buffer_chat")
                        return "buffer_chat"
            return None

        # 🔥 初始化连续闲聊计数器（如果不存在）
        if not hasattr(self, '_consecutive_buffer_count'):
            self._consecutive_buffer_count = 0
        max_buffer_rounds = self._get_max_consecutive_buffer_chat()

        # 认知任务后插入 buffer_chat（但受计数器限制）
        if self._last_task_id and self._is_cognitive_task(self._last_task_id):
            if any(self._is_cognitive_task(t) for t in candidates):
                # 超过连续闲聊上限后，强制走任务选择
                if self._consecutive_buffer_count >= max_buffer_rounds:
                    print(f"[TaskPool] ⚠️ 连续闲聊 {self._consecutive_buffer_count} 次，不再插入 buffer")
                    # 继续走 LLM 选择逻辑
                else:
                    self._consecutive_buffer_count += 1
                    return "buffer_chat"
        
        # 🆕 用LLM动态选择下一个任务（根据对话上下文）
        if len(candidates) == 1:
            self._consecutive_buffer_count = 0  # 选到唯一任务，重置计数
            return candidates[0]
        
        return self._llm_select_task(candidates)

    def _normalize_text(self, text: str) -> str:
        s = (text or "").strip().lower()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[\?？!！。．，,、:：;；\"'“”‘’\(\)（）\[\]【】{}]", "", s)
        return s

    def _is_similar_text(self, a: str, b: str) -> bool:
        na = self._normalize_text(a)
        nb = self._normalize_text(b)
        if not na or not nb:
            return False
        if na in nb or nb in na:
            return True
        # 使用 0.75 阈值，平衡防重复和多样性
        if SequenceMatcher(None, na, nb).ratio() >= 0.75:
            print(f"[AgentFC] ⚠️ 检测到相似问题 (ratio={SequenceMatcher(None, na, nb).ratio():.2f})")
            return True
        return False

    def _ensure_question_not_repeated(
        self, question: str, patient_profile: Dict, chat_history: List, task_id: Optional[str] = None
    ) -> str:
        if not question:
            return question
        # 非 buffer 任务（评估题）允许重复提问，不做“改问闲聊”替换
        if task_id and task_id not in self.BUFFER_TASKS:
            return question
        for prev in self._asked_questions[-12:]:
            if self._is_similar_text(question, prev):
                print(f"[AgentFC] ⚠️ 检测到重复问题: {question}")
                # 🔥 强制重新生成一个变体，而不是回退到闲聊
                # 如果是多轮任务（如计算），这里的重复意味着状态没更新，必须强制推动状态
                if self._last_task_id == "attention_calc_life_math":
                     # 尝试强制更新计算值（兜底）
                     if hasattr(self, '_calculation_current_value') and self._calculation_current_value:
                         self._calculation_current_value -= 7
                         return f"那再减7，现在是多少？"
                
                return self._fallback_followup_question(patient_profile, chat_history)
        return question

    def _fallback_followup_question(self, patient_profile: Dict, chat_history: List) -> str:
        patient_name = patient_profile.get('name', '')
        greeting = f"{patient_name}，" if patient_name else ""
        last_user = ""
        for msg in reversed(chat_history or []):
            if msg.get('role') == 'user':
                last_user = (msg.get('content') or '').strip()
                break

        if "综艺" in last_user:
            return f"{greeting}那您更喜欢轻松搞笑的，还是唱歌跳舞那种呀？"
        if "电视" in last_user or "节目" in last_user or "没看" in last_user or "没有" in last_user:
            return f"{greeting}那您平时更爱怎么解闷呀？听歌、刷手机，还是跟家里人聊聊天？"
        return f"{greeting}那您平时在家最喜欢干点啥？"

    def _is_cognitive_task(self, task_id: Optional[str]) -> bool:
        if not task_id:
            return False
        return task_id in {
            "attention_calc_life_math",
            "language_naming_watch",
            "language_naming_pencil",
            "language_repetition_sentence",
            "language_reading_close_eyes",
            "language_3step_action",
            "registration_3words",
            "recall_3words",
        }

    def _needs_consent_for_task(self, task_id: str) -> bool:
        return task_id in {
            "attention_calc_life_math",
            "language_naming_watch",
            "language_naming_pencil",
            "language_repetition_sentence",
            "language_reading_close_eyes",
            "language_3step_action",
        }

    def _build_consent_prompt(self, task_id: str, patient_profile: Dict, user_answer: str = None) -> str:
        """用 LLM 生成自然的征求同意话语，包含对用户回答的反馈"""
        patient_name = patient_profile.get('name', '')
        
        purpose = "玩个30秒小互动"
        if task_id == "attention_calc_life_math":
            purpose = "做个30秒小算术"
        elif task_id in {"language_naming_watch", "language_naming_pencil"}:
            purpose = "看张图说说是什么"
        elif task_id == "language_repetition_sentence":
            purpose = "跟着说一句小短句"
        elif task_id == "language_reading_close_eyes":
            purpose = "看一句话照着做一下"
        elif task_id == "language_3step_action":
            purpose = "做个小动作"

        # 🆕 改进方案：LLM 只生成回应，代码保证任务邀请
        try:
            from src.llm.model_pool import get_pooled_llm
            import random
            llm = get_pooled_llm(pool_key='7b_complex')
            
            # 任务邀请模板（代码保证 purpose 出现）
            invite_templates = [
                f"诶，能陪我{purpose}不？不想来也行～",
                f"对了，咱们{purpose}呗？不难的～",
                f"您能帮我个忙不？就{purpose}～",
                f"要不咱来{purpose}？不想玩就算～",
            ]
            invite = random.choice(invite_templates)
            
            # 如果有用户回答，用 LLM 生成回应部分
            if user_answer and len(user_answer.strip()) > 2:
                prompt = f"""你是{patient_name or '老人'}的晚辈，正在陪他/她聊天。
老人刚才说了：「{user_answer}」

请用1句简短的话回应老人说的内容（5-15字），要求：
- 针对老人说的具体内容回应
- 不要用万金油的「挺好」「不错」
- 口语化、接地气

示例：
- 老人说看书 → "余华的书确实好看！"
- 老人说位置 → "甘井子区那边挺方便的"
- 老人说冷 → "是挺冷的，多穿点"

直接输出回应，不要加引号："""
                
                response = llm.invoke([{"role": "user", "content": prompt}])
                ack = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                ack = ack.strip('"').strip("'").strip('「').strip('」')
                
                # 拼接：回应 + 邀请（邀请由代码保证）
                if ack and len(ack) < 30:
                    result = f"{ack}，{invite}" if not ack.endswith(("！", "!", "。", ".")) else f"{ack}{invite}"
                    print(f"[AgentFC] 🎲 生成征求同意: ack='{ack}', invite='{invite}'")
                    return result
            
            # 没有用户回答，直接用邀请模板
            if patient_name:
                result = f"{patient_name}，{invite}"
            else:
                result = invite
            print(f"[AgentFC] 🎲 使用模板征求同意: {result}")
            return result
        except Exception as e:
            print(f"[AgentFC] ⚠️ LLM 生成征求同意失败: {e}")
        
        # 如果 LLM 失败，从多个模板中随机选择
        import random
        if user_answer and len(user_answer.strip()) > 2:
            templates = [
                f"好嘞！{patient_name}，来，咱们{purpose}呗？不想玩就算～",
                f"嗯嗯！诶{patient_name}，能陪我{purpose}不？",
                f"说得好！对了{patient_name}，咱来{purpose}，您看行不？",
            ]
        else:
            templates = [
                f"{patient_name}，来，咱们玩个小游戏呗？不想玩就算～",
                f"诶{patient_name}，能陪我{purpose}不？就当逗我玩儿～",
                f"{patient_name}，我想跟您{purpose}，您看行不？不勉强哈～",
            ]
        return random.choice(templates)

    def _is_user_asking_question(self, user_answer: str) -> bool:
        """🔥 规则化检测用户是否在提问（替代 LLM，省 1-2 秒）"""
        if not user_answer or len(user_answer.strip()) < 4:
            return False
        
        text = user_answer.strip()

        # 排除：抱怨/反问式（不算真正提问）
        complaint_patterns = [
            '行不行', '好不好', '能不能别', '烦不烦', '有完没完',
            '不行吗', '不好吗', '可以吗', '算了吧', '别问', '不想',
            '我不知道', '忘了', '不记得', '没听清',
        ]
        if any(p in text for p in complaint_patterns):
            return False

        # 短句且无问号 → 不是提问
        if len(text) <= 5 and '？' not in text and '?' not in text:
            return False

        # 信息性提问关键短语（直接命中）
        info_phrases = ['天气怎么', '你叫什么', '你是谁', '现在几点', '能告诉我', '请问']
        if any(p in text for p in info_phrases):
            print(f"[AgentFC] 🔍 用户提问检测(规则): '{text[:30]}' → 是提问")
            return True

        # 疑问词 + 问号 → 大概率是提问
        question_words = ['什么', '怎么', '哪里', '哪个', '谁', '几点', '几号', '多少', '为什么', '为啥', '啥时候']
        has_qword = any(qw in text for qw in question_words)
        has_qmark = '？' in text or '?' in text

        if has_qword and has_qmark:
            print(f"[AgentFC] 🔍 用户提问检测(规则): '{text[:30]}' → 是提问")
            return True

            return False

    def _generate_answer_to_user_question(self, user_question: str, patient_profile: Dict, chat_history: List) -> str:
        """生成对用户问题的回答"""
        patient_name = patient_profile.get('name', '')
        greeting = f"{patient_name}，" if patient_name else ""
        
        # 🔥 传入结构化对话历史（而非字符串）
        recent_history = None
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-6:]
        
        # 用LLM生成回答
        result_json = self.question_tool._run(
            dimension_name="闲聊",
            dimension_description="回答对方的问题，然后自然地继续聊天",
            patient_name=patient_profile.get('name'),
            patient_gender=patient_profile.get('gender'),
            patient_age=patient_profile.get('age'),
            conversation_history=recent_history,
            task_instruction=f"对方问了一个问题：'{user_question}'。请先简短回答这个问题，然后自然地继续聊天。不要反问同样的问题。",
        )
        
        try:
            result = json.loads(result_json)
            if result.get('success') and result.get('question'):
                return result['question']
        except:
            pass
        
        # 兜底回答
        return f"{greeting}这个我也不太清楚呢，您怎么看？"
    
    def _llm_select_task(self, candidates: List[str]) -> str:
        """
        话题驱动的任务选择
        
        逻辑：
        1. 优先返回 LLM 建议的自然话题（Topic） -> 第一步（保留）
        2. 后续的"判断话题是否属于任务" -> 移至后台并行（第二步已移除）
        """
        import random
        import re
        import json
        import os
        from langchain_openai import ChatOpenAI
        import json
        import os
        
        # 初始化连续闲聊计数器
        if not hasattr(self, '_consecutive_buffer_count'):
            self._consecutive_buffer_count = 0
            
        # 任务描述
        task_descriptions = {
            "persona_collect_1": "了解兴趣爱好",
            "persona_collect_2": "了解生活习惯",
            "orientation_time_weekday": "聊今天星期几",
            "orientation_time_date_month_season": "聊日期/季节",
            "orientation_place_city_district": "聊居住地点",
            "registration_3words": "记3个词",
            "recall_3words": "回忆3个词",
            "attention_calc_life_math": "简单算术",
            "language_naming_watch": "命名(表)",
            "language_naming_pencil": "命名(笔)",
            "language_repetition_sentence": "复述句子",
            "language_reading_close_eyes": "读字动作",
            "language_3step_action": "三步动作"
        }

        # 构建候选任务列表
        candidate_list = []
        non_buffer_candidates = []
        for task_id in candidates:
            desc = task_descriptions.get(task_id, task_id)
            candidate_list.append(f"- {task_id}: {desc}")
            if task_id != "buffer_chat":
                non_buffer_candidates.append(task_id)

        # 获取最近对话
        recent_chat = ""
        if hasattr(self, 'session_data') and self.session_data.get('chat_history'):
            history = self.session_data['chat_history'][-4:]
            for msg in history:
                role = "对方" if msg.get('role') == 'user' else "你"
                content = msg.get('content', '')[:50]
                if content:
                    recent_chat += f"{role}：{content}\n"

        # 🛑 保底机制：连续闲聊超过3次，强制选择任务
        max_buffer_rounds = self._get_max_consecutive_buffer_chat()
        if self._consecutive_buffer_count >= max_buffer_rounds and non_buffer_candidates:
            print(f"[TaskPool] ⚠️ 连续闲聊 {self._consecutive_buffer_count} 次，强制选择任务")
            
            forced_system_message = f"""你是一个对话策略师。
现在需要从下面的任务列表中选择一个最合适的任务。

【候选任务】（必须选择其中一个）
{chr(10).join([f"- {tid}: {task_descriptions.get(tid, tid)}" for tid in non_buffer_candidates])}

【输出格式】严格输出 JSON：
{{
    "from_topic": "当前话题",
    "to_topic": "下一个话题",
    "selected_task_id": "必须填一个候选任务ID",
    "reason": "选择理由"
}}

【要求】
- selected_task_id 必须是上面列出的候选任务之一
- 选择与当前对话最相关、过渡最自然的任务
"""
            forced_user_message = f"""最近对话：
{recent_chat if recent_chat else '（刚开始聊天）'}

请选择一个任务："""
            
            try:
                from src.llm.http_client_pool import get_siliconflow_chat_openai
                router_model = os.getenv("TASK_ROUTER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
                llm = get_siliconflow_chat_openai(
                    model=router_model,
                    temperature=0.5,
                    timeout=15,
                    max_retries=1,
                )
                
                response = llm.invoke([
                    {"role": "system", "content": forced_system_message},
                    {"role": "user", "content": forced_user_message}
                ])
                
                content = response.content.strip()
                content = re.sub(r'```json\s*|\s*```', '', content)
                data = json.loads(content)
                
                selected = data.get("selected_task_id")
                from_topic = data.get("from_topic", "").strip()
                to_topic = data.get("to_topic", "").strip()
                
                if selected in non_buffer_candidates:
                    self._consecutive_buffer_count = 0  # 重置计数器
                    self._last_bridge_hint = f"{from_topic}→{to_topic}" if from_topic else to_topic
                    self._current_turn_topic_set = True  # 🔥 标记本轮 Step1 已运行
                    print(f"[TaskPool] ✅ 强制选择任务: {selected}")
                    return selected
            except Exception as e:
                print(f"[TaskPool] ❌ 强制模式失败: {e}")
            
            # 兜底：随机选一个非闲聊任务
            self._consecutive_buffer_count = 0
            selected = random.choice(non_buffer_candidates)
            print(f"[TaskPool] 🎲 兜底选择: {selected}")
            return selected
        
        # ========== 正常模式：两步分离 ==========
        
        # 📍 第一步：LLM 完全自由选择话题（不给它看任务列表！）
        step1_system = """你是一个对话策略师，正在陪老人聊天。
根据对话上下文，选择一个最自然、最相关的话题作为接下来的聊天方向。

【输出格式】严格输出 JSON：
{
    "from_topic": "当前话题（2-4字）",
    "to_topic": "下一个话题（2-4字）",
    "reason": "选择理由"
}

【核心策略】
1. **顺藤摸瓜**：必须基于用户刚才说的话（Content）顺势延伸。
2. **逻辑桥梁**：如果必须切换话题，必须在 reason 里想好过渡逻辑。
3. **禁止生硬**：不要无厘头地跳到完全无关的话题（如从"吃饭"突然跳到"算术"），除非你能找到非常好的借口（如"算算饭钱"）。

【示例】
- 用户："最近老忘事。"
- 输出：{"from_topic": "忘事", "to_topic": "记忆", "reason": "用户提到忘事，顺势聊记忆力最自然"}

- 用户："刚吃完饺子。"
- 输出：{"from_topic": "吃饭", "to_topic": "算术", "reason": "从买菜算账过渡到算术"}
"""
        step1_user = f"""最近对话：
{recent_chat if recent_chat else '（刚开始聊天）'}

请选择下一个话题："""

        try:
            print(f"[TaskPool] 🧠 第一步：自由选择话题...")
            
            from src.llm.http_client_pool import get_siliconflow_chat_openai
            router_model = os.getenv("TASK_ROUTER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            llm = get_siliconflow_chat_openai(
                model=router_model,
                temperature=0.7,
                timeout=15,
                max_retries=1,
            )
            
            response1 = llm.invoke([
                {"role": "system", "content": step1_system},
                {"role": "user", "content": step1_user}
            ])
            
            content1 = response1.content.strip() if hasattr(response1, 'content') else str(response1).strip()
            content1 = re.sub(r'```json\s*|\s*```', '', content1)
            
            data1 = json.loads(content1)
            from_topic = data1.get("from_topic", "").strip()
            to_topic = data1.get("to_topic", "").strip()
            
            # 生成 bridge hint
            bridge_hint = f"{from_topic}→{to_topic}" if from_topic and to_topic else to_topic
            self._last_bridge_hint = bridge_hint
            self._current_turn_topic_set = True  # 🔥 标记本轮 Step1 已运行
            
            print(f"[TaskPool] 📝 话题过渡: '{bridge_hint}'")
            
            # 🔥 Step 2 已移除（移至后台并行执行）
            # 直接返回 buffer_chat，让 QuestionGen 根据 bridge_hint 生成自然对话
            # 后台线程会并行检查这个 bridge_hint 是否属于某个 Task
            if self._consecutive_buffer_count >= max_buffer_rounds and non_buffer_candidates:
                selected = random.choice(non_buffer_candidates)
                self._consecutive_buffer_count = 0
                print(f"[TaskPool] ⚠️ buffer 上限触发，兜底切到任务: {selected}")
                return selected

            self._consecutive_buffer_count += 1
            return "buffer_chat"
            
        except Exception as e:
            print(f"[TaskPool] ⚠️ 第一步选择失败: {e}")
            # 兜底：达到上限时优先返回非 buffer 任务，避免连续闲聊过多
            if self._consecutive_buffer_count >= max_buffer_rounds and non_buffer_candidates:
                selected = random.choice(non_buffer_candidates)
                self._consecutive_buffer_count = 0
                print(f"[TaskPool] 🎲 失败兜底切任务: {selected}")
                return selected
            return "buffer_chat"

    def _get_expected_answer_for_task(self, task_id: str, patient_profile: Dict) -> Optional[str]:
        """获取任务的期望答案（用于精准评估）"""
        from src.utils.location_service import get_realtime_context
        
        if not task_id or task_id in self.BUFFER_TASKS:
            return None
        
        ctx = get_realtime_context()
        time_info = ctx.get('time', {})
        location = ctx.get('location', {})
        
        if task_id == "orientation_time_weekday":
            return f"星期{time_info.get('weekday', '未知')}"
        elif task_id == "orientation_time_date_month_season":
            return f"{time_info.get('month')}月{time_info.get('day')}日，{time_info.get('season')}"
        elif task_id == "orientation_place_city_district":
            return f"{location.get('city', '')}，{location.get('district', '')}"
        elif task_id == "registration_3words":
            words = self.session_data.get('memory_words')
            return f"记住三个词：{words}" if words else None
        elif task_id == "recall_3words":
            words = self.session_data.get('memory_words')
            return f"三个词是：{words}" if words else None
        elif task_id == "attention_calc_life_math":
            return "连续减法结果"
        elif task_id == "language_naming_watch":
            return "手表"
        elif task_id == "language_naming_pencil":
            return "铅笔"
        elif task_id == "language_repetition_sentence":
            return "四十四只石狮子"
        elif task_id == "language_reading_close_eyes":
            return "闭上眼睛"
        elif task_id == "language_3step_action":
            return "完成三步指令"
        
        return None

    def _get_task_instruction(self, task_id: str) -> str:
        """获取任务的内部指令（给LLM看，不给用户）"""
        instructions = {
            "persona_collect_1": "自然地问问对方平时喜欢做什么、有什么爱好，收集个人信息",
            "persona_collect_2": "继续闲聊，了解对方的生活习惯、家庭情况等",
            "buffer_chat": "纯闲聊，聊聊天气、新闻、生活琐事，不做任何评估",
            "orientation_time_weekday": "自然地聊到今天星期几，比如'今天周几来着？'",
            "orientation_time_date_month_season": "聊聊今天几号、什么季节，比如结合天气或节日",
            "orientation_place_city_district": "聊聊现在在哪个城市、什么区，可以结合天气或当地特色",
            "registration_3words": "告诉对方三个词，请对方帮忙记一下，稍后会问",
            "attention_calc_life_math": "用生活场景出一道连续减法，比如买菜找零、数苹果",
            "language_naming_watch": "展示手表图片，问这是什么",
            "language_naming_pencil": "展示铅笔图片，问这是什么",
            "language_repetition_sentence": "请对方跟着说一句绕口令或俗语",
            "language_reading_close_eyes": "展示'请闭上眼睛'的图片，看对方是否照做",
            "language_3step_action": "给对方一个简单的三步指令，比如'拿起杯子、喝口水、放下'",
            "recall_3words": "问问刚才说的那三个词还记得吗",
        }
        return instructions.get(task_id, "自然地继续对话")

    def _generate_buffer_question(self, task_id: str, patient_profile: Dict, chat_history: List) -> str:
        """生成缓冲任务的闲聊问题 - 使用 LLM 生成自然多样的开场白"""
        
        # 根据任务类型给 LLM 不同的指引
        task_hints = {
            "persona_collect_1": "了解对方的兴趣爱好（喜欢做什么、看什么、玩什么）。⚠️注意：如果对方说没爱好，不要强行追问，可以问问年轻时喜欢干啥，或者直接聊别的",
            "persona_collect_2": "了解对方的生活习惯（作息、饮食、日常活动）。⚠️注意：如果对方否定（如不吃早饭、不起床），不要追问吃了啥，而是问原因或通过'那午饭呢'等方式自然切换话题",
            "buffer_chat": "顺着对方刚才的话题自然聊，不要突然切到无关内容"
        }
        default_hint = task_hints.get(task_id, "随便聊聊")
        
        # 🔥 提取最近对话（结构化列表而非字符串，让 LLM 有完整上下文）
        recent_history = None
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-6:]  # 最近3轮（6条消息）
        
        # 🔥 话题优先：如果 LLM 选了话题，用话题作为主要指引
        bridge_hint = getattr(self, '_last_bridge_hint', None)
        if bridge_hint:
            # 从 bridge_hint 提取目标话题 (格式: "A→B")
            to_topic = bridge_hint.split("→")[-1].strip() if "→" in bridge_hint else bridge_hint
            # 用话题作为主指令，buffer_chat 不再拼接固定参考文案，避免“近况+聊天气”冲突
            if task_id == "buffer_chat":
                hint = (
                    f"围绕「{to_topic}」这个话题自然聊天。"
                    f"顺着对方刚才的话往下聊，不要切到与「{to_topic}」无关的话题。"
                )
            else:
                hint = f"围绕「{to_topic}」这个话题自然聊天。可以参考：{default_hint}"
            # 🔥 注入防重复指令
            last_user_content = chat_history[-1].get('content', '') if chat_history else ''
            if last_user_content:
                hint += f"\n【注意】对方刚才说了「{last_user_content[:30]}...」，请顺着这话往下聊，不要重复问对方已经说过的信息。"
            
            print(f"[AgentFC] 🎯 话题优先: '{to_topic}' (任务 {task_id} 的指令作为参考)")
        else:
            hint = default_hint
        
        # 调用 QuestionGenerationTool（传入结构化历史）
        result_json = self.question_tool._run(
            dimension_name="闲聊",
            dimension_description=hint,
            patient_name=patient_profile.get('name'),
            patient_gender=patient_profile.get('gender'),
            patient_age=patient_profile.get('age'),
            conversation_history=recent_history,
            task_instruction=hint,
            avoid_questions=self._asked_questions,
            bridge_hint=bridge_hint,
        )
        
        try:
            result = json.loads(result_json)
            if result.get('success') and result.get('question'):
                return result['question']
        except:
            pass
        
        # 兜底：如果 LLM 失败，用简单的开场
        patient_name = patient_profile.get('name', '')
        greeting = f"{patient_name}，" if patient_name else ""
        return f"{greeting}您最近怎么样？"

    def _call_score_recording(
        self, session_id: str, dimension_id: str, eval_result: Dict,
        question: str, answer: str, max_score_override: Optional[int] = None
    ) -> Dict:
        """调用评分记录工具（定性 + MMSE定量），返回MMSE评分信息"""
        # 1. 定性评估记录
        self.score_tool._run(
            session_id=session_id,
            dimension_id=dimension_id,
            quality_level=eval_result.get('quality_level', 'fair'),
            cognitive_performance=eval_result.get('cognitive_performance', '正常'),
            question=question,
            answer=answer,
            evaluation_detail=eval_result.get('evaluation_detail', ''),
            action='save'
        )
        
        # 2. ⭐ MMSE定量评分（根据质量等级估算分数）
        mmse_score = self._convert_quality_to_mmse_score(
            dimension_id,
            eval_result.get('quality_level', 'fair'),
            eval_result.get('cognitive_performance', '正常'),
            max_score_override=max_score_override,
        )
        
        mmse_result = {
            'dimension_id': dimension_id,
            'score': mmse_score['score'],
            'max_score': mmse_score['max_score'],
            'total_score': 0
        }
        
        try:
            result_json = self.mmse_tool._run(
                session_id=session_id,
                dimension_id=dimension_id,
                score=mmse_score['score'],
                max_score=mmse_score['max_score'],
                question=question,
                answer=answer,
                evaluation_detail=f"质量等级: {eval_result.get('quality_level')} | {eval_result.get('evaluation_detail', '')}",
                action='save'
            )
            print(f"[AgentFC] ✅ MMSE评分: {dimension_id} = {mmse_score['score']}/{mmse_score['max_score']}分")
            
            # 从save的返回结果中获取总分
            save_result = json.loads(result_json)
            mmse_result['total_score'] = save_result.get('total_score', 0)
            
        except Exception as e:
            print(f"[AgentFC] ⚠️ MMSE评分失败: {e}")
        
        return mmse_result
    
    def _convert_quality_to_mmse_score(
        self,
        dimension_id: str,
        quality_level: str,
        cognitive_performance: str,
        max_score_override: Optional[int] = None,
    ) -> Dict:
        """
        将质量等级转换为MMSE分数
        
        映射逻辑：
        1. LLM按维度生成灵活问题（例如"您知道今天几号吗？"）
        2. AnswerEvaluationTool评估回答质量（excellent/good/fair/poor）
        3. 根据维度和质量等级映射到MMSE标准分数
        
        映射规则：
        - excellent: 完全正确，满分或接近满分
        - good: 基本正确，80%左右
        - fair: 部分正确或不确定，60%左右
        - poor: 明显错误或无法回答，30%或更低
        
        注：虽然问法灵活，但评估的认知能力点是固定的，所以可以映射到标准分数
        """
        # MMSE标准分值（国际通用）
        max_scores = {
            "orientation": 10,          # 时间5分+地点5分
            "registration": 3,          # 三个词即时记忆
            "attention_calculation": 5, # 连续减7，5次
            "recall": 3,                # 延迟回忆三个词
            "language": 8,              # 命名、复述、指令、阅读、书写
            "copy": 1                   # 临摹五边形
        }
        
        max_score = int(max_score_override) if max_score_override is not None else max_scores.get(dimension_id, 0)
        
        # 🎯 优化后的映射规则（更符合实际评估）
        if quality_level == "excellent":
            # 回答完全正确、清晰准确
            score = max_score  # 给满分
        elif quality_level == "good":
            # 回答基本正确，但可能有小瑕疵
            score = max(1, int(max_score * 0.80))  # 80%，至少1分
        elif quality_level == "fair":
            # 回答部分正确或不确定
            score = max(1, int(max_score * 0.50))  # 50%，至少1分
        else:  # poor
            # 回答明显错误或无法回答
            score = int(max_score * 0.20)  # 20%，可能是0分
        
        # 🔧 根据认知表现进一步微调（避免高估）
        if cognitive_performance == "重度异常":
            score = min(score, int(max_score * 0.3))
        elif cognitive_performance == "中度异常":
            score = min(score, int(max_score * 0.6))
        elif cognitive_performance == "轻度异常":
            score = min(score, int(max_score * 0.8))
        
        return {"score": score, "max_score": max_score}
    
    def _calculate_alzheimers_risk(self, total_score: int) -> Dict:
        """
        根据MMSE总分计算阿尔茨海默病风险
        
        MMSE分数解释（国际标准）：
        - 24-30分：认知功能正常
        - 18-23分：轻度认知障碍
        - 10-17分：中度痴呆
        - 0-9分：重度痴呆
        
        Returns:
            包含 risk_level, probability, description, recommendation 的字典
        """
        if total_score >= 24:
            return {
                'risk_level': '低风险',
                'probability': '低于10%',
                'score_range': '24-30分',
                'description': '认知功能正常，阿尔茨海默病风险较低',
                'recommendation': '建议保持良好的生活习惯，定期进行认知筛查（每1-2年）',
                'severity': 'normal'
            }
        elif total_score >= 18:
            return {
                'risk_level': '中度风险',
                'probability': '30-50%',
                'score_range': '18-23分',
                'description': '轻度认知障碍，存在转变为阿尔茨海默病的风险',
                'recommendation': '强烈建议到医院神经内科进行进一步检查，包括头部MRI、血液检查等',
                'severity': 'mild'
            }
        elif total_score >= 10:
            return {
                'risk_level': '高风险',
                'probability': '60-80%',
                'score_range': '10-17分',
                'description': '中度认知障碍，高度怀疑阿尔茨海默病或其他痴呆',
                'recommendation': '需要立即就医，进行全面的神经系统检查和评估，尽早干预治疗',
                'severity': 'moderate'
            }
        else:  # 0-9
            return {
                'risk_level': '极高风险',
                'probability': '高于85%',
                'score_range': '0-9分',
                'description': '重度认知障碍，极有可能已患阿尔茨海默病或其他重度痴呆',
                'recommendation': '紧急就医！需要立即到三甲医院神经内科或记忆门诊就诊，开始专业治疗和护理',
                'severity': 'severe'
            }
    
    def _generate_completion_message(
        self, 
        total_score: int, 
        risk_assessment: Dict, 
        patient_profile: Dict
    ) -> str:
        """
        生成评估完成后的温馨消息
        
        Args:
            total_score: MMSE总分
            risk_assessment: 风险评估结果
            patient_profile: 患者信息
            
        Returns:
            温馨的完成消息
        """
        patient_name = patient_profile.get('name', '')
        greeting = f"{patient_name}，" if patient_name else ""
        
        return (
            f"{greeting}今天聊得挺开心的，辛苦您啦！\n\n"
            f"咱们先歇一歇，您喝口水、活动活动。下次我再来跟您唠唠家常～"
        )
    
    # 🔥 性能优化：规则化查询模板，替代 LLM 调用（省 1-3 秒）
    _DIMENSION_QUERY_MAP: Dict[str, str] = {
        '定向力': '阿尔茨海默病 定向力 时间地点 认知评估',
        '即时记忆': '阿尔茨海默病 即时记忆 三词登记 认知评估',
        '注意力与计算': '阿尔茨海默病 注意力 计算 连续减法 认知评估',
        '延迟回忆': '阿尔茨海默病 延迟回忆 记忆 认知评估',
        '语言': '阿尔茨海默病 语言 命名复述 认知评估',
        '构图(临摹)': '阿尔茨海默病 构图 临摹 视觉空间 认知评估',
    }
    
    def _call_query_generation(self, dimension_name: str, conversation_history: List) -> Dict:
        """规则化查询生成（替代 LLM 调用，省 1-3 秒）"""
        query = self._DIMENSION_QUERY_MAP.get(
            dimension_name,
            f"阿尔茨海默病 {dimension_name} 认知评估"
        )
        return {'query': query, 'keywords': query.split()}
    
    def _call_knowledge_retrieval(self, query_result: Dict) -> Dict:
        """调用知识检索工具"""
        query = query_result.get('query', f"{self.current_dimension.get('name')} 认知评估")
        result_json = self.retrieval_tool._run(query=query, top_k=3)
        return json.loads(result_json)
    
    def _call_question_generation(
        self, dimension_name: str, knowledge_context: str,
        patient_profile: Dict, conversation_history: List, is_followup: bool,
        is_dimension_switch: bool = False,
        needs_encouragement: bool = False,
        resistance_info: Dict = None,
        task_instruction: Optional[str] = None,
        persona_hooks: Optional[List[str]] = None,
        must_include: Optional[List[str]] = None,
        patient_emotion: str = 'neutral'
    ) -> str:
        """调用问题生成工具"""
        # 提取患者信息
        patient_age = patient_profile.get('age')
        patient_education = patient_profile.get('education_years') or patient_profile.get('education')
        patient_name = patient_profile.get('name')
        patient_gender = patient_profile.get('gender', '女')
        
        # 🔥 修复：传入最近对话历史（最多6条），让LLM有上下文记忆
        history_info = None
        if conversation_history and len(conversation_history) > 0:
            # 传入最近的对话记录（包含 user 和 assistant），而非仅最后一条
            recent_history = conversation_history[-6:]  # 最近3轮（6条消息）
            history_info = recent_history  # 直接传 List[Dict]，question_tool 支持结构化历史

        result_json = self.question_tool._run(
            dimension_name=dimension_name,
            dimension_description="",
            knowledge_context=knowledge_context,
            patient_age=patient_age,
            patient_education=patient_education,
            patient_name=patient_name,
            patient_gender=patient_gender,
            conversation_history=history_info,
            patient_emotion=patient_emotion,
            task_instruction=task_instruction,
            persona_hooks=persona_hooks,
            must_include=must_include,
            avoid_questions=self._asked_questions,
            bridge_hint=self._last_bridge_hint,  # 🔥 传入过渡提示
        )
        result = json.loads(result_json)
        return result.get('question', '请继续')

    def _extract_persona_hooks(self, patient_profile: Dict, chat_history: List) -> List[str]:
        hooks = []
        for key in [
            'hobby', 'hobbies', 'interest', 'interests', 'occupation', 'job', 'hometown',
            'city', 'district', 'nickname'
        ]:
            val = patient_profile.get(key)
            if isinstance(val, str) and val.strip():
                hooks.append(val.strip())
            elif isinstance(val, list):
                hooks.extend([str(x).strip() for x in val if str(x).strip()])

        if chat_history:
            last_user = None
            for msg in reversed(chat_history):
                if msg.get('role') == 'user':
                    last_user = (msg.get('content') or '').strip()
                    break
            if last_user and len(last_user) <= 20:
                hooks.append(last_user)

        # 去重
        dedup = []
        seen = set()
        for h in hooks:
            if h not in seen:
                dedup.append(h)
                seen.add(h)
        return dedup[:5]

    def _get_must_include_for_task(self, task_id: str) -> Optional[List[str]]:
        if task_id == 'orientation_time_weekday':
            return ['星期']
        if task_id == 'orientation_time_date_month_season':
            return ['几月', '几号', '季节']
        if task_id == 'orientation_place_city_district':
            return ['城市', '区']
        if task_id == 'language_reading_close_eyes':
            return ['读', '照着做']
        if task_id == 'language_3step_action':
            return ['先', '再', '最后']
        return None
    
    def _check_and_display_image(self, question: str, session_id: str) -> Dict:
        """
        检测问题是否需要展示图片
        
        根据问题内容判断是否是命名/阅读任务，并返回相应的图片展示指令
        """
        question_lower = question.lower()
        
        # 检测关键词
        naming_keywords = ['这是什么', '叫什么', '名字', '物品', '东西']
        reading_keywords = ['闭上眼睛', '照着做', '按照', '文字', '看到']
        
        # 判断是否是命名任务
        if any(keyword in question for keyword in naming_keywords):
            # 随机选择手表或铅笔（或者根据评估次数轮换）
            # 这里简化处理，根据session_id哈希决定
            import hashlib
            hash_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
            image_id = 'watch' if hash_val % 2 == 0 else 'pencil'
            title = "请看下面的图片，这是什么东西？"
            
            result_json = self.image_tool._run(
                image_id=image_id,
                title=title,
                action='show'
            )
            result = json.loads(result_json)
            
            return {
                'should_display': result.get('success', False),
                'image_id': image_id,
                'display_command': result.get('display_command')
            }
        
        # 判断是否是阅读任务
        elif any(keyword in question for keyword in reading_keywords):
            title = "请照着上面的文字做"
            
            result_json = self.image_tool._run(
                image_id='close_eyes',
                title=title,
                action='show'
            )
            result = json.loads(result_json)
            
            return {
                'should_display': result.get('success', False),
                'image_id': 'close_eyes',
                'display_command': result.get('display_command')
            }
        
        # 不需要展示图片
        return {
            'should_display': False,
            'image_id': None,
            'display_command': None
        }
    
    def _call_conversation_storage(self, session_id: str, user_input: str, generated_question: str):
        """调用对话存储工具"""
        # 构建 turn_data
        turn_data = {
            "user_question": user_input,
            "assistant_response": generated_question,
            "dimension_id": self.current_dimension.get('id'),
            "dimension_name": self.current_dimension.get('name')
        }
        
        self.storage_tool._run(
            session_id=session_id,
            action='save_turn',
            turn_data=json.dumps(turn_data, ensure_ascii=False)
        )
    
    def _check_user_willing_to_continue(self, user_input: str) -> bool:
        """
        检查用户是否表示愿意继续评估
        这个检查应该在用户回复"引导语"之后进行
        """
        positive_keywords = [
            "好", "行", "可以", "愿意", "试试", "继续", "开始",
            "没问题", "OK", "ok", "嗯", "是的"
        ]
        negative_keywords = [
            "不", "别", "算了", "不想", "不愿意", "不行", 
            "累了", "休息", "等会", "以后"
        ]
        
        # 简单的关键词匹配
        for keyword in positive_keywords:
            if keyword in user_input:
                # 再检查是否有否定词
                has_negative = any(neg in user_input for neg in negative_keywords)
                if not has_negative:
                    return True
        
        return False
    
    def _check_mmse_complete(self, session_id: str) -> bool:
        """
        检查 MMSE 是否已完成所有必要维度
        返回 True 表示可以结束对话
        """
        # 核心评估任务（不含缓冲任务）
        core_tasks = [t for t in self.REQUIRED_TASKS if t not in self.BUFFER_TASKS]
        completed_core = self._task_done & set(core_tasks)
        
        coverage = len(completed_core) / len(core_tasks) if core_tasks else 1.0
        
        print(f"[AgentFC] 📊 MMSE完成度: {len(completed_core)}/{len(core_tasks)} ({coverage:.0%})")
        print(f"[AgentFC]   已完成: {completed_core}")
        print(f"[AgentFC]   未完成: {set(core_tasks) - completed_core}")
        
        # 至少完成 80% 的核心任务才算完成
        return coverage >= 1
    
    def _generate_soft_continuation(
        self, patient_profile: Dict, user_answer: str, chat_history: List
    ) -> str:
        """
        当用户想结束但 MMSE 未完成时，用 LLM 生成自然的过渡话语
        """
        patient_name = patient_profile.get('name', '')
        
        try:
            from src.llm.model_pool import get_pooled_llm
            llm = get_pooled_llm(pool_key='7b_complex')
            
            # 获取最近几轮对话作为上下文
            recent_history = chat_history[-6:] if chat_history else []
            history_text = "\n".join([
                f"{'我' if h.get('role') == 'assistant' else '老人'}: {h.get('content', '')[:50]}"
                for h in recent_history
            ])
            
            prompt = f"""你是{patient_name or '老人'}的晚辈，正在陪他/她聊天。
老人刚才说想结束对话：「{user_answer}」

但你想继续聊一会儿。请生成一句自然的回复，要求：
1. 先温柔地回应老人（"好嘞"/"行"/"没事"等）
2. 然后自然地抛出一个新话题继续聊（不要太刻意）
3. 话题可以是：天气、吃饭、身体、家人、兴趣爱好等
4. 口语化、接地气，不要太长（20-40字）
5. 不要问"您还有什么想聊的吗"这种刻意的话

最近对话：
{history_text}

直接输出回复，不要加引号："""
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            result = result.strip('"').strip("'").strip('「').strip('」')
            
            if result and 10 < len(result) < 80:
                print(f"[AgentFC] 🎲 LLM 生成过渡语: {result}")
                return result
        except Exception as e:
            print(f"[AgentFC] ⚠️ LLM 生成过渡语失败: {e}")
        
        # LLM 失败时的备选模板
        import random
        templates = [
            f"{patient_name}，好嘞，那您歇会儿。诶，您今天身体咋样？",
            f"行，{patient_name}您休息。对了，最近睡眠怎么样？",
            f"好，{patient_name}您先歇着。说起来，您家那边天气怎么样？",
            f"没事{patient_name}，您累了就歇。您孩子最近来看您了吗？",
        ]
        return random.choice(templates)
    
    def _generate_assessment_question(
        self, dimension_name: str, patient_profile: Dict, 
        chat_history: List, start_time: float,
        task_id: Optional[str] = None  # 🔥 新增：任务ID
    ) -> Dict[str, Any]:
        """
        生成评估问题（从闲聊回到评估）
        重新走正常的问题生成流程
        """
        # 生成下一个问题（正常流程）
        print("[AgentFC] 🤔 回到评估：生成下一个问题...")
        
        # 🔥 获取任务指令（如果有任务ID）
        task_instruction = None
        if task_id:
            task_instruction = self._get_task_instruction(task_id)
            print(f"[AgentFC] 📝 任务指令: {task_instruction[:50] if task_instruction else 'None'}...")
        
        # 生成检索查询
        query_result = self._call_query_generation(dimension_name, chat_history)
        
        # 检索知识
        retrieval_result = self._call_knowledge_retrieval(query_result)
        
        # 生成问题（从闲聊回到评估，标记为维度切换以便生成自然过渡）
        next_question = self._call_question_generation(
            dimension_name,
            retrieval_result.get('knowledge_context', ''),
            patient_profile,
            chat_history,
            is_followup=False,
            is_dimension_switch=True,  # 🔥 标记为维度切换，触发过渡语生成
            needs_encouragement=False,
            resistance_info=None,
            task_instruction=task_instruction,  # 🔥 传入任务指令
        )
        
        # 问题生成已经包含对用户回答的回应，不需要额外过渡语
        
        return {
            'output': next_question,
            'response': next_question,
            'is_comfort_mode': False,
            'returned_to_assessment': True,
            'dimension': dimension_name,
            'total_time': time.time() - start_time
        }
    
    def _call_natural_transition(
        self, user_answer: str, dimension_name: str, 
        patient_profile: Dict, chat_history: List, current_emotion: str
    ) -> str:
        """调用工具生成自然过渡回应"""
        result_json = self.question_tool.generate_natural_transition(
            user_answer=user_answer,
            dimension_name=dimension_name,
            patient_name=patient_profile.get('name'),
            patient_gender=patient_profile.get('gender'),
            patient_age=patient_profile.get('age'),
            chat_history=chat_history,
            current_emotion=current_emotion,
        )
        result = json.loads(result_json)
        return result.get('transition', '嗯，您说得对。')
    
    def get_current_dimension(self) -> Dict[str, str]:
        """获取当前评估维度"""
        return self.current_dimension
    
    def set_dimension(self, dimension_id: str):
        """设置当前评估维度"""
        for dim in MMSE_DIMENSIONS:
            if dim['id'] == dimension_id:
                self.current_dimension = dim
                break
    async def _background_global_analysis(self, topic: str, chat_history: List[Dict]):
        """
        后台全能侦探（与 TTS 并行执行）：
        1. 话题映射：判断当前过渡到的 Topic (如 '周末活动') 是否属于某个待完成任务
        2. 全局覆盖：扫描整个历史，看是否有其它任务被隐式完成
        """
        print(f"[AgentFC] 🕵️‍♀️ 后台侦探启动：Topic='{topic}'")
        
        # 1. 话题映射 (Topic Mapping)
        # 🔥 这是从 _llm_select_task 的 Step2 拆出来的，与 TTS 并行执行以节省时间
        if topic:
            try:
                undone_tasks = [t for t in self.REQUIRED_TASKS if t not in self._task_done]
                print(f"[AgentFC-Background] 🔍 待完成任务列表: {undone_tasks}")
                
                if not undone_tasks:
                    print(f"[AgentFC-Background] ✅ 所有任务已完成，跳过映射")
                else:
                    # 🔥 先用关键词预筛（快速，不需要 LLM）
                    _TOPIC_TASK_KEYWORDS = {
                        "orientation_time_weekday": ["星期", "周几", "今天"],
                        "orientation_time_date_month_season": ["日期", "几月", "季节", "几号"],
                        "orientation_place_city_district": ["城市", "住哪", "地方", "地址", "区"],
                        "persona_collect_1": ["爱好", "兴趣", "喜欢"],
                        "persona_collect_2": ["习惯", "作息", "起床", "吃"],
                        "registration_3words": ["记忆", "记词", "记住"],
                        "recall_3words": ["回忆", "想起", "刚才的词"],
                        "attention_calc_life_math": ["算术", "计算", "减法", "算账", "算钱"],
                        "language_naming_watch": ["手表", "时间工具"],
                        "language_naming_pencil": ["铅笔", "写字工具"],
                        "language_repetition_sentence": ["复述", "重复"],
                        "language_reading_close_eyes": ["读字", "闭眼"],
                        "language_3step_action": ["动作", "指令", "步骤"],
                    }
                    
                    # 从 topic 中提取（topic 格式可能是 "吃饭→算术" 或 "算术"）
                    topic_text = topic.split("→")[-1].strip() if "→" in topic else topic.strip()
                    
                    mapped_task = None
                    for task_id in undone_tasks:
                        keywords = _TOPIC_TASK_KEYWORDS.get(task_id, [])
                        if any(kw in topic_text for kw in keywords):
                            mapped_task = task_id
                            print(f"[AgentFC-Background] ⚡ 关键词命中：'{topic_text}' -> {task_id}")
                            break
                    
                    if mapped_task:
                        # 关键词直接命中，无需 LLM
                        self._precomputed_next_task = mapped_task
                        self._consecutive_buffer_count = 0  # 重置计数器
                        print(f"[AgentFC-Background] 🎯 Topic映射完成（关键词）: {mapped_task}")
                    else:
                        # 关键词未命中，使用 LLM 判断
                        from src.llm.http_client_pool import get_siliconflow_chat_openai
                        background_model = os.getenv(
                            "BACKGROUND_ANALYSIS_MODEL",
                            os.getenv("TASK_ROUTER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
                        )
                        llm = get_siliconflow_chat_openai(
                            model=background_model,
                            temperature=0.3,
                            timeout=10,
                            max_retries=1,
                        )
                        
                        prompt_topic = f"""当前对话正过渡到话题：「{topic_text}」。
请判断这个话题是否直接对应以下待完成任务之一：
{', '.join(undone_tasks)}

如果对应，输出任务ID。
否则输出 None。
只输出结果，不要解释。"""
                        res = await llm.ainvoke([{"role": "user", "content": prompt_topic}])
                        res_content = res.content.strip()
                        
                        if res_content in undone_tasks:
                            print(f"[AgentFC-Background] 🎯 Topic命中（LLM）：'{topic_text}' -> {res_content}")
                            self._precomputed_next_task = res_content
                            self._consecutive_buffer_count = 0  # 重置计数器
                        else:
                            print(f"[AgentFC-Background] ℹ️ Topic未命中任何任务: '{topic_text}' -> '{res_content}'")
                            
            except Exception as e:
                print(f"[AgentFC-Background] ⚠️ Topic映射失败: {e}")

        # 2. 全局历史覆盖 (Global History Coverage)
        try:
            undone = [t for t in self.REQUIRED_TASKS if t not in self._task_done]
            if not undone:
                return

            # 构建最近历史文本
            history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in chat_history[-10:]]) # 限制长度
            
            prompt_global = f"""请审查以下对话历史，判断用户是否**已经回答**了以下任务的问题：
待完成任务：{', '.join(undone)}

任务说明：
- orientation_time_weekday: 说了星期几
- orientation_place_city: 说了所在城市
- persona_collect: 说了兴趣或习惯

输出格式：JSON list，如 ["task_id_1", "task_id_2"]。如果无，输出 []。

对话历史：
{history_text}
"""
            from src.llm.http_client_pool import get_siliconflow_chat_openai
            background_model = os.getenv(
                "BACKGROUND_ANALYSIS_MODEL",
                os.getenv("TASK_ROUTER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            )
            global_llm = get_siliconflow_chat_openai(
                model=background_model,
                temperature=0.3,
                timeout=10,
                max_retries=1,
            )
            res = await global_llm.ainvoke([{"role": "user", "content": prompt_global}])
            content = res.content.strip()
            
            import re
            import json
            content = re.sub(r'```json\s*|\s*```', '', content)
            
            # 🔥 改进的JSON解析：尝试提取JSON数组
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                hit_tasks = json.loads(json_match.group())
            elif content.strip() == '[]' or content.strip().lower() == 'none' or not content.strip():
                hit_tasks = []
            else:
                # 尝试直接解析（兼容旧格式）
                hit_tasks = json.loads(content)
            
            if isinstance(hit_tasks, list) and hit_tasks:
                # 限制每轮最多自动完成 1 个任务，避免误判导致进度跳跃
                try:
                    max_auto_done = int(os.getenv("GLOBAL_SCAN_MAX_AUTO_DONE", "1"))
                except ValueError:
                    max_auto_done = 1
                max_auto_done = max(0, max_auto_done)

                unique_hits = []
                for task in hit_tasks:
                    if task in undone and task not in unique_hits:
                        unique_hits.append(task)

                capped_hits = unique_hits[:max_auto_done] if max_auto_done > 0 else []
                if capped_hits:
                    print(f"[AgentFC-Background] 🌍 全局扫描命中: {unique_hits}")
                    if len(unique_hits) > len(capped_hits):
                        print(
                            f"[AgentFC-Background] ℹ️ 本轮仅自动完成 {len(capped_hits)} 个任务，"
                            f"其余延后确认: {unique_hits[len(capped_hits):]}"
                        )
                else:
                    print(f"[AgentFC] 🌊 全局扫描无可自动确认任务")

                for done_task in capped_hits:
                    if done_task in undone:
                        # 标记任务完成
                        self._task_done.add(done_task)
                        print(f"[AgentFC-Background] ✅ 自动标记任务完成: {done_task}")
                        
                        # 重要：同时记录一个满分评分，防止空缺
                        try:
                            # 尝试找到对应的维度ID
                            task_cfg = self.TASK_CONFIG.get(done_task, {})
                            dim_id = task_cfg.get('dimension_id')
                            max_points = task_cfg.get('max_points', 1)
                            
                            if dim_id:
                                # 1. 异步记录定性评分
                                self.score_tool._run(
                                    session_id=self.session_data.get('session_id', 'unknown'),
                                    dimension_id=dim_id,
                                    quality_level="good",
                                    cognitive_performance="隐式回答正确",
                                    question="(后台全局扫描)",
                                    answer="(历史对话隐式包含)",
                                    evaluation_detail="全局历史扫描发现用户已回答此问题",
                                    action="save"
                                )
                                
                                # 2. 🔥 必须调用 MMSE 定量评分，否则报告里是0分
                                mmse_res = self.mmse_tool._run(
                                    session_id=self.session_data.get('session_id', 'unknown'),
                                    dimension_id=dim_id,
                                    score=max_points,
                                    max_score=max_points,
                                    action="update"
                                )
                                print(f"[AgentFC-Background] 📊 MMSE自动记分: {done_task} (+{max_points}) -> {mmse_res}")
                                
                        except Exception as e_score:
                             print(f"[AgentFC-Background] ⚠️ 评分记录失败: {e_score}")
            else:
                print(f"[AgentFC] 🌊 全局扫描无新发现")
            
        except Exception as e:
             print(f"[AgentFC] ⚠️ 全局扫描失败: {e}")
