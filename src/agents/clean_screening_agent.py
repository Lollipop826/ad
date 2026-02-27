"""
干净的筛查Agent - 严格遵循工具化设计原则
Agent只负责任务调度和流程编排，具体逻辑交给tools处理
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from src.tools.agent_tools import (
    QuestionGenerationTool,
    AnswerEvaluationTool, 
    ResistanceDetectionTool,
    ComfortResponseTool,
    ConversationStorageTool,
    StandardQuestionTool
)
from src.domain.dimensions import MMSE_DIMENSIONS


class CleanScreeningAgent:
    """
    干净的筛查Agent
    - Agent: 专注任务调度和流程编排
    - Tools: 处理具体的问题生成、评估等逻辑
    """
    
    # 灵活化任务配置
    TASK_CONFIG: Dict[str, Dict[str, Any]] = {
        # 缓冲任务
        "persona_collect_1": {"dimension_id": None, "max_points": 0, "type": "buffer"},
        "persona_collect_2": {"dimension_id": None, "max_points": 0, "type": "buffer"},
        "buffer_chat": {"dimension_id": None, "max_points": 0, "type": "buffer"},
        
        # 灵活化认知评估任务
        "orientation_assessment": {"dimension_id": "orientation", "max_points": 5, "type": "task", "flexible": True},
        "registration_3words": {"dimension_id": "registration", "max_points": 3, "type": "task"},
        "attention_calculation": {"dimension_id": "attention_calculation", "max_points": 5, "type": "task", "flexible": True},
        "language_assessment": {"dimension_id": "language", "max_points": 7, "type": "task", "flexible": True},
        "recall_3words": {"dimension_id": "recall", "max_points": 3, "type": "task"},
    }
    
    REQUIRED_TASKS = [
        "persona_collect_1",
        "persona_collect_2", 
        "orientation_assessment",
        "registration_3words",
        "attention_calculation",
        "language_assessment",
        "recall_3words",
    ]
    
    BUFFER_TASKS = {"persona_collect_1", "persona_collect_2", "buffer_chat"}
    
    def __init__(self, use_local: bool = True):
        self.use_local = use_local
        
        # 初始化工具
        self.question_tool = QuestionGenerationTool(use_local=use_local)
        self.eval_tool = AnswerEvaluationTool(use_local=use_local)
        self.resistance_tool = ResistanceDetectionTool(use_local=use_local)
        self.comfort_tool = ComfortResponseTool(use_local=use_local)
        self.storage_tool = ConversationStorageTool()
        self.standard_tool = StandardQuestionTool(use_local=use_local)
        
        # 任务池状态
        self._task_done = set()
        self._last_task_id = None
        self._registration_ts = None
        self._turn_counter = 0
        self._task_cooldown_until = {}
        self._pending_consent_task_id = None
        
        # 对话状态管理（从原Agent学习）
        self.is_in_comfort_mode = False  # 是否在闲聊模式
        self.comfort_turn_count = 0      # 闲聊轮次计数
        self.has_started_assessment = False  # 是否已开始正式评估
        
        self.session_data = {}
        self._active_session_id = None
        
    def process_turn(
        self,
        user_input: str,
        dimension: Optional[Dict[str, Any]] = None,  # 兼容原Agent参数
        session_id: str = None,
        patient_profile: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        current_emotion: str = 'neutral'
    ) -> Dict[str, Any]:
        """
        处理一轮对话
        严格遵循工具化设计：Agent只做调度，Tools做具体处理
        """
        try:
            start_time = time.time()
            
            # 🆕 1. 处理默认参数（学习原Agent）
            if session_id is None:
                import uuid
                session_id = f"session_{uuid.uuid4().hex[:8]}"
            self.session_id = session_id
            
            if getattr(self, "_active_session_id", None) != session_id:
                self._reset_task_pool(session_id)
                
            if patient_profile is None:
                patient_profile = {}
            if chat_history is None:
                chat_history = []
                
            self.session_data['chat_history'] = chat_history
            
            # 🆕 2. 设置当前维度（兼容原Agent）
            if dimension is not None:
                self.current_dimension = dimension
            dimension_id = getattr(self, 'current_dimension', {}).get('id', 'orientation')
            dimension_name = getattr(self, 'current_dimension', {}).get('name', '定向力')
            
            # 🆕 3. 提取最后一个医生问题
            doctor_question = "请开始评估"
            for msg in reversed(chat_history):
                if msg.get("role") == "assistant":
                    doctor_question = msg.get("content", "请开始评估")
                    break
            
            user_answer = user_input  # 统一变量名
            
            self._turn_counter += 1
            
            # 🆕 4. 详细调试日志（学习原Agent）
            print(f"\n{'='*60}")
            print(f"[CleanAgent] 🚀 开始处理对话")
            print(f"[CleanAgent] 📅 维度: {dimension_name} ({dimension_id})")
            print(f"[CleanAgent] 😊 情绪: {current_emotion}")
            print(f"[CleanAgent] 👤 医生: {doctor_question[:50]}...")
            print(f"[CleanAgent] 🗣️ 患者: {user_answer}")
            print(f"{'='*60}\n")
            
            # 🆕 步骤1: 对话状态管理（学习原Agent逻辑）
            if self.is_in_comfort_mode:
                return self._handle_comfort_mode(user_input, patient_profile, chat_history, current_emotion)
            
            # 🆕 步骤2: 检测是否需要进入闲聊模式
            if not self.has_started_assessment and self._turn_counter > 1:
                # 检测用户抵抗或不配合
                resistance_check = self._check_need_comfort_mode(user_input, current_emotion)
                if resistance_check:
                    return resistance_check
                    
            # 🆕 步骤3: 智能评估策略（学习原Agent复杂逻辑）
            eval_result = {}
            if self._last_task_id:
                eval_result = self._smart_evaluate_answer(
                    user_answer, doctor_question, self._last_task_id, 
                    dimension_id, patient_profile, chat_history
                )
                
            # 步骤4: 选择下一个任务
            next_task_id = self._select_next_task()
            if next_task_id is None:
                return self._generate_completion_response(patient_profile)
                
            # 步骤5: 生成问题 (完全交给工具处理)
            next_question = self._generate_question_via_tools(
                next_task_id, patient_profile, chat_history, eval_result
            )
            
            # 步骤4: 更新状态
            self._update_task_state(next_task_id, eval_result)
            
            # 步骤5: 存储对话
            if session_id:
                self._store_conversation(session_id, user_input, next_question, chat_history)
                
            return {
                'response': next_question,
                'task_id': next_task_id,
                'evaluation': eval_result,
                'session_data': self.session_data
            }
            
        except Exception as e:
            print(f"❌ [CleanAgent] 处理失败: {e}")
            return {'response': "抱歉，我需要重新理解一下您的回答。能否再说一遍？", 'error': str(e)}
    
    def _evaluate_answer(self, user_input: str, task_id: str, patient_profile: Dict, chat_history: List) -> Dict:
        """使用工具评估用户回答"""
        task_cfg = self.TASK_CONFIG.get(task_id, {})
        dimension_id = task_cfg.get('dimension_id')
        
        if not dimension_id:
            return {}
            
        try:
            # 并发执行抵抗检测和答案评估
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_resistance = executor.submit(
                    self.resistance_tool._run, 
                    question="上一个问题", 
                    answer=user_input
                )
                future_eval = executor.submit(
                    self.eval_tool._run,
                    dimension_id=dimension_id,
                    question="上一个问题",
                    answer=user_input,
                    expected_answer="",
                    patient_profile=patient_profile
                )
                
                resistance_result = json.loads(future_resistance.result())
                eval_result = json.loads(future_eval.result())
                
                return {
                    'is_correct': eval_result.get('is_correct', False),
                    'quality_level': eval_result.get('quality_level', 'poor'),
                    'is_resistant': resistance_result.get('is_resistant', False),
                    'resistance_category': resistance_result.get('category', 'none')
                }
                
        except Exception as e:
            print(f"⚠️ [CleanAgent] 评估失败: {e}")
            return {'is_correct': False, 'quality_level': 'poor'}
    
    def _select_next_task(self) -> Optional[str]:
        """任务池调度：选择下一个要执行的任务"""
        candidates = []
        now = time.time()
        
        for task_id in self.REQUIRED_TASKS:
            if task_id in self._task_done:
                continue
                
            # 硬约束：recall需要等registration完成后2分钟
            if task_id == "recall_3words":
                if "registration_3words" not in self._task_done:
                    continue
                if self._registration_ts and (now - self._registration_ts) < 120:
                    continue
                    
            # 硬约束：registration需要在persona收集后
            if task_id == "registration_3words":
                if "persona_collect_1" not in self._task_done or "persona_collect_2" not in self._task_done:
                    continue
                    
            candidates.append(task_id)
        
        # 如果没有候选任务，检查是否需要等待
        if not candidates:
            remaining = set(self.REQUIRED_TASKS) - self._task_done
            if "recall_3words" in remaining and self._registration_ts:
                wait_time = 120 - (now - self._registration_ts)
                if wait_time > 0:
                    return "buffer_chat"  # 等待时插入缓冲
            return None  # 所有任务完成
            
        # 简单调度：返回第一个候选任务
        return candidates[0]
    
    def _generate_question_via_tools(self, task_id: str, patient_profile: Dict, chat_history: List, eval_result: Dict) -> str:
        """完全通过工具生成问题，Agent不含任何问题生成逻辑"""
        
        if task_id in self.BUFFER_TASKS:
            # 缓冲任务：使用QuestionGenerationTool
            return self._call_buffer_question_tool(task_id, patient_profile, chat_history)
            
        elif task_id in {"registration_3words", "recall_3words"}:
            # 标准化任务：使用StandardQuestionTool
            return self._call_standard_question_tool(task_id, patient_profile)
            
        else:
            # 灵活任务：使用QuestionGenerationTool
            return self._call_flexible_question_tool(task_id, patient_profile, chat_history, eval_result)
    
    def _call_buffer_question_tool(self, task_id: str, patient_profile: Dict, chat_history: List) -> str:
        """调用工具生成缓冲问题"""
        task_hints = {
            "persona_collect_1": "了解对方的兴趣爱好",
            "persona_collect_2": "了解对方的生活习惯", 
            "buffer_chat": "随便聊聊，自然过渡"
        }
        
        result_json = self.question_tool._run(
            dimension_name="闲聊",
            dimension_description=task_hints.get(task_id, "随便聊聊"),
            patient_name=patient_profile.get('name'),
            patient_gender=patient_profile.get('gender'),
            patient_age=patient_profile.get('age'),
            conversation_history=self._format_chat_history(chat_history)
        )
        
        try:
            result = json.loads(result_json)
            if result.get('success'):
                question = result.get('question', '您最近怎么样？')
                print(f"🔍 [CleanAgent] 工具返回问题: {question}")
                return question
        except Exception as e:
            print(f"⚠️ [CleanAgent] JSON解析失败: {e}")
            print(f"🔍 [CleanAgent] 原始返回: {result_json[:200]}...")
            
        print("🔍 [CleanAgent] 使用兜底问题")
        return '您最近怎么样？'  # 兜底
    
    def _call_standard_question_tool(self, task_id: str, patient_profile: Dict) -> str:
        """调用StandardQuestionTool处理标准化任务"""
        task_cfg = self.TASK_CONFIG.get(task_id, {})
        dimension_id = task_cfg.get('dimension_id')
        
        result_json = self.standard_tool._run(
            dimension_id=dimension_id,
            is_dimension_switch=True,
            memory_words=self.session_data.get('memory_words'),
            patient_name=patient_profile.get('name')
        )
        
        try:
            result = json.loads(result_json)
            if result.get('has_standard_question'):
                # 保存记忆词等状态
                if result.get('memory_words'):
                    self.session_data['memory_words'] = result['memory_words']
                return result.get('question', '请配合我做个小测试。')
        except:
            pass
            
        return '请配合我做个小测试。'  # 兜底
    
    def _call_flexible_question_tool(self, task_id: str, patient_profile: Dict, chat_history: List, eval_result: Dict) -> str:
        """调用工具生成灵活的维度评估问题"""
        task_cfg = self.TASK_CONFIG.get(task_id, {})
        dimension_id = task_cfg.get('dimension_id')
        
        # 获取维度名称
        dimension_name = "认知评估"
        for dim in MMSE_DIMENSIONS:
            if dim.get('id') == dimension_id:
                dimension_name = dim.get('name', dimension_id)
                break
        
        result_json = self.question_tool._run(
            dimension_name=dimension_name,
            dimension_description="",
            patient_name=patient_profile.get('name'),
            patient_gender=patient_profile.get('gender'), 
            patient_age=patient_profile.get('age'),
            conversation_history=self._format_chat_history(chat_history),
            task_instruction=f"进行{dimension_name}相关的自然评估"
        )
        
        try:
            result = json.loads(result_json)
            if result.get('success'):
                return result.get('question', f'我们聊聊{dimension_name}相关的话题吧。')
        except:
            pass
            
        return f'我们聊聊{dimension_name}相关的话题吧。'  # 兜底
    
    def _format_chat_history(self, chat_history: List) -> str:
        """格式化对话历史"""
        if not chat_history:
            return ""
            
        history_text = ""
        for msg in chat_history[-4:]:  # 最近4轮
            role = "对方" if msg.get('role') == 'user' else "你"
            content = msg.get('content', '')[:50]
            if content:
                history_text += f"{role}：{content}\n"
        return history_text
    
    def _update_task_state(self, task_id: str, eval_result: Dict):
        """更新任务池状态"""
        # 标记任务完成
        if self._last_task_id:
            self._task_done.add(self._last_task_id)
            
            # 记录registration时间戳
            if self._last_task_id == "registration_3words":
                self._registration_ts = time.time()
                
        self._last_task_id = task_id
    
    def _store_conversation(self, session_id: str, user_input: str, agent_response: str, chat_history: List):
        """存储对话历史"""
        try:
            self.storage_tool._run(
                session_id=session_id,
                user_message=user_input,
                agent_message=agent_response,
                message_type="assessment"
            )
        except Exception as e:
            print(f"⚠️ [CleanAgent] 存储对话失败: {e}")
    
    def _handle_comfort_mode(self, user_input: str, patient_profile: Dict, chat_history: List, current_emotion: str) -> Dict:
        """处理闲聊模式（学习原Agent逻辑）"""
        print(f"[CleanAgent] 💬 当前处于闲聊模式 (第 {self.comfort_turn_count + 1}/3 轮)")
        self.comfort_turn_count += 1
        
        # 判断是否该尝试引导回评估
        positive_emotions = ['happy', 'excited', 'joy', 'positive', 'neutral']
        is_positive = current_emotion in positive_emotions
        
        ask_to_continue = False
        if self.comfort_turn_count >= 3 and is_positive:
            print(f"[CleanAgent] 💡 已闲聊 {self.comfort_turn_count} 轮且情绪积极，尝试引导回评估")
            ask_to_continue = True
            self.is_in_comfort_mode = False
            self.has_started_assessment = True
        elif self.comfort_turn_count >= 3:
            print(f"[CleanAgent] 💬 已闲聊 {self.comfort_turn_count} 轮，但情绪不佳，继续闲聊")
        
        # 使用ComfortResponseTool生成闲聊回复
        emotion_to_category = {
            'sad': 'fatigue', 'angry': 'hostility', 'fear': 'avoidance'
        }
        category = emotion_to_category.get(current_emotion, 'boredom')
        
        comfort_result = self.comfort_tool._run(
            resistance_category=category,
            user_response=user_input,
            dimension_id="comfort",
            ask_to_continue=ask_to_continue
        )
        
        try:
            comfort_data = json.loads(comfort_result)
            response = comfort_data.get('response', '我理解您的感受，我们先聊聊别的。')
            
            if ask_to_continue:
                # 如果要引导回评估，选择下一个真正的任务
                next_task_id = self._select_next_task()
                if next_task_id:
                    self._last_task_id = next_task_id
                
            return {
                'response': response,
                'task_id': 'comfort',
                'evaluation': {},
                'session_data': self.session_data
            }
        except:
            return {
                'response': '我理解您的感受，我们先聊聊别的。',
                'task_id': 'comfort',
                'evaluation': {},
                'session_data': self.session_data
            }
    
    def _check_need_comfort_mode(self, user_input: str, current_emotion: str) -> Optional[Dict]:
        """检测是否需要进入闲聊模式"""
        # 检测抵抗情绪
        negative_emotions = ['sad', 'angry', 'fear', 'frustrated', 'bored']
        if current_emotion in negative_emotions:
            print(f"[CleanAgent] 😟 检测到负面情绪 {current_emotion}，进入闲聊模式")
            self.is_in_comfort_mode = True
            self.comfort_turn_count = 0
            return self._handle_comfort_mode(user_input, {}, [], current_emotion)
        
        # 检测拒绝关键词
        resistance_keywords = ["不想", "不愿意", "算了", "累了", "休息", "不行"]
        if any(keyword in user_input for keyword in resistance_keywords):
            print(f"[CleanAgent] 🚫 检测到抵抗关键词，进入闲聊模式")
            self.is_in_comfort_mode = True
            self.comfort_turn_count = 0
            return self._handle_comfort_mode(user_input, {}, [], current_emotion)
        
        # 如果是第一轮对话，检测是否同意开始评估
        if self._turn_counter == 1 and not self._user_agrees_to_start(user_input):
            print(f"[CleanAgent] 🤔 用户未明确同意开始，先闲聊了解情况")
            self.is_in_comfort_mode = True  
            self.comfort_turn_count = 0
            return self._handle_comfort_mode(user_input, {}, [], current_emotion)
        
        return None
    
    def _user_agrees_to_start(self, user_input: str) -> bool:
        """检测用户是否同意开始评估"""
        positive_keywords = [
            "好", "行", "可以", "愿意", "试试", "继续", "开始",
            "没问题", "OK", "ok", "嗯", "是的"
        ]
        negative_keywords = [
            "不", "别", "算了", "不想", "不愿意", "不行", 
            "累了", "休息", "等会", "以后"
        ]
        
        for keyword in positive_keywords:
            if keyword in user_input:
                has_negative = any(neg in user_input for neg in negative_keywords)
                if not has_negative:
                    return True
        return False

    def _reset_task_pool(self, session_id: str):
        """重置任务池状态（学习原Agent）"""
        print(f"[CleanAgent] 🔄 重置任务池，会话: {session_id}")
        self._active_session_id = session_id
        self._task_done = set()
        self._last_task_id = None
        self._registration_ts = None
        self._turn_counter = 0
        self._task_cooldown_until = {}
        self._pending_consent_task_id = None
        self.is_in_comfort_mode = False
        self.comfort_turn_count = 0
        self.has_started_assessment = False
        
    def _smart_evaluate_answer(self, user_answer: str, doctor_question: str, 
                             last_task_id: str, dimension_id: str, 
                             patient_profile: Dict, chat_history: List) -> Dict:
        """智能评估策略（学习原Agent复杂逻辑）"""
        
        # 🆕 1. 缓冲任务特殊处理
        last_task_is_buffer = (last_task_id in self.BUFFER_TASKS)
        if last_task_is_buffer:
            print(f"[CleanAgent] 💬 上轮是缓冲任务({last_task_id})，跳过评分但检测抵抗")
            # 仍然执行抵抗检测，以便处理用户情绪
            resistance_result = self._call_resistance_detection(doctor_question, user_answer)
            return {
                'is_correct': True, 'quality_level': 'good', 'cognitive_performance': '正常',
                'is_complete': True, 'evaluation_detail': '缓冲闲聊轮不计分',
                'need_followup': False, 'confidence': 1.0,
                'resistance_result': resistance_result
            }
        
        # 🆕 2. 简单回答快速路径（避免不必要的复杂处理）
        simple_answers = ["好", "好的", "嗯", "对", "是", "是的", "知道", "明白", "行", "可以"]
        is_simple_answer = user_answer.strip() in simple_answers or len(user_answer.strip()) <= 3
        
        if is_simple_answer:
            print(f"[CleanAgent] ⚡ 快速路径: 简单回答'{user_answer[:10]}'，跳过抵抗检测")
            resistance_result = {'is_resistant': False, 'confidence': 1.0, 'category': 'none'}
            eval_result = self._call_answer_evaluation(
                doctor_question, user_answer, dimension_id, patient_profile, ""
            )
            eval_result['resistance_result'] = resistance_result
            return eval_result
        
        # 🆕 3. 复杂回答：本地模式顺序执行（避免GPTQ冲突）
        if self.use_local:
            print("[CleanAgent] ⚡ 本地模式顺序执行抵抗检测与回答评估...")
            
            # 先做抵抗检测
            resistance_result = self._call_resistance_detection(doctor_question, user_answer)
            
            # 🆕 4. 抵抗检测结果处理
            print(f"[CleanAgent] 🔍 抵抗检测详情:")
            print(f"  - is_resistant: {resistance_result.get('is_resistant')}")
            print(f"  - category: {resistance_result.get('category')}")
            print(f"  - confidence: {resistance_result.get('confidence')}")
            
            # 如果检测到严重抵抗，跳过评估
            if resistance_result.get('is_resistant') and resistance_result.get('category') in ['refusal', 'hostility']:
                print(f"[CleanAgent] ⚡ 检测到抵抗({resistance_result.get('category')})，跳过回答评估")
                eval_result = {'is_correct': False, 'quality_level': 'poor', 'skipped': True}
            else:
                eval_result = self._call_answer_evaluation(
                    doctor_question, user_answer, dimension_id, patient_profile, ""
                )
            
            eval_result['resistance_result'] = resistance_result
            return eval_result
        else:
            # API模式：并行执行（原Agent逻辑）
            print("[CleanAgent] ⚡ API模式并行执行...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_resist = executor.submit(self._call_resistance_detection, doctor_question, user_answer)
                future_eval = executor.submit(
                    self._call_answer_evaluation, doctor_question, user_answer, dimension_id, patient_profile, ""
                )
                
                resistance_result = future_resist.result()
                if resistance_result.get('is_resistant') and resistance_result.get('category') in ['refusal', 'hostility']:
                    eval_result = {'is_correct': False, 'quality_level': 'poor', 'skipped': True}
                else:
                    eval_result = future_eval.result()
                
                eval_result['resistance_result'] = resistance_result
                return eval_result

    def _call_resistance_detection(self, doctor_question: str, user_answer: str) -> Dict:
        """调用抵抗检测工具"""
        try:
            resistance_result = self.resistance_tool._run(
                question=doctor_question, 
                answer=user_answer
            )
            return json.loads(resistance_result)
        except Exception as e:
            print(f"⚠️ [CleanAgent] 抵抗检测失败: {e}")
            return {'is_resistant': False, 'confidence': 0.0, 'category': 'none'}
    
    def _call_answer_evaluation(self, doctor_question: str, user_answer: str, 
                              dimension_id: str, patient_profile: Dict, expected_answer: str) -> Dict:
        """调用答案评估工具"""
        try:
            eval_result = self.eval_tool._run(
                dimension_id=dimension_id,
                question=doctor_question,
                answer=user_answer,
                expected_answer=expected_answer,
                patient_profile=patient_profile
            )
            return json.loads(eval_result)
        except Exception as e:
            print(f"⚠️ [CleanAgent] 答案评估失败: {e}")
            return {'is_correct': False, 'quality_level': 'poor'}

    def _generate_completion_response(self, patient_profile: Dict) -> Dict:
        """生成评估完成回应"""
        patient_name = patient_profile.get('name', '')
        greeting = f"{patient_name}，" if patient_name else ""
        
        return {
            'response': f"{greeting}我们的聊天就到这里，感谢您的配合！",
            'task_id': None,
            'evaluation': {},
            'session_data': self.session_data,
            'completed': True
        }
