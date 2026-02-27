# Agent 工具系统总结

## 📋 工具列表和作用

阿尔茨海默病初筛 Agent 使用 **ReAct 推理模式**，严格按照以下流程调用工具：

---

## 🔧 8个工具详解

### 1. **ResistanceDetectionTool** - 抵抗情绪检测 ⚡

**作用**：检测患者是否有抵抗、拒绝、回避、敌意等负面情绪

**输入**：
- `question`: 医生的问题
- `answer`: 患者的回答

**输出**：
```json
{
  "is_resistant": false,
  "category": "none|refusal|avoidance|hostility|fatigue",
  "confidence": 0.8,
  "rationale": "判断理由"
}
```

**使用 LLM**：✅ 使用 `eval_short` (128 tokens)

**典型耗时**：0.20s (7B优化) / 8.50s (14B-Int4)

**关键点**：
- 如果 `is_resistant=true` 且 `confidence>=0.8`，Agent 应调用安慰工具并直接返回
- 否则继续评估流程

---

### 2. **AnswerEvaluationTool** - 回答评估 ⚡⚡

**作用**：评估患者回答的质量和认知表现

**输入**：
- `question`: 医生的问题
- `answer`: 患者的回答
- `dimension_id`: 当前评估维度（如 orientation）
- `patient_profile`: 患者档案

**输出**：
```json
{
  "is_correct": true,
  "quality_level": "excellent|good|fair|poor",
  "cognitive_performance": "正常|轻度异常|中度异常|重度异常",
  "is_complete": true,
  "evaluation_detail": "评估详情",
  "need_followup": false
}
```

**使用 LLM**：✅ 使用 `eval_short` (128 tokens)

**典型耗时**：0.60s (7B优化) / 20.47s (14B-Int4)

**关键点**：
- **最耗时的工具**（Prompt 最长，评估最复杂）
- 输出会影响评分和下一个问题的生成

---

### 3. **ScoreRecordingTool** - 评分记录 ⚡

**作用**：将评估结果保存到数据库

**输入**：
- `session_id`: 会话ID
- `dimension_id`: 维度ID
- `quality_level`: 质量等级（来自 AnswerEvaluationTool）
- `cognitive_performance`: 认知表现
- `question`, `answer`: 问答内容
- `evaluation_detail`: 评估详情
- `action`: "save"

**输出**：
```json
{
  "success": true,
  "message": "评分记录成功"
}
```

**使用 LLM**：❌ 不使用（纯数据库操作）

**典型耗时**：< 0.01s

**关键点**：
- Agent **必须调用**，不能跳过
- 用于后续统计分析

---

### 4. **QueryTool** - 检索查询生成 ⚡

**作用**：根据当前维度、历史对话生成知识检索的查询语句

**输入**：
- `dimension`: 当前维度信息 `{"id": "orientation", "name": "定向力"}`
- `history`: 历史对话记录
- `profile`: 患者档案
- `last_emotion`: 上一轮情绪

**输出**：
```json
{
  "query": "阿尔茨海默病 定向力 评估",
  "keywords": ["阿尔茨海默病", "定向力", "评估"],
  "target_dimensions": ["orientation"]
}
```

**使用 LLM**：✅ 使用 `precise` (256 tokens)

**典型耗时**：0.17s (7B优化)

**关键点**：
- 生成的 `query` 会传给 KnowledgeRetrievalTool
- 如果 LLM 失败，有 fallback 机制

---

### 5. **KnowledgeRetrievalTool** - 知识检索 ⚡

**作用**：从向量数据库检索相关的医学知识

**输入**：
- `query`: 检索查询（来自 QueryTool）
- `top_k`: 返回结果数量（默认 5）
- `skip_reranking`: 是否跳过重排序（默认 True，已优化）
- `use_fusion`: 是否使用 RAG Fusion（默认 False，已禁用）

**输出**：
```json
{
  "success": true,
  "query": "检索查询",
  "results_count": 5,
  "results": [
    {
      "rank": 1,
      "text": "相关知识片段",
      "full_text": "完整文本",
      "source": "来源文件",
      "relevance": "high|medium"
    }
  ]
}
```

**使用 LLM**：❌ 不使用（使用 Embedding 模型）

**典型耗时**：0.02s (缓存命中) / 0.3s (首次检索)

**关键点**：
- 使用 **BGE-M3 Embedding** 模型（已池化预热）
- 有缓存机制，相同查询直接返回
- 已禁用 RAG Fusion（节省 2-3秒）

---

### 6. **QuestionGenerationTool** - 问题生成 ⚡⚡

**作用**：根据检索到的知识和患者情况，生成下一个评估问题

**输入**：
- `dimension_name`: 维度名称
- `knowledge_context`: 检索到的医学知识（来自 KnowledgeRetrievalTool）
- `patient_profile`: 患者档案
- `conversation_history`: 历史对话
- `patient_emotion`: 患者情绪

**输出**：
```json
{
  "question": "生成的评估问题",
  "followup_type": "direct|probing|confirming",
  "dimension": "orientation"
}
```

**使用 LLM**：✅ 使用 `precise` (256 tokens)

**典型耗时**：0.46s (7B优化)

**关键点**：
- Agent **必须调用**，禁止手动编造问题
- 生成的问题会作为 `Final Answer` 返回

---

### 7. **DimensionDetectionTool** - 维度检测 ⚡

**作用**：检测当前对话应该评估哪个认知维度

**输入**：
- `question`: 医生的问题
- `answer`: 患者的回答

**输出**：
```json
{
  "detected_dimensions": ["orientation", "memory"],
  "primary_dimension": "orientation",
  "confidence": 0.9
}
```

**使用 LLM**：✅ 使用 `eval_short` (128 tokens)

**典型耗时**：0.15s (7B优化)

**关键点**：
- 用于多维度同时评估的场景
- 当前流程中不常用

---

### 8. **ConversationStorageTool** - 对话存储 ⚡

**作用**：保存对话历史到数据库

**输入**：
- `action`: "save_turn"
- `session_id`: 会话ID
- `user_input`: 用户输入
- `generated_question`: 生成的问题
- ... 其他对话信息

**输出**：
```json
{
  "success": true,
  "turn_id": "t_123"
}
```

**使用 LLM**：❌ 不使用（纯数据库操作）

**典型耗时**：< 0.01s

---

## 📊 完整工作流程（单轮对话）

```
患者回答
  ↓
1. ResistanceDetectionTool (检测抵抗情绪)
  ├─ 如果 is_resistant=true → 调用安慰工具 → 返回
  └─ 如果 is_resistant=false → 继续
  ↓
2. AnswerEvaluationTool (评估回答质量)
  ↓
3. ScoreRecordingTool (记录评分)
  ↓
4. QueryTool (生成检索查询)
  ↓
5. KnowledgeRetrievalTool (检索医学知识)
  ↓
6. QuestionGenerationTool (生成下一个问题)
  ↓
返回问题给患者
```

---

## ⚡ 性能优化前后对比

### 优化前（7B-FP16, max_tokens=512）

| 工具 | 是否用 LLM | 耗时 |
|------|-----------|------|
| ResistanceDetectionTool | ✅ | 0.66s |
| AnswerEvaluationTool | ✅ | 1.89s |
| ScoreRecordingTool | ❌ | < 0.01s |
| QueryTool | ✅ | 0.33s |
| KnowledgeRetrievalTool | ❌ (Embedding) | 0.02-0.3s |
| QuestionGenerationTool | ✅ | 0.91s |
| **总计（LLM部分）** | | **3.79s** |

### 优化后（7B-FP16, max_tokens=128/256）

| 工具 | 是否用 LLM | max_tokens | 耗时 | 节省 |
|------|-----------|------------|------|------|
| ResistanceDetectionTool | ✅ | 128 | **0.20s** | ⚡ 70% |
| AnswerEvaluationTool | ✅ | 128 | **0.60s** | ⚡ 68% |
| ScoreRecordingTool | ❌ | - | < 0.01s | - |
| QueryTool | ✅ | 256 | **0.17s** | ⚡ 48% |
| KnowledgeRetrievalTool | ❌ | - | 0.02-0.3s | - |
| QuestionGenerationTool | ✅ | 256 | **0.46s** | ⚡ 49% |
| **总计（LLM部分）** | | | **~1.43s** | ⚡ **62%** |

### 14B-Int4 测试（不推荐）

| 工具 | 耗时 | 对比7B优化后 |
|------|------|-------------|
| ResistanceDetectionTool | 8.50s | ❌ 慢 42倍 |
| AnswerEvaluationTool | 20.47s | ❌ 慢 34倍 |
| **结论** | | **不适合实时对话** |

---

## 🎯 关键发现

### 1. 哪些工具使用 LLM？

**使用 LLM**（占总耗时）：
- ✅ ResistanceDetectionTool (eval_short, 128 tokens)
- ✅ AnswerEvaluationTool (eval_short, 128 tokens) ← **最耗时**
- ✅ QueryTool (precise, 256 tokens)
- ✅ QuestionGenerationTool (precise, 256 tokens)
- ✅ DimensionDetectionTool (eval_short, 128 tokens)

**不使用 LLM**（几乎不耗时）：
- ❌ ScoreRecordingTool（数据库操作）
- ❌ ConversationStorageTool（数据库操作）
- ❌ KnowledgeRetrievalTool（使用 Embedding 模型）

### 2. 优化瓶颈分析

**LLM 推理占比**：~95%（3.79s / 4s 总耗时）

**最慢的工具**：
1. AnswerEvaluationTool (1.89s → 0.60s) ← **最大瓶颈**
2. QuestionGenerationTool (0.91s → 0.46s)
3. ResistanceDetectionTool (0.66s → 0.20s)

**优化效果**：
- 减少 max_tokens：节省 **62%**
- 模型池化 + 预热：首次推理快 **2-3倍**
- Embedding 池化：RAG 检索快 **2-3倍**

### 3. 为什么 14B-Int4 不行？

**理论计算**：
```
14B 推理时间 = 7B × (14/7) × Int4加速系数
              = 7B × 2 × 0.7
              = 7B × 1.4倍
```

**实测结果**：
```
ResistanceTool: 0.20s (7B) vs 8.50s (14B-Int4) = 42倍差距！
```

**原因分析**：
- Int4 量化加速没有预期那么大（理论 30-40%，实际可能更小）
- 14B 参数量翻倍 → 计算量翻倍
- GPTQ 量化的 overhead
- 净效果：比 7B 慢 **10-40倍**

---

## 💡 未来优化方向

### 短期（立即可做）
1. ✅ **减少 max_tokens**（已完成，节省 62%）
2. ✅ **模型池化 + 预热**（已完成）
3. ✅ **Embedding 池化**（已完成）
4. ⏳ **Prompt 压缩**（AnswerEvaluationTool 的 prompt 可以精简）

### 中期（需要测试）
1. 🔨 **7B-Int4 量化**（理论快 50-100%，质量损失 < 2%）
2. 🔨 **批量推理**（多个独立工具并行调用）
3. 🔨 **KV Cache 复用**（复用 system prompt）

### 长期（架构调整）
1. 🔨 **简化工作流**（合并某些工具调用）
2. 🔨 **使用更小模型**（3B 模型用于简单任务）
3. 🔨 **分级推理**（简单任务用小模型，复杂任务用大模型）

---

## 📌 总结

当前系统已经过较好优化：
- ✅ LLM 推理时间：3.79s → **1.43s**（节省 62%）
- ✅ 单轮对话总时间：4s → **~1.5-2s**
- ✅ 模型加载：只加载一次，所有工具共享
- ✅ Embedding 预热：首次检索快 2-3倍

**建议保持当前配置**：
- 底层模型：**Qwen2.5-7B-Instruct (FP16)**
- max_tokens 优化：**128/256** (不同工具)
- 如需更快，可尝试 **7B-Int4 量化**
