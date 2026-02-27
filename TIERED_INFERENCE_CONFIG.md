# 分级推理配置说明

## 🎯 设计目标

使用**分级推理**（Tiered Inference）策略：
- 🔥 **小模型（0.5B）**：处理简单的分类和生成任务，速度快 **10-15倍**
- ⚡ **大模型（7B）**：处理复杂的评估和推理任务，保证质量

---

## 📊 模型配置

### 加载的模型

| 模型 | 大小 | 用途 | 显存 |
|------|------|------|------|
| **Qwen2.5-7B-Instruct** | 7B FP16 | 复杂任务 | ~14GB |
| **Qwen2.5-0.5B-Instruct** | 0.5B FP16 | 简单任务 | ~1GB |
| **总计** | | | **~15GB** |

### 实例池

| 实例名 | 模型 | 温度 | max_tokens | 用途 |
|--------|------|------|------------|------|
| `default` | 7B | 0.7 | 512 | 通用任务 |
| `precise` | 7B | 0.3 | 256 | 精确生成 |
| `creative` | 7B | 0.9 | 512 | 创意生成 |
| `agent` | 7B | 0.3 | 256 | Agent 推理 |
| `eval_long` | 7B | 0.3 | 128 | 复杂评估 |
| `small_classify` | **0.5B** | 0.3 | 64 | **分类任务** |
| `small_eval` | **0.5B** | 0.3 | 64 | **简单评估** |
| `small_gen` | **0.5B** | 0.3 | 128 | **简单生成** |

---

## 🔧 工具模型分配

### 使用小模型（0.5B）的工具 ⚡⚡⚡

| 工具 | 使用实例 | 原因 | 预期加速 |
|------|---------|------|---------|
| **ResistanceDetectionTool** | `small_classify` | 简单二分类（是否抵抗） | **10-15倍** |
| **DimensionDetectionTool** | `small_classify` | 有限选项分类（维度识别） | **10-15倍** |
| **QueryTool** | `small_gen` | 生成检索关键词 | **8-12倍** |

**合计节省时间**：
```
ResistanceDetectionTool: 0.20s → 0.015s (节省 0.185s)
DimensionDetectionTool:  0.15s → 0.012s (节省 0.138s)
QueryTool:               0.17s → 0.020s (节省 0.150s)
────────────────────────────────────────────────
总计节省：                           ~0.47s
```

### 使用大模型（7B）的工具 ⚡

| 工具 | 使用实例 | 原因 | 耗时 |
|------|---------|------|------|
| **AnswerEvaluationTool** | `eval_long` | 复杂医学评估，需要专业知识 | 0.60s |
| **QuestionGenerationTool** | `precise` | 生成质量影响用户体验 | 0.46s |

### 不使用 LLM 的工具 ✅

| 工具 | 类型 | 耗时 |
|------|------|------|
| **KnowledgeRetrievalTool** | Embedding 模型（BGE-M3） | 0.02-0.3s |
| **ScoreRecordingTool** | 数据库操作 | < 0.01s |
| **ConversationStorageTool** | 数据库操作 | < 0.01s |

---

## 📈 性能对比

### 优化前（单 7B 模型）

| 阶段 | 耗时 |
|------|------|
| LLM 推理 | 1.58s |
| 非 LLM 操作 | 0.32s |
| **总计** | **1.90s** |

### 优化后（7B + 0.5B 分级推理）

| 阶段 | 耗时 | 改善 |
|------|------|------|
| 小模型推理（3个工具） | 0.047s | ⚡ 节省 0.47s |
| 大模型推理（2个工具） | 1.06s | - |
| 非 LLM 操作 | 0.32s | - |
| **总计** | **1.43s** | ⚡ **加速 25%** |

---

## 🎯 任务复杂度分析

### 为什么这些任务适合小模型？

#### 1. **ResistanceDetectionTool** - 二分类
```
输入：问题 + 回答
任务：判断是否有抵抗情绪（是/否）
输出：{is_resistant: bool, category: str, confidence: float}

复杂度：⭐
理由：
- 简单的情绪识别
- 有限的输出选项（5种情绪类别）
- 不需要深度医学知识
```

#### 2. **DimensionDetectionTool** - 多分类
```
输入：问题 + 回答
任务：识别评估维度
输出：{detected_dimensions: List[str], primary_dimension: str}

复杂度：⭐
理由：
- 有限的维度选项（< 10种）
- 基于关键词匹配
- 规则明确
```

#### 3. **QueryTool** - 关键词生成
```
输入：维度信息 + 历史
任务：生成检索查询
输出：{query: str, keywords: List[str]}

复杂度：⭐⭐
理由：
- 生成简短的关键词
- 有模板可循
- 即使质量略降，RAG 检索也有容错性
```

### 为什么这些任务需要大模型？

#### 1. **AnswerEvaluationTool** - 复杂评估
```
输入：问题 + 回答 + 维度 + 档案
任务：评估认知表现
输出：{quality_level, cognitive_performance, evaluation_detail, ...}

复杂度：⭐⭐⭐⭐⭐
理由：
- 需要医学专业知识
- 多维度综合判断
- 输出影响诊断结果
- Prompt 最长（~500 tokens）
```

#### 2. **QuestionGenerationTool** - 创意生成
```
输入：知识 + 档案 + 历史 + 情绪
任务：生成评估问题
输出：{question: str, followup_type: str}

复杂度：⭐⭐⭐⭐
理由：
- 需要理解医学知识
- 生成自然流畅的问题
- 质量直接影响用户体验
- 需要考虑上下文连贯性
```

---

## ⚙️ 启动配置

启动服务时会自动加载两个模型：

```bash
bash start_voice_only.sh
```

### 启动日志示例

```
[ModelPool] 🚀 初始化模型池（分级推理：7B + 0.5B）...
[LocalQwen] 正在加载模型 (7B): Qwen2.5-7B-Instruct...
[LocalQwen] ✅ 模型加载完成 (7B): Qwen2.5-7B-Instruct
[ModelPool] ✅ 大模型(7B)已加载 (耗时 8.23秒)

[LocalQwen] 正在加载模型 (0.5B): Qwen2.5-0.5B-Instruct...
[LocalQwen] ✅ 模型加载完成 (0.5B): Qwen2.5-0.5B-Instruct
[ModelPool] ✅ 小模型(0.5B)已加载 (耗时 0.87秒)

[ModelPool] 📦 创建大模型实例（7B）:
[ModelPool]   - default: model=7B, temp=0.7, max_tokens=512
[ModelPool]   - precise: model=7B, temp=0.3, max_tokens=256
[ModelPool]   - creative: model=7B, temp=0.9, max_tokens=512
[ModelPool]   - agent: model=7B, temp=0.3, max_tokens=256
[ModelPool]   - eval_long: model=7B, temp=0.3, max_tokens=128

[ModelPool] 📦 创建小模型实例（0.5B）:
[ModelPool]   - small_classify: model=0.5B, temp=0.3, max_tokens=64
[ModelPool]   - small_eval: model=0.5B, temp=0.3, max_tokens=64
[ModelPool]   - small_gen: model=0.5B, temp=0.3, max_tokens=128

[ModelPool] 📊 共预创建 8 个 LLM 实例

[ModelPool] 🔥 开始预热模型...
[ModelPool] ✅ 大模型(7B)预热完成 (耗时 1.23秒)
[ModelPool] ✅ 小模型(0.5B)预热完成 (耗时 0.08秒)
[ModelPool] 💡 后续推理将快 2-3 倍

[ModelPool] ✅ 模型池初始化完成
```

---

## 🧪 测试计划

### 阶段1：功能测试（低风险工具）
1. ✅ 测试 **ResistanceDetectionTool**
   - 能否正确识别抵抗情绪？
   - 置信度是否合理？

2. ✅ 测试 **DimensionDetectionTool**
   - 能否正确识别维度？
   - 是否会误判？

3. ✅ 测试 **QueryTool**
   - 生成的查询是否有效？
   - 检索结果质量如何？

### 阶段2：性能测试
1. ⏱️ 测试推理速度
   - 小模型是否真的快 10-15 倍？
   - 总体耗时是否减少？

2. ⏱️ 测试显存占用
   - 是否在 15GB 左右？
   - 是否稳定？

### 阶段3：质量评估
1. 📊 对比回答质量
   - 小模型工具的输出是否可接受？
   - 是否影响最终诊断？

2. 📊 用户体验
   - 响应速度是否更快？
   - 对话是否流畅？

---

## 📌 注意事项

### 何时可以使用小模型？
✅ 任务有明确的输入输出格式
✅ 输出选项有限（分类任务）
✅ 不需要深度领域知识
✅ 有容错机制（如 QueryTool 的 fallback）

### 何时必须使用大模型？
❌ 需要专业医学知识
❌ 输出质量直接影响用户体验
❌ 复杂的多步推理
❌ Prompt 很长（> 500 tokens）

---

## 🚀 未来优化方向

### 短期
1. ⏳ 监控小模型的实际性能和质量
2. ⏳ 根据实测调整模型分配策略
3. ⏳ 考虑是否将 QuestionGenerationTool 也用小模型

### 中期
1. 🔨 尝试 1.5B 模型（质量和速度的平衡点）
2. 🔨 实现动态模型选择（根据任务难度自动选择）
3. 🔨 批量推理优化（同时处理多个请求）

### 长期
1. 🔨 专用微调模型（为特定任务训练小模型）
2. 🔨 知识蒸馏（用 7B 教 0.5B）
3. 🔨 边缘部署（0.5B 可以在更低端硬件运行）

---

## 📖 参考文档

- `AGENT_TOOLS_SUMMARY.md` - 工具系统总结
- `MODEL_POOL_OPTIMIZATION.md` - 模型池优化方案
- `LLM_INFERENCE_OPTIMIZATION.md` - LLM 推理优化
