# Agent Tools - 工具集合

所有可供LLM Agent调用的工具都在这个文件夹中，每个工具一个独立的Python文件。

## 📁 文件结构

```
agent_tools/
├── __init__.py                      # 统一导出所有工具
├── query_tool.py                    # 查询生成工具
├── retrieval_tool.py               # 知识检索工具
├── dimension_detection_tool.py     # 维度检测工具
├── resistance_detection_tool.py    # 抵抗情绪检测工具
├── question_generation_tool.py     # 问题生成工具
└── storage_tool.py                 # 对话存储工具
```

## 🔧 工具说明

### 1. query_tool.py - 查询生成工具
- **类名**: `QueryTool`
- **功能**: 根据当前维度、对话历史、用户情绪和画像，生成用于知识检索的查询语句
- **输入**: 维度信息、对话历史、用户情绪、用户画像
- **输出**: 优化的检索查询（关键词组合）

### 2. retrieval_tool.py - 知识检索工具
- **类名**: `KnowledgeRetrievalTool`
- **功能**: 从医学知识库中检索与查询相关的专业知识
- **输入**: 检索查询字符串
- **输出**: 相关文档列表（JSON格式）

### 3. dimension_detection_tool.py - 维度检测工具
- **类名**: `DimensionDetectionTool`
- **功能**: 判断用户回答是否回答了问题，并识别涉及哪些MMSE维度
- **输入**: 医生问题、用户回答
- **输出**: 是否回答、涉及维度列表、置信度

### 4. resistance_detection_tool.py - 抵抗情绪检测工具
- **类名**: `ResistanceDetectionTool`
- **功能**: 检测用户回答中是否存在抵抗、拒绝或不配合的情绪
- **输入**: 医生问题、用户回答
- **输出**: 是否抵抗、情绪类别、置信度

### 5. question_generation_tool.py - 问题生成工具
- **类名**: `QuestionGenerationTool`
- **功能**: 基于医学知识、患者信息和对话历史，生成评估问题
- **输入**: 维度名称、医学知识、患者信息
- **输出**: 生成的问题

### 6. storage_tool.py - 对话存储工具
- **类名**: `ConversationStorageTool`
- **功能**: 保存对话轮次或获取对话历史
- **输入**: 会话ID、操作类型、数据
- **输出**: 操作结果

## 💻 使用方法

### 导入所有工具
```python
from src.tools.agent_tools import (
    QueryTool,
    KnowledgeRetrievalTool,
    DimensionDetectionTool,
    ResistanceDetectionTool,
    QuestionGenerationTool,
    ConversationStorageTool,
    ALL_TOOLS,
)
```

### 在Agent中使用
```python
from src.agents.screening_agent import ADScreeningAgent

# 初始化Agent（会自动加载所有工具）
agent = ADScreeningAgent()

# Agent会自主决定调用哪些工具
result = agent.process_turn(
    user_input="我最近记忆力不好",
    dimension={"name": "记忆力", "description": "记忆功能评估"},
    session_id="session_001"
)
```

### 直接使用单个工具
```python
# 创建工具实例
retrieval_tool = KnowledgeRetrievalTool()

# 调用工具
result = retrieval_tool.run(query="阿尔茨海默病 定向力 评估")
```

## 🎯 设计原则

1. **一个工具一个文件**: 每个工具独立在一个Python文件中，便于维护
2. **统一的接口**: 所有工具都继承自`BaseTool`，实现`_run`方法
3. **清晰的命名**: 文件名和类名都清晰表达工具功能
4. **完整的文档**: 每个工具都有详细的docstring说明
5. **独立可测试**: 每个工具可以独立测试和使用

## 📊 工具调用流程示例

```
用户输入 → Agent决策
    ↓
1. dimension_detection_tool  (分析用户回答)
    ↓
2. resistance_detection_tool (检测情绪)
    ↓
3. query_tool               (生成查询)
    ↓
4. retrieval_tool           (检索知识)
    ↓
5. question_generation_tool (生成问题)
    ↓
6. storage_tool             (保存对话)
```

## 🔗 相关文档

- Agent实现: `src/agents/screening_agent.py`
- 演示脚本: `scripts/agent_demo.py`
- 类型定义: `src/common/types.py`
- 维度定义: `src/domain/dimensions.py`

