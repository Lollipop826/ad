"""
统一工具日志系统
提供醒目、结构化的日志输出，清晰展示工具执行流程和数据传递
"""
import time
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager


class ToolLogger:
    """工具日志记录器"""
    
    # 工具图标映射
    TOOL_ICONS = {
        'AgentFC': '🤖',
        'ResistanceTool': '🛡️',
        'ComfortTool': '💬',
        'QuestionGenTool': '❓',
        'AnswerEvalTool': '📝',
        'ScoreTool': '📊',
        'TaskPool': '📋',
        'StandardQuestion': '📌',
        'ImageTool': '🖼️',
        'TTS': '🎵',
        'ASR': '🎤',
        'VAD': '👂',
    }
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.icon = self.TOOL_ICONS.get(tool_name, '🔧')
        self._start_time = None
        self._inputs = {}
        self._outputs = {}
    
    def start(self, **inputs):
        """开始工具执行，记录输入"""
        self._start_time = time.time()
        self._inputs = inputs
        now = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n╔{'═'*60}╗")
        print(f"║ ⏰ [{now}] {self.icon} {self.tool_name:50} ║")
        print(f"╠{'═'*60}╣")
        
        if inputs:
            print(f"  📥 输入:")
            for key, value in inputs.items():
                # 截断过长的值
                str_val = str(value)
                if len(str_val) > 50:
                    str_val = str_val[:47] + "..."
                print(f"     ├── {key}: {str_val}")
        
        return self
    
    def log(self, message: str, level: str = "info"):
        """记录中间日志"""
        icons = {"info": "ℹ️", "warn": "⚠️", "error": "❌", "success": "✅", "step": "  ├──"}
        icon = icons.get(level, "  │  ")
        print(f"  {icon} {message}")
    
    def step(self, message: str):
        """记录执行步骤"""
        print(f"  ├── {message}")
    
    def end(self, **outputs):
        """结束工具执行，记录输出和耗时"""
        self._outputs = outputs
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        if outputs:
            print(f"  📤 输出:")
            for key, value in outputs.items():
                str_val = str(value)
                if len(str_val) > 60:
                    str_val = str_val[:57] + "..."
                print(f"     └── {key}: {str_val}")
        
        print(f"  ⏱️  耗时: {elapsed:.3f}s")
        print(f"╚{'═'*60}╝")
        
        return outputs
    
    def end_with_arrow(self, next_tool: str, data_passed: str, **outputs):
        """结束并显示数据流向下一个工具"""
        self.end(**outputs)
        next_icon = self.TOOL_ICONS.get(next_tool, '🔧')
        print(f"         │")
        print(f"         ▼ ({data_passed})")
        print(f"         │")


@contextmanager
def tool_context(tool_name: str, **inputs):
    """
    工具执行上下文管理器
    
    Usage:
        with tool_context("ResistanceTool", question=q, answer=a) as logger:
            # do work
            logger.step("BERT预测完成")
            logger.end(label="normal", confidence=0.95)
    """
    logger = ToolLogger(tool_name)
    logger.start(**inputs)
    try:
        yield logger
    except Exception as e:
        logger.log(f"执行失败: {e}", level="error")
        raise


def log_tool_start(tool_name: str, **inputs) -> ToolLogger:
    """快速启动工具日志"""
    logger = ToolLogger(tool_name)
    logger.start(**inputs)
    return logger


def log_data_flow(from_tool: str, to_tool: str, data: str):
    """显示工具间数据流动"""
    from_icon = ToolLogger.TOOL_ICONS.get(from_tool, '🔧')
    to_icon = ToolLogger.TOOL_ICONS.get(to_tool, '🔧')
    print(f"\n  {from_icon} {from_tool}")
    print(f"         │")
    print(f"         ▼ ({data})")
    print(f"         │")
    print(f"  {to_icon} {to_tool}\n")


def log_phase(phase_name: str, phase_num: int = None):
    """记录执行阶段"""
    now = datetime.now().strftime("%H:%M:%S")
    if phase_num:
        print(f"\n{'─'*20} ⏰ [{now}] 阶段 {phase_num}: {phase_name} {'─'*20}\n")
    else:
        print(f"\n{'─'*20} ⏰ [{now}] {phase_name} {'─'*20}\n")


def log_summary(title: str, items: Dict[str, Any]):
    """记录摘要信息"""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n┌{'─'*60}┐")
    print(f"│ ⏰ [{now}] 📊 {title:50} │")
    print(f"├{'─'*60}┤")
    for key, value in items.items():
        str_val = str(value)
        if len(str_val) > 45:
            str_val = str_val[:42] + "..."
        print(f"│   {key}: {str_val:52} │")
    print(f"└{'─'*60}┘")
