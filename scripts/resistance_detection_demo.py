import os
import json
from datetime import datetime

from src.tools.answer_analysis import ResistanceDetectionTool


def print_header(title: str) -> None:
    line = "=" * 72
    print(line)
    print(f"{title}")
    print(line)


def print_case(idx: int, question: str, answer: str, result_json: str) -> None:
    data = {}
    try:
        data = json.loads(result_json)
    except Exception:
        data = {"raw": result_json}

    print(f"\n[用例 {idx}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 72)
    print("对话上下文：")
    print(f"  医生: {question}")
    print(f"  患者: {answer}")
    print("\n检测结果：")
    if data and isinstance(data, dict):
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        for line in pretty.splitlines():
            print(f"  {line}")
    else:
        print(f"  {result_json}")
    print("-" * 72)


def main() -> None:
    print_header("抵抗情绪检测工具 · 演示输出")

    tool = ResistanceDetectionTool()

    # 用例1：正常配合
    q1 = "我们先做一个简单的时间定向测试，可以吗？"
    a1 = "好的，您说吧"
    r1 = tool._run(question=q1, answer=a1)
    print_case(1, q1, a1, r1)

    # 用例2：明确拒绝
    q2 = "请告诉我今天是几月几号？"
    a2 = "别问了，我不想回答，也不想做这个检查"
    r2 = tool._run(question=q2, answer=a2)
    print_case(2, q2, a2, r2)

    # 用例3：回避/疲惫
    q3 = "请您简单重复我说的三个词：苹果、钢笔、汽车。"
    a3 = "等会儿吧，我有点累，先不说了。"
    r3 = tool._run(question=q3, answer=a3)
    print_case(3, q3, a3, r3)

    print("\n提示：以上输出可直接截图用于文档展示。\n")


if __name__ == "__main__":
    main()


