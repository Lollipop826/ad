import json
from datetime import datetime

from src.tools.answer_analysis import DimensionDetectionTool


def print_header(title: str) -> None:
    line = "=" * 72
    print(line)
    print(title)
    print(line)


def print_case(idx: int, question: str, answer: str, result_json: str) -> None:
    try:
        data = json.loads(result_json)
    except Exception:
        data = {"raw": result_json}

    print(f"\n[用例 {idx}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 72)
    print("对话上下文：")
    print(f"  医生: {question}")
    print(f"  患者: {answer}")
    print("\n维度识别结果：")
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    for line in pretty.splitlines():
        print(f"  {line}")
    print("-" * 72)


def main() -> None:
    print_header("维度识别工具 · 演示输出")

    tool = DimensionDetectionTool()

    # 用例1：定向力（时间定向）
    q1 = "请告诉我现在是几月几号？"
    a1 = "今天是9月28号，星期天"
    r1 = tool._run(question=q1, answer=a1)
    print_case(1, q1, a1, r1)

    # 用例2：语言（复述）
    q2 = "请重复我说的话：今天天气真不错"
    a2 = "今天天气真不错"
    r2 = tool._run(question=q2, answer=a2)
    print_case(2, q2, a2, r2)

    # 用例3：注意力与计算（100-7）
    q3 = "请从100开始，每次减去7，连续说五个数"
    a3 = "100，93，86，79，72"
    r3 = tool._run(question=q3, answer=a3)
    print_case(3, q3, a3, r3)

    print("\n提示：以上输出可直接截图用于文档展示。\n")


if __name__ == "__main__":
    main()


