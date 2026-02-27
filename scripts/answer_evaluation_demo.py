"""
回答评估工具演示

测试回答评估和评分记录功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from src.tools.agent_tools.answer_evaluation_tool import AnswerEvaluationTool
from src.tools.agent_tools.score_recording_tool import ScoreRecordingTool
import json


def print_section(title: str):
    """打印分隔标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_evaluation():
    """演示回答评估功能"""
    print_section("回答评估工具演示")
    
    tool = AnswerEvaluationTool()
    
    # 测试场景1：定向力 - 正确回答
    print("【场景1】定向力评估 - 完全正确")
    print("问题：请问今天是几月几号？")
    print("回答：10月4号")
    
    result = tool._run(
        question="请问今天是几月几号？",
        answer="10月4号",
        dimension_id="orientation",
        patient_profile={"age": 70, "education_years": 6}
    )
    
    data = json.loads(result)
    print(f"\n✅ 评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 得分: {data['score']} (满分1.0)")
    print(f"  - 是否完整: {data['is_complete']}")
    print(f"  - 需要追问: {data['need_followup']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")
    
    # 测试场景2：定向力 - 部分正确
    print("\n" + "-"*60)
    print("【场景2】定向力评估 - 部分正确")
    print("问题：请问今天是几月几号，星期几？")
    print("回答：10月4号...星期几我不太清楚")
    
    result = tool._run(
        question="请问今天是几月几号，星期几？",
        answer="10月4号...星期几我不太清楚",
        dimension_id="orientation"
    )
    
    data = json.loads(result)
    print(f"\n⚠️  评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 得分: {data['score']}")
    print(f"  - 是否完整: {data['is_complete']}")
    print(f"  - 需要追问: {data['need_followup']}")
    if data.get('followup_suggestion'):
        print(f"  - 追问建议: {data['followup_suggestion']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")
    
    # 测试场景3：即时记忆 - 正确
    print("\n" + "-"*60)
    print("【场景3】即时记忆评估 - 完全正确")
    print("问题：请重复我刚才说的三个词：苹果、桌子、外套")
    print("回答：苹果、桌子、外套")
    
    result = tool._run(
        question="请重复我刚才说的三个词：苹果、桌子、外套",
        answer="苹果、桌子、外套",
        dimension_id="registration",
        expected_answer="苹果、桌子、外套"
    )
    
    data = json.loads(result)
    print(f"\n✅ 评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 得分: {data['score']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")
    
    # 测试场景4：即时记忆 - 部分记住
    print("\n" + "-"*60)
    print("【场景4】即时记忆评估 - 只记住2个")
    print("问题：请重复我刚才说的三个词：苹果、桌子、外套")
    print("回答：苹果...桌子...还有一个忘了")
    
    result = tool._run(
        question="请重复我刚才说的三个词：苹果、桌子、外套",
        answer="苹果...桌子...还有一个忘了",
        dimension_id="registration",
        expected_answer="苹果、桌子、外套"
    )
    
    data = json.loads(result)
    print(f"\n⚠️  评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 得分: {data['score']} (答对2/3)")
    print(f"  - 需要追问: {data['need_followup']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")
    
    # 测试场景5：注意力与计算 - 正确
    print("\n" + "-"*60)
    print("【场景5】注意力与计算 - 正确")
    print("问题：100减7等于多少？")
    print("回答：93")
    
    result = tool._run(
        question="100减7等于多少？",
        answer="93",
        dimension_id="attention_calculation",
        expected_answer="93"
    )
    
    data = json.loads(result)
    print(f"\n✅ 评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 得分: {data['score']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")
    
    # 测试场景6：完全错误
    print("\n" + "-"*60)
    print("【场景6】回答错误")
    print("问题：100减7等于多少？")
    print("回答：95")
    
    result = tool._run(
        question="100减7等于多少？",
        answer="95",
        dimension_id="attention_calculation"
    )
    
    data = json.loads(result)
    print(f"\n❌ 评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 得分: {data['score']}")
    print(f"  - 需要追问: {data['need_followup']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")


def demo_score_recording():
    """演示评分记录功能"""
    print_section("评分记录工具演示")
    
    tool = ScoreRecordingTool()
    session_id = "test_session_20241004"
    
    # 场景1：记录定向力得分
    print("【场景1】记录定向力得分")
    result = tool._run(
        session_id=session_id,
        dimension_id="orientation",
        score=0.8,  # 答对了10分中的8分
        max_score=10.0,
        question="今天是几月几号？星期几？",
        answer="10月4号，星期六",
        evaluation_detail="日期和星期都答对了，得8/10分",
        action="save"
    )
    
    data = json.loads(result)
    print(f"✅ {data['message']}")
    print(f"   当前得分: {data['current_score']:.1f}")
    print(f"   累计总分: {data['total_score']:.1f}/{data['max_total_score']:.1f}")
    
    # 场景2：记录即时记忆得分
    print("\n【场景2】记录即时记忆得分")
    result = tool._run(
        session_id=session_id,
        dimension_id="registration",
        score=1.0,  # 3个词全部记住
        max_score=3.0,
        question="请重复：苹果、桌子、外套",
        answer="苹果、桌子、外套",
        evaluation_detail="三个词全部正确复述",
        action="save"
    )
    
    data = json.loads(result)
    print(f"✅ {data['message']}")
    print(f"   当前得分: {data['current_score']:.1f}")
    print(f"   累计总分: {data['total_score']:.1f}/{data['max_total_score']:.1f}")
    
    # 场景3：记录注意力与计算得分
    print("\n【场景3】记录注意力与计算得分")
    result = tool._run(
        session_id=session_id,
        dimension_id="attention_calculation",
        score=0.6,  # 5次计算对了3次
        max_score=5.0,
        question="100连续减7",
        answer="93、86、79...",
        evaluation_detail="前3次正确，后面出错",
        action="save"
    )
    
    data = json.loads(result)
    print(f"✅ {data['message']}")
    print(f"   当前得分: {data['current_score']:.1f}")
    print(f"   累计总分: {data['total_score']:.1f}/{data['max_total_score']:.1f}")
    
    # 场景4：获取总分汇总
    print("\n" + "-"*60)
    print("【场景4】获取总分汇总")
    result = tool._run(
        session_id=session_id,
        action="summary"
    )
    
    data = json.loads(result)
    print(f"\n📊 {data['message']}")
    print(f"   完成度: {data['score_details']['completion_rate']:.1f}%")
    print(f"   已完成维度: {', '.join(data['completed_dimensions'])}")
    
    print("\n   各维度得分详情：")
    for dim_id, dim_data in data['score_details']['dimensions'].items():
        print(f"   - {dim_id}: {dim_data['total_score']:.1f}/{dim_data['max_score']:.1f}")


def main():
    """主函数"""
    try:
        demo_evaluation()
        print("\n" + "="*60 + "\n")
        demo_score_recording()
        
        print("\n" + "="*60)
        print("  ✨ 演示完成！")
        print("="*60)
        print("\n💡 说明：")
        print("  - 回答评估工具可以自动判断回答的正确性和完整性")
        print("  - 评分严格遵循MMSE标准")
        print("  - 如果回答不完整，会建议追问")
        print("  - 评分会自动记录并累计总分")
        print("  - 最终可以生成完整的MMSE评估报告")
        
    except Exception as e:
        print(f"\n❌ 演示过程出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

