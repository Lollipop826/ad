"""
认知评估工具演示 - 调整后的版本

展示灵活的质量评估系统（不是固定的MMSE评分制）
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


def demo_flexible_evaluation():
    """演示灵活的质量评估"""
    print_section("灵活的质量评估演示")
    
    print("💡 重要提示：这是智能对话式评估，不是固定的量表测试")
    print("   - 问题基于患者画像、历史对话和医学知识动态生成")
    print("   - 评估关注回答质量和认知表现，不是固定分数")
    print("   - 灵活评估，不死板套用标准\n")
    
    tool = AnswerEvaluationTool()
    
    # 场景1：自然对话式的定向力评估
    print("【场景1】自然对话式定向力评估")
    print("问题：张奶奶，您还记得今天来的时候，外面的天气怎么样吗？")
    print("回答：哦，今天天气挺好的，阳光明媚")
    
    result = tool._run(
        question="张奶奶，您还记得今天来的时候，外面的天气怎么样吗？",
        answer="哦，今天天气挺好的，阳光明媚",
        dimension_id="orientation",
        patient_profile={"age": 72, "education_years": 6, "name": "张奶奶"}
    )
    
    data = json.loads(result)
    print(f"\n✅ 评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 质量等级: {data['quality_level']}")
    print(f"  - 认知表现: {data['cognitive_performance']}")
    print(f"  - 是否完整: {data['is_complete']}")
    print(f"  - 需要追问: {data['need_followup']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")
    
    # 场景2：记忆力评估 - 部分回忆
    print("\n" + "-"*60)
    print("【场景2】记忆力评估 - 部分回忆")
    print("问题：刚才我们聊天时您提到过您的孙女，她叫什么名字来着？")
    print("回答：嗯...好像是叫...小芳还是小红来着，我一下想不起来了")
    
    result = tool._run(
        question="刚才我们聊天时您提到过您的孙女，她叫什么名字来着？",
        answer="嗯...好像是叫...小芳还是小红来着，我一下想不起来了",
        dimension_id="recall"
    )
    
    data = json.loads(result)
    print(f"\n⚠️  评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 质量等级: {data['quality_level']}")
    print(f"  - 认知表现: {data['cognitive_performance']}")
    print(f"  - 需要追问: {data['need_followup']}")
    if data.get('followup_suggestion'):
        print(f"  - 追问建议: {data['followup_suggestion']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")
    
    # 场景3：注意力评估 - 良好表现
    print("\n" + "-"*60)
    print("【场景3】注意力评估 - 良好表现")
    print("问题：王叔叔，能帮我从100开始，每次减3，连续减5次吗？")
    print("回答：好的，100、97、94、91、88")
    
    result = tool._run(
        question="王叔叔，能帮我从100开始，每次减3，连续减5次吗？",
        answer="好的，100、97、94、91、88",
        dimension_id="attention_calculation",
        patient_profile={"age": 68, "education_years": 12}
    )
    
    data = json.loads(result)
    print(f"\n✅ 评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 质量等级: {data['quality_level']}")
    print(f"  - 认知表现: {data['cognitive_performance']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")
    
    # 场景4：语言能力评估 - 异常表现
    print("\n" + "-"*60)
    print("【场景4】语言能力评估 - 混乱回答")
    print("问题：李奶奶，能不能给我描述一下您的早餐吃了什么？")
    print("回答：早餐...那个...那个东西...我忘了叫什么...反正就是那个...")
    
    result = tool._run(
        question="李奶奶，能不能给我描述一下您的早餐吃了什么？",
        answer="早餐...那个...那个东西...我忘了叫什么...反正就是那个...",
        dimension_id="language"
    )
    
    data = json.loads(result)
    print(f"\n❌ 评估结果：")
    print(f"  - 是否正确: {data['is_correct']}")
    print(f"  - 质量等级: {data['quality_level']}")
    print(f"  - 认知表现: {data['cognitive_performance']}")
    print(f"  - 需要追问: {data['need_followup']}")
    print(f"  - 评估详情: {data['evaluation_detail']}")


def demo_performance_recording():
    """演示认知表现记录"""
    print_section("认知表现记录演示")
    
    tool = ScoreRecordingTool()
    session_id = "demo_session_20241004"
    
    # 场景1：记录定向力表现
    print("【场景1】记录定向力表现 - 良好")
    result = tool._run(
        session_id=session_id,
        dimension_id="orientation",
        quality_level="good",
        cognitive_performance="正常",
        question="您还记得今天是几月几号吗？",
        answer="10月4号",
        evaluation_detail="回答正确且反应迅速",
        action="save"
    )
    
    data = json.loads(result)
    print(f"✅ {data['message']}")
    print(f"   当前表现: {data['current_performance']}")
    print(f"   整体状态: {data['overall_status']}")
    
    # 场景2：记录即时记忆表现
    print("\n【场景2】记录即时记忆表现 - 优秀")
    result = tool._run(
        session_id=session_id,
        dimension_id="registration",
        quality_level="excellent",
        cognitive_performance="正常",
        question="请重复：苹果、桌子、外套",
        answer="苹果、桌子、外套",
        evaluation_detail="三个词全部正确复述",
        action="save"
    )
    
    data = json.loads(result)
    print(f"✅ {data['message']}")
    print(f"   当前表现: {data['current_performance']}")
    print(f"   整体状态: {data['overall_status']}")
    
    # 场景3：记录延迟回忆表现 - 轻度异常
    print("\n【场景3】记录延迟回忆表现 - 轻度异常")
    result = tool._run(
        session_id=session_id,
        dimension_id="recall",
        quality_level="fair",
        cognitive_performance="轻度异常",
        question="还记得刚才让您记住的三个词吗？",
        answer="苹果...还有一个...想不起来了",
        evaluation_detail="只回忆起1个词，有明显遗漏",
        action="save"
    )
    
    data = json.loads(result)
    print(f"⚠️  {data['message']}")
    print(f"   当前表现: {data['current_performance']}")
    print(f"   整体状态: {data['overall_status']}")
    
    # 场景4：获取整体汇总
    print("\n" + "-"*60)
    print("【场景4】获取整体认知状态汇总")
    result = tool._run(
        session_id=session_id,
        action="summary"
    )
    
    data = json.loads(result)
    print(f"\n📊 {data['message']}")
    print(f"   完成度: {data['performance_details']['completion_rate']:.1f}%")
    print(f"   已完成维度: {', '.join(data['completed_dimensions'])}")
    
    print("\n   各维度表现详情：")
    for dim_id, dim_summary in data['performance_details']['dimension_summary'].items():
        print(f"   - {dim_id}: {dim_summary['latest_performance']} (评估{dim_summary['attempts']}次)")


def main():
    """主函数"""
    try:
        demo_flexible_evaluation()
        print("\n" + "="*60 + "\n")
        demo_performance_recording()
        
        print("\n" + "="*60)
        print("  ✨ 演示完成！")
        print("="*60)
        print("\n💡 核心特点：")
        print("  ✅ 灵活的质量评估（excellent/good/fair/poor）")
        print("  ✅ 关注认知表现（正常/轻度异常/中度异常/重度异常）")
        print("  ✅ 不是固定的30分评分制")
        print("  ✅ 动态生成问题，结合患者画像和历史对话")
        print("  ✅ 记录评估历史，跟踪认知状态趋势")
        print("\n🎯 这是智能对话式评估，不是传统量表测试！")
        
    except Exception as e:
        print(f"\n❌ 演示过程出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

