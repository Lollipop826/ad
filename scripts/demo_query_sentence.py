from typing import List
import os

from src.common.types import InfoDimension, ConversationTurn
from src.tools.query_sentence import QuerySentenceGenerator, QuerySentenceGeneratorConfig
from src.domain.dimensions import new_session_dimensions, update_dimension_status


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置 OPENAI_API_KEY 环境变量。")
        return

    dims: List[InfoDimension] = new_session_dimensions()
    update_dimension_status(dims, "orientation", status="unknown")

    history: List[ConversationTurn] = [
        {"role": "assistant", "content": "我们会做一些简单的记忆与定向测试。", "emotion": None},
        {"role": "user", "content": "好的", "emotion": "neutral"},
    ]

    gen = QuerySentenceGenerator(QuerySentenceGeneratorConfig(model="gpt-4o-mini", temperature=0.2))

    result = gen.generate_query(
        dimension=dims[0], history=history, last_emotion="neutral", profile={"age": 72, "education_years": 9}
    )
    print("内部查询语句:")
    print(result["query"]) 
    print(f"used_fallback={result.get('used_fallback', False)} | confidence={result.get('confidence')} ")


if __name__ == "__main__":
    main()
