"""
对话管理器

完整的对话流程管理：
1. 接收用户输入
2. 生成查询语句
3. 检索相关知识
4. 基于知识生成下一个问题
5. 保存对话记录
"""

from typing import List, Dict, Any, Optional
import logging

from src.common.types import InfoDimension, Profile
from src.common.conversation_storage import ConversationStorage, RetrievalResult
from src.tools.query_sentence.generator import QuerySentenceGenerator
from src.tools.retrieval.paragraph_retrieval import paragraph_retrieval
from src.tools.retrieval.sentence_filter import SentenceFilter, split_sentences
from src.tools.dialogue.question_generator import QuestionGenerator


class ConversationManager:
    """对话管理器"""
    
    def __init__(
        self,
        storage_dir: str = "data/conversations",
        vector_db_dir: str = "kb/.chroma_semantic",
        collection_name: str = "ad_kb_semantic",
        embedding_model: str = "BAAI/bge-m3",
    ):
        self.storage = ConversationStorage(storage_dir=storage_dir)
        self.query_generator = QuerySentenceGenerator()
        self.question_generator = QuestionGenerator()
        self.sentence_filter = SentenceFilter()
        
        self.vector_db_dir = vector_db_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Logger
        self.logger = logging.getLogger("dialogue.conversation_manager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def process_turn(
        self,
        session_id: str,
        user_input: str,
        current_dimension: InfoDimension,
        user_emotion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理一轮对话
        
        Args:
            session_id: 会话ID
            user_input: 用户输入/回答
            current_dimension: 当前评估的维度
            user_emotion: 用户情绪
        
        Returns:
            {
                'next_question': 下一个问题,
                'generated_query': 生成的查询,
                'retrieved_docs': 检索到的文档数量,
                'turn_id': 对话轮次ID
            }
        """
        
        self.logger.info(f"处理会话 {session_id} 的新轮次")
        self.logger.info(f"用户输入: {user_input}")
        self.logger.info(f"当前维度: {current_dimension.get('name')} ({current_dimension.get('id')})")
        
        # 1. 加载会话
        session = self.storage.load_session(session_id)
        profile = session.get("profile")
        
        # 2. 获取对话历史
        history = self.storage.get_conversation_history(session_id, max_turns=5)
        
        # 3. 生成查询语句
        self.logger.info("步骤1: 生成检索查询...")
        query_result = self.query_generator.generate_query(
            dimension=current_dimension,
            history=history,
            last_emotion=user_emotion,
            profile=profile
        )
        generated_query = query_result["query"]
        keywords = query_result.get("keywords", [])
        
        self.logger.info(f"  生成的查询: {generated_query}")
        self.logger.info(f"  关键词: {', '.join(keywords)}")
        
        # 4. 检索相关知识
        self.logger.info("步骤2: 检索相关知识...")
        docs = paragraph_retrieval(
            query=generated_query,
            persist_dir=self.vector_db_dir,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            k=10  # 召回更多候选
        )
        
        self.logger.info(f"  检索到 {len(docs)} 个候选文档")
        
        # 5. 准备检索结果（带句子高亮）
        retrieved_docs: List[RetrievalResult] = []
        knowledge_texts: List[str] = []
        
        import torch
        for i, doc in enumerate(docs[:5], 1):  # 只取前5个
            # 计算句子得分
            sentences = split_sentences(doc.page_content)
            if not sentences:
                continue
            
            pairs = [[generated_query, s] for s in sentences]
            with torch.inference_mode():
                scores = self.sentence_filter.model.predict(pairs).tolist()
            
            # 高亮相关句子
            highlighted = [
                {"text": s, "score": float(score), "is_relevant": score >= 0.6}
                for s, score in zip(sentences, scores)
            ]
            
            retrieved_docs.append({
                "rank": i,
                "text": doc.page_content,
                "metadata": doc.metadata,
                "highlighted_sentences": highlighted
            })
            
            # 提取高相关性的句子用于问题生成
            relevant_sentences = [s for s, score in zip(sentences, scores) if score >= 0.5]
            if relevant_sentences:
                knowledge_texts.append("\n".join(relevant_sentences[:3]))  # 最多3句
            else:
                knowledge_texts.append(doc.page_content[:300])  # 没有高相关句子时用开头
        
        self.logger.info(f"  准备了 {len(knowledge_texts)} 段知识文本")
        
        # 6. 生成下一个问题
        self.logger.info("步骤3: 生成下一个问题...")
        next_question = self.question_generator.generate_question(
            dimension=current_dimension,
            retrieved_knowledge=knowledge_texts[:3],  # 最多用3段知识
            profile=profile,
            conversation_history=history,
            last_emotion=user_emotion,
        )
        
        self.logger.info(f"  生成的问题: {next_question}")
        
        # 7. 保存对话轮次
        self.logger.info("步骤4: 保存对话轮次...")
        updated_session = self.storage.add_turn(
            session_id=session_id,
            user_question=user_input,
            generated_query=generated_query,
            query_keywords=keywords,
            retrieved_documents=retrieved_docs,
            assistant_response=next_question,
            dimension_id=current_dimension.get("id"),
            dimension_name=current_dimension.get("name"),
            user_emotion=user_emotion,
            retrieval_method="paragraph_with_highlight",
            response_metadata={
                "retrieval_count": len(docs),
                "confidence": query_result.get("confidence", 0.5),
                "knowledge_segments_used": len(knowledge_texts)
            }
        )
        
        turn_id = len(updated_session["dialogue_turns"])
        self.logger.info(f"✅ 对话轮次 {turn_id} 已保存")
        
        return {
            "next_question": next_question,
            "generated_query": generated_query,
            "query_keywords": keywords,
            "retrieved_docs": len(docs),
            "turn_id": turn_id,
            "session": updated_session
        }
    
    def create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
        dimensions: Optional[List[Dict[str, Any]]] = None,
    ):
        """创建新会话"""
        return self.storage.create_session(
            session_id=session_id,
            user_id=user_id,
            profile=profile,
            dimensions=dimensions
        )
    
    def get_session(self, session_id: str):
        """获取会话"""
        return self.storage.load_session(session_id)
    
    def update_dimension_status(
        self,
        session_id: str,
        dimension_id: str,
        status: str,
        value: Optional[str] = None
    ):
        """更新维度状态"""
        session = self.storage.load_session(session_id)
        dimensions = session["dimensions"]
        
        for dim in dimensions:
            if dim["id"] == dimension_id:
                dim["status"] = status
                if value is not None:
                    dim["value"] = value
                break
        
        return self.storage.update_dimensions(session_id, dimensions)
    
    def export_session(self, session_id: str, output_file: str):
        """导出会话"""
        self.storage.export_session_to_jsonl(session_id, output_file)

