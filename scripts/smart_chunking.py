#!/usr/bin/env python3
"""
智能语义分块工具 - 针对医学文献优化
策略：语义段落分块 + 重叠
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class SmartChunker:
    """智能分块器"""
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        min_chunk_size: int = 150,
        max_chunk_size: int = 600,
    ):
        """
        初始化分块器
        
        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块之间的重叠大小
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 中文句子分隔符
        sentences = re.split(r'([。！？\.!?])', text)
        
        # 重组句子（将分隔符附加到句子末尾）
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            separator = sentences[i + 1] if i + 1 < len(sentences) else ''
            if sentence:
                result.append(sentence + separator)
        
        # 处理最后一个可能没有分隔符的句子
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result
    
    def create_chunks_with_overlap(self, text: str) -> List[Dict]:
        """
        创建带重叠的语义块
        
        Returns:
            List of chunks with metadata
        """
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        sentence_start_idx = 0  # 当前块的起始句子索引
        
        for i, sentence in enumerate(sentences):
            # 尝试添加当前句子
            potential_chunk = current_chunk + sentence
            
            # 判断是否需要创建新块
            if len(potential_chunk) >= self.chunk_size:
                # 如果当前块已经达到最小大小，保存它
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_index': chunk_index,
                        'start_sentence': sentence_start_idx,
                        'end_sentence': i - 1,
                        'char_count': len(current_chunk),
                    })
                    
                    # 计算重叠部分
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = overlap_text + sentence
                    chunk_index += 1
                    sentence_start_idx = i
                else:
                    # 当前块太小，继续添加
                    current_chunk = potential_chunk
            else:
                current_chunk = potential_chunk
            
            # 如果块超过最大大小，强制分割
            if len(current_chunk) > self.max_chunk_size:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'start_sentence': sentence_start_idx,
                    'end_sentence': i,
                    'char_count': len(current_chunk),
                })
                
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text
                chunk_index += 1
                sentence_start_idx = i + 1
        
        # 添加最后一个块
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                'start_sentence': sentence_start_idx,
                'end_sentence': len(sentences) - 1,
                'char_count': len(current_chunk),
            })
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """获取文本末尾的重叠部分"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # 从末尾取overlap_size的文本
        overlap = text[-self.chunk_overlap:]
        
        # 尝试从完整句子开始
        # 找到第一个句号、问号或感叹号之后的位置
        match = re.search(r'[。！？\.!?]\s*', overlap)
        if match:
            return overlap[match.end():]
        
        return overlap
    
    def extract_metadata(self, text: str, filename: str) -> Dict:
        """提取文档元数据"""
        metadata = {
            'filename': filename,
            'title': '',
            'abstract': '',
            'keywords': [],
        }
        
        # 提取标题（通常在第一行或包含"摘要"的前面）
        lines = text.split('\n')
        for i, line in enumerate(lines[:5]):  # 只看前5行
            if '摘要' in line or '【摘要】' in line:
                if i > 0:
                    metadata['title'] = lines[i-1].strip()
                break
        
        # 提取摘要
        abstract_match = re.search(r'【?摘要】?[：:]?\s*(.+?)(?=【?关键词|$)', text, re.DOTALL)
        if abstract_match:
            metadata['abstract'] = abstract_match.group(1).strip()[:500]  # 限制长度
        
        # 提取关键词
        keywords_match = re.search(r'【?关键词】?[：:]?\s*(.+?)(?=\n|$)', text)
        if keywords_match:
            keywords_text = keywords_match.group(1)
            metadata['keywords'] = [
                kw.strip() for kw in re.split(r'[；;，,]', keywords_text)
                if kw.strip()
            ]
        
        return metadata
    
    def process_document(
        self,
        text: str,
        filename: str,
        source_path: str = None
    ) -> List[Dict]:
        """
        处理单个文档
        
        Args:
            text: 文档文本
            filename: 文件名
            source_path: 源文件路径
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # 提取文档级元数据
        doc_metadata = self.extract_metadata(text, filename)
        
        # 创建分块
        chunks = self.create_chunks_with_overlap(text)
        
        # 为每个块添加完整元数据
        result = []
        for chunk in chunks:
            chunk_dict = {
                'text': chunk['text'],
                'metadata': {
                    'source': source_path or filename,
                    'filename': filename,
                    'title': doc_metadata['title'],
                    'chunk_index': chunk['chunk_index'],
                    'char_count': chunk['char_count'],
                    'chunking_strategy': 'semantic_overlap',
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                }
            }
            
            # 如果是第一个块，添加摘要和关键词
            if chunk['chunk_index'] == 0:
                chunk_dict['metadata']['abstract'] = doc_metadata['abstract']
                chunk_dict['metadata']['keywords'] = doc_metadata['keywords']
            
            result.append(chunk_dict)
        
        return result


def process_markdown_files(
    input_dir: str,
    output_file: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    min_chunk_size: int = 150,
    max_chunk_size: int = 600,
):
    """处理目录中的所有Markdown文件"""
    
    chunker = SmartChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )
    
    input_path = Path(input_dir)
    all_chunks = []
    stats = {
        'total_files': 0,
        'total_chunks': 0,
        'avg_chunk_size': 0,
        'files': []
    }
    
    # 处理所有MD文件
    for md_file in sorted(input_path.glob('*.md')):
        print(f'处理: {md_file.name}')
        
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            print(f'  跳过空文件')
            continue
        
        # 创建chunks
        chunks = chunker.process_document(
            text=text,
            filename=md_file.name,
            source_path=str(md_file)
        )
        
        all_chunks.extend(chunks)
        
        # 统计
        file_stat = {
            'filename': md_file.name,
            'chunks': len(chunks),
            'avg_size': sum(c['metadata']['char_count'] for c in chunks) / len(chunks) if chunks else 0,
        }
        stats['files'].append(file_stat)
        stats['total_files'] += 1
        stats['total_chunks'] += len(chunks)
        
        print(f'  生成 {len(chunks)} 个块, 平均大小: {file_stat["avg_size"]:.0f} 字符')
    
    # 计算总体统计
    if all_chunks:
        stats['avg_chunk_size'] = sum(
            c['metadata']['char_count'] for c in all_chunks
        ) / len(all_chunks)
    
    # 保存到JSONL
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    # 保存统计信息
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f'\n✅ 处理完成！')
    print(f'📊 统计信息:')
    print(f'  - 处理文件数: {stats["total_files"]}')
    print(f'  - 生成块数: {stats["total_chunks"]}')
    print(f'  - 平均块大小: {stats["avg_chunk_size"]:.0f} 字符')
    print(f'  - 输出文件: {output_file}')
    print(f'  - 统计文件: {stats_file}')


def main():
    parser = argparse.ArgumentParser(
        description='智能语义分块工具 - 针对医学文献优化'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='kb/mk_final_cleaned',
        help='输入目录（默认: kb/mk_final_cleaned）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='kb/chunks_semantic.jsonl',
        help='输出JSONL文件（默认: kb/chunks_semantic.jsonl）'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=400,
        help='目标块大小（默认: 400字符）'
    )
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=80,
        help='块重叠大小（默认: 80字符）'
    )
    parser.add_argument(
        '--min_chunk_size',
        type=int,
        default=150,
        help='最小块大小（默认: 150字符）'
    )
    parser.add_argument(
        '--max_chunk_size',
        type=int,
        default=600,
        help='最大块大小（默认: 600字符）'
    )
    
    args = parser.parse_args()
    
    process_markdown_files(
        input_dir=args.input_dir,
        output_file=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
    )


if __name__ == '__main__':
    main()
