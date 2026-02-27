#!/usr/bin/env python3
"""
智能段落合并工具
解决MD文件中段落被错误拆分的问题
"""

import os
import re
import argparse
from pathlib import Path
from typing import List


def should_merge_with_next(current_line: str, next_line: str) -> bool:
    """
    判断当前行是否应该与下一行合并
    
    规则：
    1. 当前行以不完整结尾（无句号、问号、感叹号）
    2. 当前行以连词、介词结尾
    3. 当前行以数字、符号结尾但不完整
    4. 下一行以小写字母或中文开始（延续上文）
    5. 当前行无任何标点结尾（在句子中间断开）
    """
    
    # 跳过空行
    if not current_line.strip() or not next_line.strip():
        return False
    
    current = current_line.rstrip()
    next_start = next_line.lstrip()
    
    # 规则0：当前行完全无标点符号结尾，很可能是中间断开的，应该合并
    # 除非下一行明显是新段落的开始
    if not re.search(r'[，。！？、；：,.!?;:\)）」』"]$', current):
        # 检查下一行是否明显是新段落开始
        new_para_start = [
            r'^[一二三四五六七八九十\d]+[、\.．)]',  # 编号开始
            r'^第[一二三四五六七八九十\d]+[章节条款]',  # 章节开始
            r'^[\(（][一二三四五六七八九十\d]+[\)）]',  # 括号编号
            r'^[①②③④⑤⑥⑦⑧⑨⑩]',  # 圆圈数字
        ]
        if not any(re.match(pattern, next_start) for pattern in new_para_start):
            return True
    
    # 规则1：当前行以标准句尾符号结束，不合并
    if re.search(r'[。！？\.!?]$', current):
        # 除非下一行明显是延续（如：以"等"、"即"开头）
        if re.match(r'^[等即]', next_start):
            return True
        return False
    
    # 规则2：当前行以逗号、分号、冒号结束，可能需要合并
    if re.search(r'[，,；;：:]$', current):
        return True
    
    # 规则3：当前行以连词结尾
    conjunctions = ['与', '和', '或', '及', '以及', '并', '且', '而', '但', '的', '地', '得']
    if any(current.endswith(conj) for conj in conjunctions):
        return True
    
    # 规则4：当前行以数字+中文数量单位结尾，很可能不完整
    # 如："1677.4亿"后面应该接"美元"
    # 或"占全国人口的" 后面应该接百分比
    if re.search(r'(?:\d+(?:\.\d+)?(?:亿|万|千|百|十|多)?|的\s*)$', current):
        return True
    
    # 规则5：当前行以括号开始但未闭合
    if current.count('(') > current.count(')') or current.count('（') > current.count('）'):
        return True
    
    # 规则6：当前行以引号开始但未闭合
    if current.count('"') % 2 == 1 or current.count('"') % 2 == 1:
        return True
    if current.count('「') > current.count('」') or current.count('『') > current.count('』'):
        return True
    
    # 规则7：当前行太短（可能是被错误拆分的）
    if len(current) < 15:
        # 如果下一行不是以明显的段落开始标志开头，则合并
        if not re.match(r'^[一二三四五六七八九十0-9\(（]', next_start):
            return True
    
    # 规则8：当前行以"在"、"于"、"从"、"向"、"对"等介词结尾
    prepositions = ['在', '于', '从', '向', '对', '为', '由', '经', '按', '据', '依', '根据']
    if any(current.endswith(prep) for prep in prepositions):
        return True
    
    # 规则9：下一行以小写英文字母开始（明显是延续）
    if re.match(r'^[a-z]', next_start):
        return True
    
    return False


def merge_broken_paragraphs(lines: List[str]) -> List[str]:
    """智能合并被错误拆分的段落"""
    
    if not lines:
        return []
    
    # 清理所有行，去掉首尾空白
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    if not cleaned_lines:
        return []
    
    merged = []
    i = 0
    
    while i < len(cleaned_lines):
        # 开始新段落
        paragraph = cleaned_lines[i]
        i += 1
        
        # 持续合并后续行，直到遇到不应该合并的行
        while i < len(cleaned_lines):
            if should_merge_with_next(paragraph, cleaned_lines[i]):
                # 合并时添加适当的空格
                if paragraph and not paragraph[-1] in '，。！？、；：,.!?;:':
                    paragraph += ''  # 不加空格，直接连接
                paragraph += cleaned_lines[i]
                i += 1
            else:
                # 不合并，当前段落结束
                break
        
        # 添加合并后的段落
        if paragraph:
            merged.append(paragraph)
    
    return merged


def split_long_paragraphs(paragraph: str, max_length: int = 500) -> List[str]:
    """
    拆分过长的段落
    在句号、问号、感叹号处拆分
    """
    if len(paragraph) <= max_length:
        return [paragraph]
    
    # 找到所有句子结尾
    sentences = re.split(r'([。！？\.!?])', paragraph)
    
    result = []
    current = ""
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        ending = sentences[i + 1] if i + 1 < len(sentences) else ""
        
        if len(current) + len(sentence) + len(ending) <= max_length:
            current += sentence + ending
        else:
            if current:
                result.append(current)
            current = sentence + ending
    
    if current:
        result.append(current)
    
    return result if result else [paragraph]


def remove_reference_paragraphs(paragraphs: List[str]) -> List[str]:
    """删除参考文献段落"""
    reference_pattern = r'[A-Z][^。]+?[\.,]\s+.+?\[J\]'  # 英文参考文献
    chinese_ref_pattern = r'[\u4e00-\u9fa5]{2,}[\.,，].+?[\[［]J[\]］]'  # 中文参考文献
    
    cleaned = []
    for para in paragraphs:
        # 如果段落主要是参考文献格式，跳过
        if re.search(reference_pattern, para) or re.search(chinese_ref_pattern, para):
            continue
        cleaned.append(para)
    
    return cleaned


def process_file(input_path: str, output_path: str, max_para_length: int = 500):
    """处理单个文件"""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 智能合并段落
    merged_paragraphs = merge_broken_paragraphs(lines)
    
    # 删除参考文献段落
    merged_paragraphs = remove_reference_paragraphs(merged_paragraphs)
    
    # 拆分过长段落
    final_paragraphs = []
    for para in merged_paragraphs:
        final_paragraphs.extend(split_long_paragraphs(para, max_para_length))
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(final_paragraphs))
    
    return {
        'input': input_path,
        'output': output_path,
        'original_lines': len(lines),
        'merged_paragraphs': len(merged_paragraphs),
        'final_paragraphs': len(final_paragraphs)
    }


def main():
    parser = argparse.ArgumentParser(description='智能合并Markdown段落')
    parser.add_argument('--input_dir', type=str, default='kb/mk_cleaned', help='输入目录')
    parser.add_argument('--output_dir', type=str, default='kb/mk_merged', help='输出目录')
    parser.add_argument('--max_length', type=int, default=500, help='最大段落长度')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_dir = Path(args.input_dir)
    results = []
    
    for md_file in input_dir.glob('*.md'):
        output_file = Path(args.output_dir) / md_file.name
        
        print(f'处理: {md_file.name}')
        result = process_file(str(md_file), str(output_file), args.max_length)
        results.append(result)
        
        print(f'  原始行数: {result["original_lines"]}')
        print(f'  合并后段落: {result["merged_paragraphs"]}')
        print(f'  最终段落: {result["final_paragraphs"]}')
    
    print(f'\n处理完成！共处理 {len(results)} 个文件')
    print(f'输出目录: {args.output_dir}')


if __name__ == '__main__':
    main()
