#!/usr/bin/env python3
"""
增强版Markdown文本清洗工具
专门处理学术论文转换的MD文件中的乱码和格式问题
"""

import os
import re
import json
import argparse
from typing import List
from pathlib import Path


def clean_latex_math(text: str) -> str:
    """清除LaTeX数学公式"""
    # 行内公式 $...$
    text = re.sub(r'\$[^\$]+\$', '', text)
    # 行间公式 $$...$$
    text = re.sub(r'\$\$[^\$]+\$\$', '', text)
    return text


def clean_references(text: str) -> str:
    """清除文献引用标记"""
    # [1], [2-3], [1,2,3], [1][2][3]
    text = re.sub(r'\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\]', '', text)
    return text


def clean_reference_section(text: str) -> str:
    """清除完整的参考文献部分"""
    
    # 查找参考文献开始的位置
    patterns = [
        r'\n\s*参考文献[\s：:]*\n',
        r'\n\s*References[\s：:]*\n',
        r'\n\s*REFERENCES[\s：:]*\n',
        r'\n\s*文献[\s：:]*\n',
        r'\n\s*Bibliography[\s：:]*\n',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # 找到参考文献标题，删除从这里到文末的所有内容
            text = text[:match.start()]
            break
    
    # 另一种方式：检测文末的引用格式行（作者. 标题[J]. 期刊...）
    # 统计文末有多少条参考文献，如果>=3条，全部删除
    lines = text.split('\n')
    
    # 扩展参考文献检测模式，包括更多格式
    reference_patterns = [
        # 带标记的参考文献
        r'^[A-Z][^。]+?[\.,]\s+.+?\[J\]',  # 英文参考文献 [J]
        r'^[A-Z][^。]+?[\.,]\s+.+?\[M\]',  # 英文参考文献 [M]
        r'^[A-Z][^。]+?[\.,]\s+.+?\[D\]',  # 英文参考文献 [D]
        r'^[A-Z][^。]+?[\.,]\s+.+?\[EB/OL\]',  # 英文参考文献 [EB/OL]
        r'^[\u4e00-\u9fa5]{2,}[\.,，].+?[\[［][JMND][\]］]',  # 中文参考文献 [J][M][N][D]
        r'^[\u4e00-\u9fa5]{2,}[\.,，].+?[\[［]EB/OL[\]］]',  # 中文参考文献 [EB/OL]
        
        # 不带标记但格式明显的参考文献
        r'^[\u4e00-\u9fa5]{2,}[\.,，].+?\[\d{4}\]',  # 中文：作者. 标题[年份]
        r'^[A-Z][^。]+?[\.,].*?\(\d{4}\)',  # 英文：Author. Title (2020)
        r'^[A-Z][a-zA-Z\s]+,\s+[A-Z][a-zA-Z\s]+[\.,].+?\d{4}',  # Author A, Author B. Title. 2020
        r'^[A-Z][^。]{10,100}[\.,]\s+[A-Z].+?\d{4}[;,\.]',  # 标准英文文献格式
        
        # 编号格式
        r'^\d+[\.\)]\s*[\u4e00-\u9fa5]{2,}[\.,，]',  # 编号的中文参考文献
        r'^\[\d+\]\s*[\u4e00-\u9fa5]{2,}[\.,，]',  # [1] 格式的中文参考文献
        r'^\[\d+\]\s*[A-Z]',  # [1] 格式的英文参考文献
        
        # 特殊格式
        r'^[A-Z][A-Z\s]+[A-Z][\.,]\s+.+?\d{4}',  # 全大写作者名
        r'^\[PubMed',  # PubMed标记
        r'^doi[:：]\s*\d',  # DOI标记
    ]
    
    # 从后往前扫描，找到第一个参考文献的位置
    first_ref_idx = None
    ref_count = 0
    
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue  # 跳过空行
        
        is_reference = False
        for pattern in reference_patterns:
            if re.search(pattern, line):
                is_reference = True
                break
        
        if is_reference:
            ref_count += 1
            first_ref_idx = i  # 持续更新，最终得到最前面的参考文献位置
        elif ref_count >= 3:
            # 已经找到至少3条参考文献，且遇到非参考文献行，说明参考文献部分结束
            break
    
    # 如果文末有至少2条参考文献（降低阈值），删除从第一条开始的所有内容
    if ref_count >= 2 and first_ref_idx is not None:
        text = '\n'.join(lines[:first_ref_idx])
    
    return text


def clean_inline_references(text: str) -> str:
    """清除混在段落中的参考文献行 - 更保守的策略"""
    lines = text.split('\n')
    cleaned_lines = []
    
    # 更严格的参考文献特征判断
    def is_likely_reference(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) < 15:  # 太短的不算
            return False
        
        # 强特征：明确的文献标记
        strong_indicators = [
            r'[\[［][JMND][\]］]',  # [J][M][N][D]标记
            r'[\[［]EB/OL[\]］]',  # [EB/OL]标记
            r'\[PubMed',  # PubMed
            r'^doi[:：\s]+10\.',  # DOI
        ]
        
        for pattern in strong_indicators:
            if re.search(pattern, stripped):
                return True
        
        # 中等特征：作者格式 + 年份（需要组合判断）
        # 英文：Author A, Author B. Title. 2020.
        if re.search(r'^[A-Z][a-z]+\s+[A-Z][a-z]*,\s+[A-Z][a-z]+\s+[A-Z][a-z]*[\.,]', stripped):
            if re.search(r'\d{4}[;,\.\)]', stripped):
                return True
        
        # 中文：作者. 标题. 期刊，年份
        # 但要确保有期刊、年份等多个特征
        if re.search(r'^[\u4e00-\u9fa5]{2,5}[\.,，]', stripped):
            # 必须同时包含年份和期刊号/卷号
            has_year = re.search(r'[，,]\s*\d{4}', stripped)
            has_volume = re.search(r'\d+\(\d+\)[:：]', stripped) or re.search(r',\s*\d+\(\d+\)', stripped)
            if has_year and has_volume:
                return True
        
        return False
    
    for line in lines:
        if not is_likely_reference(line):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_superscript_numbers(text: str) -> str:
    """清除上标数字（通常是作者标注）"""
    # Unicode上标数字
    text = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+', '', text)
    # 可能的上标格式
    text = re.sub(r'(?<=[\u4e00-\u9fa5])[¹²³⁴⁵⁶⁷⁸⁹]+', '', text)
    # HTML上标标签
    text = re.sub(r'<sup>\d+</sup>', '', text)
    return text


def clean_metadata_lines(text: str) -> str:
    """清除元数据行"""
    patterns = [
        r'^中图分类号[：:].+$',
        r'^文献标志码[：:].+$', 
        r'^文章编号[：:].+$',
        r'^DOI[：:].+$',
        r'^doi[：:].+$',
        r'^\[摘要\].+$',
        r'^\[关键词\].+$',
        r'^\[Abstract\].+$',
        r'^\[Keywords\].+$',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text


def clean_author_info(text: str) -> str:
    """清除作者信息（包含单位编号的长串）"""
    # 匹配包含多个中文姓名和单位编号的行
    text = re.sub(r'^[^。！？\n]{0,200}[\u4e00-\u9fa5]{2,4}[¹²³⁴⁵⁶⁷⁸⁹\d]{1,2}[，,\s]+[\u4e00-\u9fa5]{2,4}[¹²³⁴⁵⁶⁷⁸⁹\d]{1,2}.{0,200}$', '', text, flags=re.MULTILINE)
    
    # 匹配单位信息行（括号内包含地址和邮编）
    text = re.sub(r'^\([^)]{0,500}[\d]{6}[^)]{0,200}\)$', '', text, flags=re.MULTILINE)
    
    return text


def clean_urls(text: str) -> str:
    """清除URL链接"""
    # http/https链接
    text = re.sub(r'https?://[^\s\)]+', '', text)
    # www链接
    text = re.sub(r'www\.[^\s\)]+', '', text)
    return text


def clean_html_tables(text: str) -> str:
    """清除HTML表格标签"""
    # 删除完整的HTML表格（包括内容）
    text = re.sub(r'<table>.*?</table>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 删除可能残留的单个HTML标签
    text = re.sub(r'</?(?:table|tr|td|th|thead|tbody|tfoot)[^>]*>', '', text, flags=re.IGNORECASE)
    
    return text


def clean_conference_notices(text: str) -> str:
    """清除会议通知、征文通知等无关内容"""
    # 常见的会议通知关键词
    keywords = [
        r'征文通知',
        r'会议通知', 
        r'学术会议',
        r'论坛.*?召开',
        r'投稿.*?截稿',
        r'截稿日期',
        r'网上论文投稿',
        r'主办.*?承办',
        r'协办',
        r'征文.*?为未公开发表',
        r'健康管理.*?会议',
        r'中华医学会',
        r'本次大会',
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    skip_mode = False
    skip_counter = 0  # 跳过模式的行计数
    
    for i, line in enumerate(lines):
        # 检测是否包含会议通知关键词
        is_notice = False
        for keyword in keywords:
            if re.search(keyword, line):
                is_notice = True
                skip_mode = True
                skip_counter = 0
                break
        
        # 如果进入跳过模式
        if skip_mode:
            skip_counter += 1
            # 如果空行或持续跳过超过15行，尝试恢复
            if line.strip() == '' or skip_counter > 15:
                # 查看接下来的几行
                future_lines_clean = True
                for j in range(i + 1, min(i + 3, len(lines))):
                    future_line = lines[j].strip()
                    if future_line:
                        for keyword in keywords:
                            if re.search(keyword, future_line):
                                future_lines_clean = False
                                break
                    if not future_lines_clean:
                        break
                
                if future_lines_clean and skip_counter > 3:
                    skip_mode = False
                    skip_counter = 0
            continue
        
        if not is_notice:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_special_markers(text: str) -> str:
    """清除特殊标记"""
    # 图表标注
    text = re.sub(r'^[\s]*[图表]\s*\d+[：:].+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*Fig(?:ure)?\s*\d+[：:].+$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^[\s]*Table\s*\d+[：:].+$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # 页眉页脚标记
    text = re.sub(r'^[-·•]\s*.*·\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^特别报道·$', '', text, flags=re.MULTILINE)
    
    return text


def clean_markdown_formatting(text: str) -> str:
    """清除Markdown格式标记（保留内容）"""
    # 标题标记
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # 加粗
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # 斜体
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # 代码块标记
    text = re.sub(r'```[^\n]*\n', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    return text


def clean_extra_whitespace(text: str) -> str:
    """清除多余空白"""
    # 多个空行合并为一个
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # 行首行尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # 多个连续空格
    text = re.sub(r'  +', ' ', text)
    
    return text


def clean_academic_markdown(text: str, aggressive: bool = False) -> str:
    """
    综合清洗学术Markdown文本
    
    Args:
        text: 原始文本
        aggressive: 是否使用激进模式（会清除更多内容）
    """
    # 基础清洗
    text = clean_html_tables(text)  # 先清除HTML表格
    text = clean_reference_section(text)  # 清除参考文献部分
    text = clean_inline_references(text)  # 清除混在正文中的参考文献行
    text = clean_conference_notices(text)  # 清除会议通知等无关内容
    text = clean_latex_math(text)
    text = clean_references(text)
    text = clean_superscript_numbers(text)
    text = clean_urls(text)
    text = clean_metadata_lines(text)
    text = clean_special_markers(text)
    
    # 激进模式
    if aggressive:
        text = clean_author_info(text)
        text = clean_markdown_formatting(text)
    
    # 最后清理空白
    text = clean_extra_whitespace(text)
    
    return text


def process_markdown_file(
    input_path: str,
    output_path: str,
    aggressive: bool = False,
    min_line_length: int = 10
) -> dict:
    """处理单个Markdown文件"""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    # 清洗
    cleaned_text = clean_academic_markdown(original_text, aggressive=aggressive)
    
    # 按段落分割并过滤短行
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    paragraphs = [p for p in paragraphs if len(p) >= min_line_length]
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(paragraphs))
    
    return {
        'input': input_path,
        'output': output_path,
        'original_size': len(original_text),
        'cleaned_size': len('\n\n'.join(paragraphs)),
        'paragraph_count': len(paragraphs)
    }


def main():
    parser = argparse.ArgumentParser(description='清洗学术Markdown文件')
    parser.add_argument('--input_dir', type=str, default='kb/mk', help='输入目录')
    parser.add_argument('--output_dir', type=str, default='kb/mk_cleaned', help='输出目录')
    parser.add_argument('--aggressive', action='store_true', help='激进清洗模式')
    parser.add_argument('--min_length', type=int, default=10, help='最小行长度')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理所有MD文件
    input_dir = Path(args.input_dir)
    results = []
    
    for md_file in input_dir.glob('*.md'):
        output_file = Path(args.output_dir) / md_file.name
        
        print(f'处理: {md_file.name}')
        result = process_markdown_file(
            str(md_file),
            str(output_file),
            aggressive=args.aggressive,
            min_line_length=args.min_length
        )
        results.append(result)
        
        print(f'  原始大小: {result["original_size"]} bytes')
        print(f'  清洗后: {result["cleaned_size"]} bytes')
        print(f'  段落数: {result["paragraph_count"]}')
    
    # 保存统计
    stats_file = Path(args.output_dir) / 'cleaning_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f'\n清洗完成！共处理 {len(results)} 个文件')
    print(f'输出目录: {args.output_dir}')
    print(f'统计文件: {stats_file}')


if __name__ == '__main__':
    main()
