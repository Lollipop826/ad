"""
阿尔茨海默病患者抵抗情绪检测模型微调脚本 - 8类多分类版本

基于 Johnson8187/Chinese-Emotion 模型进行微调，
专门用于检测MMSE评估中患者的抵抗情绪类型。

标签体系（8类）：
0 - 正常配合（normal）
1 - 病耻感（shame）
2 - 否认病情（denial）
3 - 疲劳放弃（fatigue）
4 - 焦虑回避（anxiety）
5 - 愤怒对抗（anger）
6 - 虚构掩饰（confabulation）
7 - 不信任（distrust）
"""

import os
import json

# 国内镜像配置（必须在导入 transformers 之前设置）
USE_MIRROR = True  # 是否使用国内镜像
HF_MIRROR = "https://hf-mirror.com"  # HuggingFace 镜像地址

if USE_MIRROR:
    # 设置 HuggingFace 镜像环境变量（必须在导入任何 huggingface 相关库之前）
    os.environ['HF_ENDPOINT'] = HF_MIRROR
    # 同时设置其他可能需要的环境变量
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.expanduser('~/.cache/huggingface'))

import torch
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset

# 配置
BASE_MODEL = "Johnson8187/Chinese-Emotion"  # 基础模型
OUTPUT_DIR = "./models/ad_resistance_detector_multiclass"  # 微调后模型保存路径
DATA_FILE = "data/resistance_detection_multiclass.jsonl"  # 训练数据
NUM_LABELS = 8  # 8类分类

# 标签映射
LABEL_NAMES = {
    0: "normal",
    1: "shame", 
    2: "denial",
    3: "fatigue",
    4: "anxiety",
    5: "anger",
    6: "confabulation",
    7: "distrust",
}

LABEL_ZH = {
    0: "正常配合",
    1: "病耻感",
    2: "否认病情",
    3: "疲劳放弃",
    4: "焦虑回避",
    5: "愤怒对抗",
    6: "虚构掩饰",
    7: "不信任",
}


def load_data(file_path: str) -> List[Dict]:
    """加载训练数据"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            if sample.get('label', -1) >= 0:
                samples.append({
                    'text': sample['text'],
                    'label': sample['label']
                })
    return samples


def create_datasets(samples: List[Dict], test_size: float = 0.1, val_size: float = 0.1):
    """创建训练集、验证集和测试集"""
    texts = [s['text'] for s in samples]
    labels = [s['label'] for s in samples]
    
    # 先分出测试集
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # 再从训练验证集中分出验证集
    val_ratio = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=val_ratio, random_state=42, stratify=train_val_labels
    )
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    
    return train_dataset, val_dataset, test_dataset


def tokenize_function(examples, tokenizer):
    """分词函数"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
    )


def compute_metrics(eval_pred):
    """计算评估指标（多分类）"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 计算准确率
    accuracy = (predictions == labels).mean()
    
    # 计算宏平均和加权平均指标
    from sklearn.metrics import precision_recall_fscore_support
    
    # 宏平均（每个类别权重相同）
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    # 加权平均（按类别样本数加权）
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
    }


def main():
    print("=" * 60)
    print("🚀 阿尔茨海默病患者抵抗情绪检测模型微调 - 8类多分类")
    print("=" * 60)
    
    # 显示镜像配置
    if USE_MIRROR:
        print(f"\n🌐 使用国内镜像: {HF_MIRROR}")
        print(f"   HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
    
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n📌 使用设备: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. 加载数据
    print(f"\n📂 加载数据: {DATA_FILE}")
    samples = load_data(DATA_FILE)
    
    # 统计各类别数量
    label_counts = {}
    for s in samples:
        label = s['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"   总样本数: {len(samples)}")
    print(f"   各类别数量:")
    for label_id in sorted(label_counts.keys()):
        print(f"     [{label_id}] {LABEL_ZH[label_id]}: {label_counts[label_id]}")
    
    # 2. 加载预训练模型和分词器
    print(f"\n📦 加载基础模型: {BASE_MODEL}")
    
    # 确保使用镜像（在加载前再次确认环境变量）
    if USE_MIRROR:
        os.environ['HF_ENDPOINT'] = HF_MIRROR
        print(f"   镜像地址: {os.environ.get('HF_ENDPOINT', '未设置')}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # 创建标签映射
    id2label = {i: LABEL_ZH[i] for i in range(NUM_LABELS)}
    label2id = {LABEL_ZH[i]: i for i in range(NUM_LABELS)}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    )
    
    # 移动模型到设备
    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.to("mps")
    
    print(f"   ✅ 模型加载完成")
    
    # 3. 准备数据集
    print("\n📊 准备数据集...")
    train_dataset, val_dataset, test_dataset = create_datasets(samples, test_size=0.1, val_size=0.1)
    print(f"   - 训练集: {len(train_dataset)} 条")
    print(f"   - 验证集: {len(val_dataset)} 条")
    print(f"   - 测试集: {len(test_dataset)} 条")
    
    # 分词
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    
    # 数据收集器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 4. 训练参数
    print("\n⚙️ 配置训练参数...")
    
    # 根据设备调整训练参数
    if device == "cuda":
        fp16 = True
        batch_size = 32
    elif device == "mps":
        fp16 = False  # MPS不支持fp16
        batch_size = 16
    else:
        fp16 = False
        batch_size = 8
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # 训练参数
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # 评估和保存
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        
        # 日志
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="none",
        
        # 其他
        fp16=fp16,
        dataloader_num_workers=2 if device != "mps" else 0,
        use_mps_device=(device == "mps"),
    )
    
    print(f"   - 批量大小: {batch_size}")
    print(f"   - 学习率: 2e-5")
    print(f"   - 训练轮数: 5")
    print(f"   - FP16: {fp16}")
    
    # 5. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # 6. 开始训练
    print("\n🏃 开始训练...")
    print("-" * 40)
    trainer.train()
    
    # 7. 在测试集上评估
    print("\n📈 在测试集上评估...")
    test_results = trainer.evaluate(test_dataset)
    print(f"   准确率: {test_results['eval_accuracy']:.4f}")
    print(f"   宏平均 F1: {test_results['eval_macro_f1']:.4f}")
    print(f"   加权平均 F1: {test_results['eval_weighted_f1']:.4f}")
    
    # 8. 保存模型
    print(f"\n💾 保存模型到: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 保存标签映射
    label_config = {
        "label_names": LABEL_NAMES,
        "label_zh": LABEL_ZH,
        "num_labels": NUM_LABELS,
    }
    with open(f"{OUTPUT_DIR}/label_config.json", 'w', encoding='utf-8') as f:
        json.dump(label_config, f, ensure_ascii=False, indent=2)
    
    # 9. 详细分类报告
    print("\n📊 详细分类报告...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels_true = predictions.label_ids
    
    # 打印分类报告
    target_names = [f"{i}-{LABEL_ZH[i]}" for i in range(NUM_LABELS)]
    print(classification_report(labels_true, preds, target_names=target_names))
    
    # 10. 测试推理
    print("\n🧪 测试推理...")
    test_texts = [
        "好的，我试试",
        "别问了，我记性不好丢人",
        "我脑子好着呢，没病",
        "太累了，不想答了",
        "这个太难了，换一个吧",
        "问什么问，烦不烦",
        "我刚才不是说了吗",
        "你是谁派来的",
    ]
    
    from transformers import pipeline
    classifier = pipeline(
        "text-classification",
        model=OUTPUT_DIR,
        tokenizer=OUTPUT_DIR,
        device=0 if device == "cuda" else (-1 if device == "cpu" else "mps"),
    )
    
    print("-" * 50)
    for text in test_texts:
        result = classifier(text)[0]
        label = result['label']
        score = result['score']
        print(f"[{label}] ({score:.2f}) {text}")
    
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print(f"   模型保存在: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
