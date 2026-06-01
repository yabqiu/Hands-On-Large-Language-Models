import kick_zscaler

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

# 在 macOS 下还能用 load_dataset("rotten_tomatoes") 加载数据，在 Linux 下必须加上 namespace
tomatoes = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train_data, test_data = tomatoes["train"], tomatoes["test"] # 各 8530， 1066 条数据
# 训练集和测试集又各自有一半是正面评价(label=1，一半是负面评价(label=0)

model_id = "bert-base-cased"

# num_label=2 表示二分类任务，BERT 顶层会接一个 2 维分类头
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 对批次中的序列进行填充，使其长度与最长序列一致
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# batch 后 example 的格式为 {"text": ["sentence1", "sentence2", ...], "label": [0, 1, ...]}
def preprocess_function(examples):
    # 对 examples.text 中的文本进行分词后编码，每段文本编码后长度不一
    return tokenizer(examples["text"], truncation=True)

# 对训练数据和测试数据进行分词处理
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

# 这是用来观测，不参与训练过程，也就不会影响训练结果
def compute_metrics(eval_pred):
    """
    计算 F1 分数(F-score: F1 = F-measure with β=1), 它是衡量精确率和召回率的综合指标，以正面为例
    精确率(Precision)为预测为正面的样本，真正是正面的比例; 召回率(Recall)为所有是正面的样本，被正确预测为正面的比例
    """
    logits, labels = eval_pred  # 分别为 eval_pred.elements 中两组数据

    # argmax 是返回最大值所在的索引，如 logits=[[-1.2, 2.5], [3.1, -0.8], [0.3, 1.9]] 表示负面和正面评分, 最后它返回 [1, 0, 1]
    predictions = np.argmax(logits, axis=-1)

    # 预测的结果 predictions 与实际的标签 labels 对比, 算出 F1 分数
    load_f1 = evaluate.load("f1")
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"f1": f1}

# 用于参数调估的训练参数
training_args = TrainingArguments(
    "sentiment_model",
    learning_rate=2e-5,    # BERT 微调的典型学习率
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,     # L2 正则化，防止过拟合
    save_strategy="epoch", # 每轮结束保存一次
    eval_strategy="epoch", # 每轮训练后自动评估并调用 compute_metrics
    report_to="none"
)

# 执行训练过程的 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class = tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics, # 前面没有配置 eval_strategy, compute_metrics 不会被调用
)

trainer.train()
trainer.save_model()