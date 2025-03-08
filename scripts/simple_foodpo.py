#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的FOODPO（食物偏好优化）实现脚本
该脚本基于DPO算法思想，实现一个简单的食物偏好训练过程
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# 定义食物偏好数据集
class FoodPreferenceDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'input' in item and 'chosen' in item and 'rejected' in item:
                        self.examples.append(item)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析JSON行: {line}")
        
        print(f"加载了 {len(self.examples)} 条食物偏好数据")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 构建对话格式
        system_prompt = "你是一个专业的食品健康顾问，能够提供准确、科学的食品建议。"
        human = "Human: "
        assistant = "Assistant: "
        
        # 创建对话文本
        better_text = f"{system_prompt}\n{human}{example['input']}\n{assistant}{example['chosen']}"
        worse_text = f"{system_prompt}\n{human}{example['input']}\n{assistant}{example['rejected']}"
        
        # 对输入文本进行编码
        better_encoding = self.tokenizer(better_text, max_length=self.max_length, 
                                         padding='max_length', truncation=True, 
                                         return_tensors='pt')
        worse_encoding = self.tokenizer(worse_text, max_length=self.max_length, 
                                        padding='max_length', truncation=True, 
                                        return_tensors='pt')
        
        # 判断响应部分的长度
        better_response_text = example['chosen']
        worse_response_text = example['rejected']
        better_response_ids = self.tokenizer.encode(better_response_text, 
                                                   add_special_tokens=False)
        worse_response_ids = self.tokenizer.encode(worse_response_text, 
                                                  add_special_tokens=False)
        
        return {
            'better_input_ids': better_encoding['input_ids'].squeeze(0),
            'better_attention_mask': better_encoding['attention_mask'].squeeze(0),
            'worse_input_ids': worse_encoding['input_ids'].squeeze(0),
            'worse_attention_mask': worse_encoding['attention_mask'].squeeze(0),
            'better_response_length': len(better_response_ids),
            'worse_response_length': len(worse_response_ids),
            'query': example['input']
        }

# 简单的FOODPO训练器
class SimpleFOODPOTrainer:
    def __init__(self, model_name, data_file, output_dir, 
                 learning_rate=5e-5, batch_size=1, num_epochs=1, beta=0.1):
        self.model_name = model_name
        self.data_file = data_file
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta = beta
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型和tokenizer
        print(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 冻结参考模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
        # 将模型移动到GPU（如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.model.to(self.device)
        self.reference_model.to(self.device)
        
        # 准备数据集
        self.dataset = FoodPreferenceDataset(data_file, self.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # 设置优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
    def compute_loss(self, batch):
        """计算DPO损失"""
        # 处理输入
        better_input_ids = batch['better_input_ids'].to(self.device)
        better_attention_mask = batch['better_attention_mask'].to(self.device)
        worse_input_ids = batch['worse_input_ids'].to(self.device)
        worse_attention_mask = batch['worse_attention_mask'].to(self.device)
        
        # 获取响应部分的起始位置
        batch_size = better_input_ids.size(0)
        better_response_length = batch['better_response_length']
        worse_response_length = batch['worse_response_length']
        
        # 计算策略模型输出
        with torch.set_grad_enabled(True):
            better_outputs = self.model(input_ids=better_input_ids, 
                                       attention_mask=better_attention_mask)
            worse_outputs = self.model(input_ids=worse_input_ids,
                                      attention_mask=worse_attention_mask)
        
        # 计算参考模型输出
        with torch.no_grad():
            ref_better_outputs = self.reference_model(input_ids=better_input_ids,
                                                     attention_mask=better_attention_mask)
            ref_worse_outputs = self.reference_model(input_ids=worse_input_ids,
                                                    attention_mask=worse_attention_mask)
        
        losses = []
        for i in range(batch_size):
            try:
                # 使用更安全的方法来截取响应部分
                # 防止索引超出范围
                seq_len = better_input_ids.size(1)
                
                # 为了安全，限制响应长度不超过序列长度的一半
                safe_better_len = min(better_response_length[i], seq_len // 2)
                safe_worse_len = min(worse_response_length[i], seq_len // 2)
                
                # 获取响应对应的输入和标签
                # 使用安全的索引取值
                better_start_idx = seq_len - safe_better_len
                better_end_idx = seq_len
                
                worse_start_idx = seq_len - safe_worse_len
                worse_end_idx = seq_len
                
                # 输入为前一个token，标签为当前token
                better_logits = better_outputs.logits[i, better_start_idx-1:better_end_idx-1, :]
                better_labels = better_input_ids[i, better_start_idx:better_end_idx]
                
                worse_logits = worse_outputs.logits[i, worse_start_idx-1:worse_end_idx-1, :]
                worse_labels = worse_input_ids[i, worse_start_idx:worse_end_idx]
                
                # 确保logits和labels长度相同
                if better_logits.size(0) != better_labels.size(0):
                    print(f"警告: better响应的logits与labels长度不一致: {better_logits.size(0)} vs {better_labels.size(0)}")
                    # 取最小长度
                    min_len = min(better_logits.size(0), better_labels.size(0))
                    better_logits = better_logits[:min_len, :]
                    better_labels = better_labels[:min_len]
                
                if worse_logits.size(0) != worse_labels.size(0):
                    print(f"警告: worse响应的logits与labels长度不一致: {worse_logits.size(0)} vs {worse_labels.size(0)}")
                    # 取最小长度
                    min_len = min(worse_logits.size(0), worse_labels.size(0))
                    worse_logits = worse_logits[:min_len, :]
                    worse_labels = worse_labels[:min_len]
                
                # 计算log概率
                better_log_probs = self._compute_log_probs(better_logits, better_labels)
                worse_log_probs = self._compute_log_probs(worse_logits, worse_labels)
                
                # 参考模型的log概率
                ref_better_logits = ref_better_outputs.logits[i, better_start_idx-1:better_end_idx-1, :]
                ref_worse_logits = ref_worse_outputs.logits[i, worse_start_idx-1:worse_end_idx-1, :]
                
                # 确保参考模型的logits和labels长度相同
                if ref_better_logits.size(0) != better_labels.size(0):
                    min_len = min(ref_better_logits.size(0), better_labels.size(0))
                    ref_better_logits = ref_better_logits[:min_len, :]
                    better_labels = better_labels[:min_len]
                
                if ref_worse_logits.size(0) != worse_labels.size(0):
                    min_len = min(ref_worse_logits.size(0), worse_labels.size(0))
                    ref_worse_logits = ref_worse_logits[:min_len, :]
                    worse_labels = worse_labels[:min_len]
                
                ref_better_log_probs = self._compute_log_probs(ref_better_logits, better_labels)
                ref_worse_log_probs = self._compute_log_probs(ref_worse_logits, worse_labels)
                
                # 计算log比率
                better_log_ratio = better_log_probs.sum() - ref_better_log_probs.sum()
                worse_log_ratio = worse_log_probs.sum() - ref_worse_log_probs.sum()
                
                # 应用DPO损失函数
                loss = -torch.nn.functional.logsigmoid(self.beta * (better_log_ratio - worse_log_ratio))
                losses.append(loss)
                
            except Exception as e:
                print(f"警告: 处理样本 {i} 时出错: {e}")
                # 跳过有问题的样本
                continue
        
        # 确保至少有一个样本被成功处理
        if len(losses) == 0:
            print("错误: 所有样本处理失败，无法计算损失")
            # 返回一个零损失，防止训练中断
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算批次损失
        loss = torch.stack(losses).mean()
        return loss
    
    def _compute_log_probs(self, logits, labels):
        """计算token的对数概率"""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs
    
    def train(self):
        """训练模型"""
        print(f"开始训练，共 {self.num_epochs} 轮...")
        self.model.train()
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                if step % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Step {step}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.num_epochs} 完成, 平均损失: {avg_loss:.4f}")
            
            # 保存模型
            save_path = os.path.join(self.output_dir, f"foodpo_model_epoch_{epoch+1}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"模型已保存到 {save_path}")
        
        print("训练完成！")

def main():
    parser = argparse.ArgumentParser(description="简单的FOODPO模型训练脚本")
    parser.add_argument("--model_name", type=str, default="/root/models/Qwen1.5-0.5B", 
                        help="预训练模型路径或名称")
    parser.add_argument("--data_file", type=str, required=True, 
                        help="训练数据文件路径")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="输出目录")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="学习率")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=1, 
                        help="训练轮数")
    parser.add_argument("--beta", type=float, default=0.1, 
                        help="DPO损失的缩放系数")
    
    args = parser.parse_args()
    
    # 创建并运行训练器
    trainer = SimpleFOODPOTrainer(
        model_name=args.model_name,
        data_file=args.data_file,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        beta=args.beta
    )
    
    trainer.train()

if __name__ == "__main__":
    main() 