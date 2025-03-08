#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FOODPO数据模板"""

class FOODPOTemplate:
    """FOODPO数据模板"""

    def __init__(self):
        self.system_prompt = "你是一个专业的食品健康顾问，能够提供准确、科学的食品建议。"
    
    def check_equal(self, sample):
        """检查样本中的chosen和rejected是否相同"""
        if 'chosen' not in sample or 'rejected' not in sample:
            return False
        return sample['chosen'] == sample['rejected']
        
    def check_validation(self, sample):
        """验证样本是否有效"""
        return ('input' in sample and 'chosen' in sample and 'rejected' in sample and
                sample['input'] and sample['chosen'] and sample['rejected'])
    
    def format_preference_sample(self, sample):
        """格式化偏好样本"""
        # 创建对话格式
        human = "Human: "
        assistant = "Assistant: "
        
        # 优质回答
        better_conversation = f"{self.system_prompt}\n{human}{sample['input']}\n{assistant}{sample['chosen']}"
        
        # 较差回答
        worse_conversation = f"{self.system_prompt}\n{human}{sample['input']}\n{assistant}{sample['rejected']}"
        
        # 元信息
        meta_info = {
            "query": sample['input'],
            "better_response": sample['chosen'],
            "worse_response": sample['rejected']
        }
        
        return better_conversation, worse_conversation, meta_info 