#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
准备FOODPO数据集的脚本
"""

import os
import json
import argparse

def create_example_dataset(output_dir, num_examples=10):
    """创建示例数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    examples = [
        {
            "input": "哪些食物对健康有益？", 
            "chosen": "水果、蔬菜、全谷物、瘦肉和鱼类都是对健康有益的食物。它们富含维生素、矿物质和纤维，有助于维持身体健康。", 
            "rejected": "快餐、甜食和加工食品都挺好的，偶尔吃一些没关系的。"
        },
        {
            "input": "请推荐一些减肥食物", 
            "chosen": "减肥期间可以多吃低热量高纤维的食物，如西兰花、菠菜、鸡胸肉、鱼肉等。这些食物热量低但饱腹感强，有助于控制总热量摄入。", 
            "rejected": "减肥期间可以吃一些米饭、面条，只要少吃点就行。"
        },
        {
            "input": "素食主义者应该吃什么？", 
            "chosen": "素食主义者可以通过豆类、坚果、种子、豆腐、蘑菇和全谷物获取足够的蛋白质和其他营养物质。重要的是确保饮食多样化，以获取所有必需的营养素。", 
            "rejected": "素食主义者基本上只能吃素食，比如青菜。不过这样营养不够全面，可能会导致健康问题。"
        },
        {
            "input": "哪些食物对心脏健康有益？", 
            "chosen": "富含omega-3脂肪酸的食物如三文鱼、沙丁鱼等鱼类，以及橄榄油、坚果、蔬菜水果和全谷物都对心脏健康有益。这些食物可以帮助降低炎症和改善血脂水平。", 
            "rejected": "只要少吃油炸食品就可以了，其他食物都差不多，没什么特别需要注意的。"
        }
    ]
    
    # 重复示例直到达到所需数量
    while len(examples) < num_examples:
        examples.extend(examples[:num_examples-len(examples)])
    
    # 保存到JSON文件
    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"已创建示例数据集，共{len(examples)}条数据，保存在{os.path.join(output_dir, 'train.json')}")
    
    return os.path.join(output_dir, 'train.json')

def parse_args():
    parser = argparse.ArgumentParser(description='准备FOODPO数据集')
    parser.add_argument('--output_dir', type=str, default='../data/foodpo_data', help='输出目录')
    parser.add_argument('--num_examples', type=int, default=10, help='示例数量')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    create_example_dataset(args.output_dir, args.num_examples) 