# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trainer for FOODPO training."""


import argparse
import os
import sys
import json
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text.preference import PreferenceBatch, PreferenceDataset
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.device_utils import torch_gc, torch_set_device
from align_anything.utils.multi_process import (
    get_all_reduce_mean,
    get_current_device,
    is_main_process,
)
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    gather_log_probabilities,
    prepare_ds_eval_cfgs,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    update_dict,
)


def strip_pad(seq: torch.Tensor, pad_token_id: int):
    # remove the pad token in the tensor
    return seq[seq != pad_token_id]


class FOODPOTrainer(SupervisedTrainerBase):

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize trainer."""
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.ds_eval_cfgs = prepare_ds_eval_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0
        self.infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}

        self.init_check()
        dist.barrier()
        self.init_models()
        if hasattr(self.model, 'infer_batch'):
            self.infer_batch = self.model.infer_batch
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

    def init_check(self) -> None:
        """Initial configuration checking."""
        super().init_check()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_cfgs)
        if self.ds_eval_cfgs['zero_optimization']['stage'] == 3:
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_cfgs)
        self.bnb_cfgs = self.cfgs.bnb_cfgs
        self.lora_cfgs = self.cfgs.lora_cfgs
        self.model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=True,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
        )
        self.reference_model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            PreferenceDataset, PreferenceDataset
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines()
        self.reference_model, *_ = deepspeed.initialize(
            model=self.reference_model,
            config=self.ds_eval_cfgs,
        )

    def compute_log_probs(
        self,
        model: AutoModelForCausalLM,
        batch: PreferenceBatch,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(**self.infer_batch(batch)).logits
        device = logits.device
        input_ids = batch['input_ids']
        batch_size = len(batch['meta_info']['response_lens'])
        logprob_list = []
        for idx in range(batch_size):
            response_length = batch['meta_info']['response_lens'][idx]
            raw_input_id = strip_pad(input_ids[idx], self.tokenizer.pad_token_id)
            logit = logits[idx][-response_length:].unsqueeze(0)
            input_id = raw_input_id[-response_length:].unsqueeze(0)
            log_p = gather_log_probabilities(logit[:, :-1], input_id[:, 1:])
            logprob_list.append(log_p.squeeze(0))
        return torch.nn.utils.rnn.pad_sequence(
            logprob_list, batch_first=True, padding_value=0.0
        ).to(device)

    def loss(  # pylint: disable=too-many-locals
        self,
        batch: PreferenceBatch,
    ) -> dict[str, torch.Tensor]:
        """Loss function for the FOODPO algorithm."""
        sequence_log_probs = self.compute_log_probs(
            self.model.module,
            batch,
        )
        (
            better_sequence_log_probs,  # size = (B, L - 1)
            worse_sequence_log_probs,  # size = (B, L - 1)
        ) = sequence_log_probs.chunk(chunks=2, dim=0)

        with torch.no_grad():
            ref_sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                self.reference_model.module,
                batch,
            )
            ref_better_sequence_log_probs, ref_worse_sequence_log_probs = (
                ref_sequence_log_probs.chunk(chunks=2, dim=0)
            )

        losses = []
        better_sample_rewards = []
        worse_sample_rewards = []

        batch_size = better_sequence_log_probs.size(0)
        for i in range(batch_size):
            better_log_prob = better_sequence_log_probs[i, :].sum(dim=-1)
            worse_log_prob = worse_sequence_log_probs[i, :].sum(dim=-1)
            ref_better_log_prob = ref_better_sequence_log_probs[i, :].sum(dim=-1)
            ref_worse_log_prob = ref_worse_sequence_log_probs[i, :].sum(dim=-1)
            better_log_ratio = better_log_prob - ref_better_log_prob
            worse_log_ratio = worse_log_prob - ref_worse_log_prob

            losses.append(
                -F.logsigmoid(
                    self.cfgs.train_cfgs.scale_coeff * (better_log_ratio - worse_log_ratio),
                ),
            )
            better_sample_rewards.append(
                self.cfgs.train_cfgs.scale_coeff * better_log_ratio.detach(),
            )
            worse_sample_rewards.append(self.cfgs.train_cfgs.scale_coeff * worse_log_ratio.detach())
        loss = torch.stack(losses).mean()  # size = ()
        better_sample_reward = torch.stack(better_sample_rewards)
        worse_sample_reward = torch.stack(worse_sample_rewards)  # size = (B,)
        reward = better_sample_reward + worse_sample_reward  # size = (B,)
        reward_accuracy = (better_sample_reward > worse_sample_reward).float().mean()  # size = ()
        reward_margin = better_sample_reward - worse_sample_reward  # size = (B,)

        return {
            'loss': loss,
            'reward': reward,
            'better_sample_reward': better_sample_reward,
            'worse_sample_reward': worse_sample_reward,
            'reward_accuracy': reward_accuracy,
            'reward_margin': reward_margin,
        }

    def train_step(
        self,
        batch: PreferenceBatch,
    ) -> dict[str, Any]:
        """Perform a single training step for FOODPO."""
        loss_dict = self.loss(batch=batch)
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        with torch.no_grad():
            reward = loss_dict['reward'].mean()
            better_sample_reward = loss_dict['better_sample_reward'].mean()
            worse_sample_reward = loss_dict['worse_sample_reward'].mean()
            reward_accuracy = loss_dict['reward_accuracy']
            reward_margin = loss_dict['reward_margin'].mean()

            loss = get_all_reduce_mean(loss)
            reward = get_all_reduce_mean(reward)
            better_sample_reward = get_all_reduce_mean(better_sample_reward)
            worse_sample_reward = get_all_reduce_mean(worse_sample_reward)
            reward_accuracy = get_all_reduce_mean(reward_accuracy)
            reward_margin = get_all_reduce_mean(reward_margin)

        return {
            'train/loss': loss.item(),
            'train/reward': reward.item(),
            'train/better_sample_reward': better_sample_reward.item(),
            'train/worse_sample_reward': worse_sample_reward.item(),
            'train/reward_accuracy': reward_accuracy.item(),
            'train/reward_margin': reward_margin.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.cfgs.train_cfgs.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.cfgs.train_cfgs.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        for epoch in range(self.cfgs.train_cfgs.epochs):
            self.model.train()
            self.train_dataloader.sampler.set_epoch(epoch)

            for batch in self.train_dataloader:
                metrics = self.train_step(batch=batch)
                self.global_step += 1
                
                progress_bar.set_description(f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch')
                progress_bar.update(1)
                self.logger.log(metrics, step=self.global_step)

                if (
                    is_main_process()
                    and self.global_step % self.cfgs.logger_cfgs.save_interval == 0
                ):
                    self.save(tag=self.global_step)

            if self.cfgs.data_cfgs.eval_datasets and self.cfgs.train_cfgs.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.cfgs.train_cfgs.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)
            self.model.tput_timer.update_epoch_count()

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        return {}

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        self.save_transformers(model=model, tag=tag)


def main():
    """命令行入口函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name_or_path', type=str, required=True, help='模型路径或名称')
    parser.add_argument('--train_data_files', type=str, default=None, help='训练数据文件路径')
    parser.add_argument('--train_datasets', type=str, default=None, help='训练数据集名称')
    parser.add_argument('--train_template', type=str, default=None, help='训练数据模板')
    parser.add_argument('--train_split', type=str, default=None, help='训练数据分割')
    parser.add_argument('--train_size', type=int, default=None, help='训练数据大小')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='每个设备的训练批次大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='学习率')
    parser.add_argument('--save_interval', type=int, default=100000, help='保存模型的间隔步数')
    
    args, unparsed_args = parser.parse_known_args()

    # 读取默认配置
    try:
        task = os.path.join('text_to_text', 'foodpo')
        dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)
    except FileNotFoundError:
        # 如果foodpo配置不存在，使用dpo配置
        print("未找到foodpo配置，将使用dpo配置...")
        task = os.path.join('text_to_text', 'dpo')
        dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)
    
    # 更新命令行参数到配置
    if args.model_name_or_path:
        dict_cfgs = update_dict(dict_cfgs, {'model_cfgs': {'model_name_or_path': args.model_name_or_path}})
    if args.train_datasets:
        dict_cfgs = update_dict(dict_cfgs, {'data_cfgs': {'train_datasets': args.train_datasets}})
    if args.train_template:
        dict_cfgs = update_dict(dict_cfgs, {'data_cfgs': {'train_template': args.train_template}})
    if args.train_split:
        dict_cfgs = update_dict(dict_cfgs, {'data_cfgs': {'train_split': args.train_split}})
    if args.train_size:
        dict_cfgs = update_dict(dict_cfgs, {'data_cfgs': {'train_size': args.train_size}})
    if args.train_data_files:
        dict_cfgs = update_dict(dict_cfgs, {'data_cfgs': {'train_data_files': args.train_data_files}})
    if args.output_dir:
        dict_cfgs = update_dict(dict_cfgs, {'logger_cfgs': {'output_dir': args.output_dir}})
    if args.epochs:
        dict_cfgs = update_dict(dict_cfgs, {'train_cfgs': {'epochs': args.epochs}})
    if args.per_device_train_batch_size:
        dict_cfgs = update_dict(dict_cfgs, {'train_cfgs': {'per_device_train_batch_size': args.per_device_train_batch_size}})
    if args.gradient_accumulation_steps:
        dict_cfgs = update_dict(dict_cfgs, {'train_cfgs': {'gradient_accumulation_steps': args.gradient_accumulation_steps}})
    if args.learning_rate:
        dict_cfgs = update_dict(dict_cfgs, {'train_cfgs': {'learning_rate': args.learning_rate}})
    if args.save_interval:
        dict_cfgs = update_dict(dict_cfgs, {'logger_cfgs': {'save_interval': args.save_interval}})
        
    # 处理其他命令行参数
    keys = [k[2:] for k in unparsed_args[::2]]
    values = unparsed_args[1::2]
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # 设置分布式训练
    try:
        deepspeed.init_distributed()
    except Exception as e:
        print(f"警告: 分布式初始化失败: {e}")
        print("将使用单设备训练...")
    
    current_device = get_current_device()
    torch_set_device(current_device)

    # 设置训练
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    print("配置加载完成，开始初始化FOODPO训练器...")
    
    # 检查本地数据文件
    if cfgs.data_cfgs.train_data_files:
        print(f"使用本地数据文件: {cfgs.data_cfgs.train_data_files}")
        if not os.path.exists(cfgs.data_cfgs.train_data_files):
            raise FileNotFoundError(f"找不到训练数据文件: {cfgs.data_cfgs.train_data_files}")
    
    # 检查输出目录
    os.makedirs(cfgs.logger_cfgs.output_dir, exist_ok=True)
    
    # 训练模型
    trainer = FOODPOTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()

if __name__ == "__main__":
    main() 