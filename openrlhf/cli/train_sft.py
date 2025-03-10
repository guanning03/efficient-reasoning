import argparse
import math
import os
from datetime import datetime
import multiprocessing

from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.models import Actor
from openrlhf.trainer import SFTTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

multiprocessing.set_start_method('spawn', force=True)

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # 在创建模型之前，确保使用正确的设备
    if args.flash_attn:
        import torch
        torch.set_default_device('cuda')

    # configure model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        packing_samples=args.packing_samples,
    )
    
    # 修改这行，添加默认值False
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=True)
    strategy.print(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
        num_processors=1,
    )
    eval_dataset = SFTDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
        num_processors=1,
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        shuffle=True,
        pin_memory=False,
        collate_fn=train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        shuffle=False,
        pin_memory=False,
        collate_fn=eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # prepare models
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint 相关参数 - 用于模型检查点的保存和加载
    parser.add_argument("--save_path", type=str, default="./ckpt")  # 模型保存路径
    parser.add_argument("--save_steps", type=int, default=-1)  # 每多少步保存一次模型，-1表示不保存
    parser.add_argument("--logging_steps", type=int, default=1)  # 每多少步记录一次日志
    parser.add_argument("--eval_steps", type=int, default=-1)  # 每多少步评估一次，-1表示不评估
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")  # 检查点加载路径
    parser.add_argument("--max_ckpt_num", type=int, default=3)  # 最多保存的检查点数量
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)  # 检查点最大内存限制
    parser.add_argument("--load_checkpoint", action="store_true", default=False)  # 是否加载检查点

    # DeepSpeed 相关参数 - 分布式训练和性能优化设置
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")  # 每个GPU的批次大小
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")  # 全局训练批次大小
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")  # 梯度裁剪阈值
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)  # 是否启用梯度检查点
    parser.add_argument("--seed", type=int, default=42)  # 随机种子
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")  # DeepSpeed本地进程排名
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")  # ZeRO优化阶段
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")  # 是否启用bfloat16精度
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")  # ZeRO++最大分区大小
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")  # 是否将Adam优化器卸载到CPU
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")  # 是否启用Flash Attention 2
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False,
                      help="Use reentrant version of gradient checkpointing")
    
    # SFT (Supervised Fine-Tuning) 相关参数 - 监督微调训练设置
    parser.add_argument("--max_epochs", type=int, default=2)  # 最大训练轮数
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")  # MoE平衡损失系数
    parser.add_argument("--pretrain", type=str, default=None)  # 预训练模型路径
    parser.add_argument("--learning_rate", type=float, default=5e-6)  # 学习率
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)  # 学习率预热比例
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")  # 学习率调度器类型

    # Ring-attention 相关参数 - 环形注意力机制设置
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")  # 环形注意力组大小
    parser.add_argument("--ring_head_stride", type=int, default=1)  # 每次执行环形注意力的头数

    # LoRA 相关参数 - 低秩适应训练设置
    parser.add_argument("--load_in_4bit", action="store_true", default=False)  # 是否使用4位量化加载
    parser.add_argument("--lora_rank", type=int, default=0)  # LoRA秩
    parser.add_argument("--lora_alpha", type=int, default=16)  # LoRA缩放因子
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")  # 目标LoRA模块
    parser.add_argument("--lora_dropout", type=float, default=0)  # LoRA dropout率

    # 数据集相关参数 - 训练数据配置
    parser.add_argument("--dataset", type=str, default=None)  # 数据集名称
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")  # 数据集采样概率
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")  # 输入模板格式
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")  # 最大样本数
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")  # 最大标记长度

    # Wandb 相关参数 - 实验追踪设置
    parser.add_argument("--use_wandb", type=str, default=None)  # 是否使用Wandb
    parser.add_argument("--wandb_org", type=str, default=None)  # Wandb组织名称
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_sft")  # Wandb项目名称
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # 添加新的参数
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False, 
                      help="Disable fast tokenizer")

    # 添加 packing_samples 参数
    parser.add_argument("--packing_samples", action="store_true", default=False,
                      help="Enable sample packing for more efficient training")

    # 优化器相关参数
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.999),
                      help="Adam optimizer betas (default: (0.9, 0.999))")
    parser.add_argument("--l2", type=float, default=0.0,
                      help="Weight decay coefficient (default: 0.0)")

    parser.add_argument('--pretrain_mode', type=bool, default=False,
                       help='Whether to run in pretrain mode')

    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    # TODO: [packing samples]
    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"
    
    train(args)
