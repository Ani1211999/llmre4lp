# Usage: deepspeed train_lora_arxiv.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adapted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os
import sys
import wandb
import json
try:
    import deepspeed
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
except ImportError as e:
    print(f"Error: DeepSpeed not found. Install with 'pip install deepspeed==0.16.6'")
    sys.exit(1)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig
import torch

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    make_supervised_data_module,
)
from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = field(default=False)
    report_to: typing.Optional[typing.List[str]] = field(
        default_factory=lambda: ["wandb"],  # <-- Enable wandb logging
        metadata={"help": "List of integrations to report the results and logs to."},
    )


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = field(default=False)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def is_zero3_enabled(deepspeed_config_path: str):
    """
    Detects if ZeRO-3 is enabled by checking the DeepSpeed config file.
    Returns True if stage == 3.
    """
    import json
    try:
        with open(deepspeed_config_path, "r") as f:
            config = json.load(f)
        zero_stage = config.get("zero_optimization", {}).get("stage", 0)
        return zero_stage == 3
    except Exception as e:
        print(f"[WARNING] Could not parse DeepSpeed config: {e}")
        return False


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Early detection of ZeRO-3 mode
    zero3_enabled = False
    if training_args.deepspeed:
        zero3_enabled = is_zero3_enabled(training_args.deepspeed)
        print("[INFO] ZeRO-3 Enabled:", "✅" if zero3_enabled else "❌")

    # Fail early if QLoRA + ZeRO-3 conflict
    if zero3_enabled and lora_args.q_lora:
        raise ValueError("QLoRA and ZeRO-3 are incompatible. Disable one.")

    # Flash Attention patch
    if training_args.flash_attn:
        replace_llama_attn_with_flash_attn()

    # Device map logic
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None

    # Compute dtype
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load base model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        ) if lora_args.q_lora else None,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    # Prepare model for kbit training
    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

    # Wrap with PEFT
    model = get_peft_model(model, lora_config)

    # Cast some modules to lower precision if needed
    if training_args.flash_attn:
        for name, module in model.named_modules():
            if "norm" in name:
                module.to(compute_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module.to(compute_dtype)

    # Print trainable parameters if local_rank == 0
    if training_args.deepspeed and training_args.local_rank == 0:
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        print("✅ Gradient checkpointing enabled")

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    if training_args.report_to and "wandb" in training_args.report_to:
        def safe_dict(d):
            safe = {}
            for k, v in d.items():
                try:
                    json.dumps(v)  # test if serializable
                    safe[k] = v
                except (TypeError, OverflowError):
                    safe[k] = str(v)  # fallback: store as string
            return safe

        # Initialize wandb with safe config
        wandb.init(
            project="lora-finetune-arxiv",
            name=f"{model_args.model_name_or_path}-finetune",
            config={
                **safe_dict(vars(training_args)),
                **safe_dict(vars(model_args)),
                **safe_dict(vars(data_args)),
                **safe_dict(vars(lora_args)),
            },
        )

    # Trainer
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # Start training
    model.config.use_cache = False
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # Save final model
    if zero3_enabled:
        if hasattr(trainer.model_wrapped, '_zero3_consolidated_16bit_state_dict'):
            state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        else:
            state_dict = None
    else:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    if training_args.report_to and "wandb" in training_args.report_to:
        wandb.finish()

if __name__ == "__main__":
    train()