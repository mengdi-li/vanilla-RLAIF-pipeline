# train a reward model based on sft model using the ai (llama3.3-70b) labeled preference dataset 
# code is based on example: https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py

""" only train score weight 
python ./rlaif/gpt2/summarization/tldr/train_rm_rlhf.py \
--model_name_or_path "./rlaif/gpt2/summarization/tldr/exps/sft/2024-12-29 18:26:08.039195/checkpoint-1824" \
--dataset_name "openai/summarize_from_feedback" \
--output_dir "./rlaif/gpt2/summarization/tldr/exps/rm_rlhf" \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 10 \
--num_train_epochs 1 \
--gradient_checkpointing True \
--learning_rate 3.16e-5 \
--logging_steps 50 \
--eval_strategy steps \
--eval_steps 50 \
--save_steps 200 \
--load_best_model_at_end True \
--metric_for_best_model "accuracy" \
--max_length 1024 \
--use_peft False \
--use_soft_label False \
--report_to "wandb" 
"""

import warnings
import torch
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

from datasets import load_dataset
import os
import sys
import pathlib
from dataclasses import dataclass
import wandb
import numpy as np

import utils 

def construct_sample(sample):
    choice_idx = sample["choice"]
    chosen_summary = sample["summaries"][choice_idx]["text"]
    rejected_summary = sample["summaries"][1-choice_idx]["text"]
    implicit_preference_sample = {
        "chosen": f"SUBREDDIT: r/{sample["info"]["subreddit"]}\n\nTITLE: {sample["info"]["title"]}\n\nPOST: {sample["info"]["post"]}\n\nTL;DR:{chosen_summary}<|endoftext|>", #manually add the eos token for gpt2 as its tokenizer does not automatilcally do it. 
        "rejected": f"SUBREDDIT: r/{sample["info"]["subreddit"]}\n\nTITLE: {sample["info"]["title"]}\n\nPOST: {sample["info"]["post"]}\n\nTL;DR:{rejected_summary}<|endoftext|>",
    }
    return implicit_preference_sample

@dataclass
class CustomizedArguments:
    task_name: str = "summarization"
    use_soft_label: bool = True

if __name__ == "__main__":

    ## set random seeds
    utils.set_all_seeds(0)
    
    ## perse args
    parser = HfArgumentParser(
        (CustomizedArguments, ScriptArguments, RewardConfig, ModelConfig)
    )
    customized_args, script_args, training_args, model_config = (
        parser.parse_args_into_dataclasses()
    )
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ## prepare output dir
    training_args.output_dir = utils.prepare_dirs(exp_dir=training_args.output_dir, create_random_sub_dir=True)

    # logging
    if "wandb" in training_args.report_to:
        os.environ["WANDB_LOG_MODEL"] = "false"  # log all model checkpoints
        args_all = {**script_args.__dict__, **training_args.__dict__, **model_config.__dict__, **customized_args.__dict__}
        run = wandb.init(
            project="rlaif",
            name=training_args.output_dir.split("/")[-1],
            tags=["gpt2","tldr","rm_rlhf"],
            config=args_all,
        )

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs
    )
    
    # Align padding tokens between tokenizer and model
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    ##############
    # Load dataset
    ##############
    dataset = load_dataset(script_args.dataset_name, "comparisons", split="train")
    dataset = dataset.map(construct_sample, remove_columns=dataset.features)

    dataset = dataset.train_test_split(test_size=0.05, shuffle=False)

    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        peft_config=get_peft_config(model_config),
    )

    # print params info  
    if model_config.use_peft:
        trainer.model.print_trainable_parameters()
    else:
        num_trainable_params = trainer.get_num_trainable_parameters()
        print(f"number of trainable parameters: {num_trainable_params}\n")

    metrics = trainer.train()
    

    #############
    # Save model
    #############
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save
    if training_args.load_best_model_at_end:
        trainer.save_model(os.path.join(training_args.output_dir, "best_model")) # save the best model if the argment "load_best_model_at_end" is set True
