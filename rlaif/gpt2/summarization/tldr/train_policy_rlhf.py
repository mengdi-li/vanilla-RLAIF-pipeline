# train a policy model based on the sft model.
# basic models needed: 
#   - sft model, which will work as 
#       - the initialization of the policy model
#       - the reference policy 
#   - reward model, which was trained based on the sft model using the ai (llama3.3-70b) labeled preference dataset 
# dataset:
#   - Used: prompts from the filtered 123k trdr datset, the one used in sft.
#   - Not used: prompts from the openai tldr preference dataset, the one used for generating our own preference dataset and reward model training. 
# training on 4 80G A100 will take about 6 hours
# based on example: https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo_tldr.py
# the corresponding example wandb log: https://wandb.ai/huggingface/trl/runs/dd2o3g35?nw=nwuserkrasul


""" with accelerate, using deepspeed zero-2 when prioritizing training speed over memory footprint
accelerate launch --config_file ./rlaif/gpt2/summarization/tldr/deepspeed_zero2_nprocesses4.yaml \
    ./rlaif/gpt2/summarization/tldr/train_policy.py \
    --sft_model_path "./rlaif/gpt2/summarization/tldr/exps/sft/2024-12-29 18:26:08.039195/checkpoint-1824" \
    --reward_model_path "./rlaif/gpt2/summarization/tldr/exps/rm_rlhf/2025-01-20 10:30:04.145924/best_model" \
    --dataset_name "trl-internal-testing/tldr-preference-sft-trl-style" \
    --response_length 128 \
    --learning_rate 1.00e-5 \
    --output_dir "./rlaif/gpt2/summarization/tldr/exps/policy" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 10 \
    --eval_accumulation_steps 1 \
    --num_train_epochs 1 \
    --logging_strategy "steps" \
    --logging_first_step True \
    --logging_steps 0.01 \
    --num_sample_generations 5 \
    --eval_strategy "steps" \
    --eval_steps 0.02 \
    --save_strategy "epoch" \
    --seed 1 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --use_peft False \
    --report_to "none" \
    --if_debug False \
"""

import os
import torch
import wandb
from dataclasses import dataclass
from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
)

from trl import (
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    PPOConfig,
    ModelConfig,
    PPOTrainer,
    ScriptArguments,
)

import utils

@dataclass
class CustomizedArguments:
    task_name: str = "summarization"
    if_debug: bool = False

def config_policy_model_generation(model, tokenizer, configs):
    # model generation config
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.max_new_tokens = configs.policy_max_new_tokens
    model.generation_config.do_sample = configs.policy_do_sample
    model.generation_config.temperature = configs.policy_temperature
    model.generation_config.top_p = configs.policy_top_p

if __name__ == "__main__":

    ## perse args
    parser = HfArgumentParser(
        (CustomizedArguments, ScriptArguments, PPOConfig, ModelConfig)
    )
    customized_args, script_args, training_args, model_config = (
        parser.parse_args_into_dataclasses()
    )
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ## set random seeds
    utils.set_all_seeds(training_args.seed)

    # customized args
    # generation config policy
    customized_args.policy_max_new_tokens = 128 
    customized_args.policy_do_sample = True
    customized_args.policy_temperature = 0.9
    customized_args.policy_top_p = 1.0  # 0.9

    ## prepare output dir
    training_args.output_dir = utils.prepare_dirs(exp_dir=training_args.output_dir, create_random_sub_dir=True)

    # logging
    if "wandb" in training_args.report_to:
        os.environ["WANDB_LOG_MODEL"] = "false"  # log all model checkpoints
        args_all = {**script_args.__dict__, **training_args.__dict__, **model_config.__dict__, **customized_args.__dict__}
        if "deepspeed_plugin" in args_all:
            args_all.pop("deepspeed_plugin")
        run = wandb.init(
            project="rlaif",
            name=training_args.output_dir.split("/")[-1],
            tags=["gpt2","tldr","ppo_rlhf"],
            config=args_all,
        )


    ################
    # Model & Tokenizer
    ################
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.sft_model_path,
        padding_side="left", # "right" in training, "left" in inference
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # policy model
    torch_dtype = torch.float32  # torch.bfloat16
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_config.trust_remote_code,
    )
    policy.config.pad_token_id = tokenizer.pad_token_id
    config_policy_model_generation(policy, tokenizer, customized_args)

    # reference policy
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_config.trust_remote_code,
        )
        ref_policy.config.pad_token_id = tokenizer.pad_token_id
        ref_policy.generation_config.pad_token_id = tokenizer.pad_token_id
    else:
        ref_policy = None

    # value model
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        torch_dtype=torch_dtype,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id

    # reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        torch_dtype=torch_dtype,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
    )
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name) # a dataset dict contains train, valiation, test splits
    train_dataset = dataset["train"]
    eval_dataset = (
        dataset["validation"]
        if training_args.eval_strategy != "no"
        else None
    )

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""
        def tokenize(element):
            return tokenizer(
                element["prompt"],
                padding=False,
            )

        ds = dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=8,
            batched=True,
        )
        return ds

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer) # the response label in the evaluation split is also discarded. we will only use the reward value on the validation split when selecting the checkpoint of the policy.

        # for debug
        if customized_args.if_debug:
            train_dataset = train_dataset.select(range(50))
            if eval_dataset is not None:
                eval_dataset = eval_dataset.select(range(50))

    assert (
        train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id
    ), "The last token should not be an EOS token"

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

    # trainer.generate_completions() # this will be done on the whole eval_dataset
    # trainer.generate_completions(sampling=True) # this will be done on the first batch of the eval_dataset
