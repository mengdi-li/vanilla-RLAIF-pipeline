# sft gpt2-large on filtered TL;DR dataset as in paper: "Stiennon et al., learning to summarize from human feedback, 2020"
# dataset: https://huggingface.co/datasets/trl-internal-testing/tldr-preference-sft-trl-style
# train split: 117k, validation split: 6.45k, test split 6.55k

# based on code: trl/examples/scripts/sft.py
# docs: https://huggingface.co/docs/trl/en/sft_trainer

"""
# Full training
python ./rlaif/gpt2/summarization/tldr/sft.py \
    --model_name_or_path gpt2-large \
    --dataset_name "trl-internal-testing/tldr-preference-sft-trl-style" \
    --output_dir "./rlaif/gpt2/summarization/tldr/exps/sft/" \
    --learning_rate 5.0e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --seed 0 \
    --push_to_hub False \
    --log_level "info" \
    --logging_steps 50 \
    --report_to "wandb" 
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import wandb
import utils

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM
)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    utils.set_all_seeds(training_args.seed)
    training_args.output_dir = utils.prepare_dirs(exp_dir=training_args.output_dir, create_random_sub_dir=True)

    # logging
    if "wandb" in training_args.report_to:
        os.environ["WANDB_LOG_MODEL"] = "false"  # log all model checkpoints
        args_all = {**script_args.__dict__, **training_args.__dict__, **model_config.__dict__}
        run = wandb.init(
            project="rlaif",
            name=training_args.output_dir.split("/")[-1],
            tags=["gpt2","tldr","sft"],
            config=args_all,
        )

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch.float32, # check model card: https://huggingface.co/openai-community/gpt2-large
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=model_config.trust_remote_code, 
        use_fast=True, 
        padding_side = "right", # "right" in training, "left" in inference
    )


    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # add a pad token that is different from the eos token to make the model learn the prediction of the eos token. 
    model.resize_token_embeddings(len(tokenizer))

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name) # a dataset dict contains train, valiation, test splits

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = f"{example['prompt'][i]} {example['summary'][i]}{tokenizer.eos_token}" #manually add the eos token for gpt2 as its tokenizer does not automatilcally do it. 
            output_texts.append(text)
        return output_texts

    response_template = "TL;DR:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.evaluate()

    trainer.train()

    # Save and push to hub
    if training_args.load_best_model_at_end:
        trainer.save_model(training_args.output_dir) # save the best model if the argment "load_best_model_at_end" is set True



