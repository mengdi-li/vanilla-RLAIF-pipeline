# evaluate the prediction accuracy of a reward model on a human-labeled preference dataset
# we want to get similar results as reported in appendix G of paper "RLAIF vs RLHF: ..."
# we will use a downsampled split of the evaluation split of dataset: https://huggingface.co/datasets/openai/summarize_from_feedback
# we will only use samples of the tldr dataset and filter out samples from the cnn/dm dataset.
# we will only use samples with a high confidence score (select samples with a confidence score of 8 or 9 in a score range [1,9]), as in section 3 of paper "RLAIF vs RLHF"

""" 
python ./rlaif/gpt2/summarization/tldr/evaluate_rm_acc.py \
    --reward_model_name_or_path "./rlaif/gpt2/summarization/tldr/exps/rm/2025-01-14 21:30:18.186895/best_model" \
    --number_samples 2000 \
    --confidence_threshold 8 \
    --output_dir "./rlaif/gpt2/summarization/tldr/exps/evaluate_rm_acc" \
    --per_device_eval_batch_size 200 \
    --report_to "none" 
"""

from dataclasses import dataclass, field
from typing import Optional
import gc

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import RewardTrainer, RewardConfig

import utils 

@dataclass
class ScriptArguments:
    reward_model_name_or_path: str = field(metadata={"help": "The model name or path to the model to evaluate."})
    number_samples: int = field(default=3000, metadata={"help": "Number of samples to select for evaluation. Maximum number is 21488. "})
    confidence_threshold: int = field(default=8, metadata={"help": "Confidence threshold to filter out evaluation samples. Only samples with a confidence score >= this threshold will be used for evaluation. The labeled confidence score is in [1,9]"})

def preprocess_ds(ds, number_samples, confidence_threshold):
    # filter sources
    ds = ds.filter(lambda example: example["info"]["post"] != None) # 86086 -> 83802
    # filter confidence
    ds = ds.filter(lambda example: example["extra"]["confidence"] >= confidence_threshold) # -> 21448
    # downsample 
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(number_samples))
    return ds

def construct_sample(sample):
    choice_idx = sample["choice"]
    chosen_summary = sample["summaries"][choice_idx]["text"]
    rejected_summary = sample["summaries"][1-choice_idx]["text"]
    implicit_preference_sample = {
        "chosen": f"SUBREDDIT: r/{sample["info"]["subreddit"]}\n\nTITLE: {sample["info"]["title"]}\n\nPOST: {sample["info"]["post"]}\n\nTL;DR:{chosen_summary}<|endoftext|>", #manually add the eos token for gpt2 as its tokenizer does not automatilcally do it. 
        "rejected": f"SUBREDDIT: r/{sample["info"]["subreddit"]}\n\nTITLE: {sample["info"]["title"]}\n\nPOST: {sample["info"]["post"]}\n\nTL;DR:{rejected_summary}<|endoftext|>",
    }
    return implicit_preference_sample
 
if __name__ == "__main__":
    # Parse the arguments
    parser = HfArgumentParser((ScriptArguments, RewardConfig))
    script_args, reward_args = parser.parse_args_into_dataclasses()

    ## prepare output dir
    reward_args.output_dir = utils.prepare_dirs(exp_dir=reward_args.output_dir, create_random_sub_dir=True)

    # load dataset
    dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split="validation")

    # load model and tokenizer
    model_kwargs = dict(
        device_map=None,
        torch_dtype=torch.float32,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.reward_model_name_or_path,
        # num_labels=1,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.reward_model_name_or_path,
        use_fast=True,
    )
    # preprocess dataset
    ds = preprocess_ds(dataset, script_args.number_samples, script_args.confidence_threshold)

    # convert preference dataset to implicit preference dataset. refer to code of "preprocess_preference_dataset_for_rm_training.py"
    columns_to_remove = [x for x in ds.features if x not in ["chosen", "rejected"]]
    ds = ds.map(construct_sample, remove_columns=columns_to_remove)
    dummy_ds = ds.select(range(5))

    # initualize a reward trainer. we use it only for evaluation 
    trainer = RewardTrainer( 
        model=model,
        processing_class=tokenizer,
        args=reward_args,
        train_dataset=dummy_ds,
        eval_dataset=ds,
    )

    metrics = trainer.evaluate()
    print(metrics)
