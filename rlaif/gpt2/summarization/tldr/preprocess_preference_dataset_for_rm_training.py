# preprocess preference dataset:
# 1. convert it into the implicit preference format that is required by TRL for reward model training.
# 2. add a column of "soft_label", which indicates the probability of the chosen response. we apply "clamping" on the probability with a threshold of 0.6, as suggested in paper "constitutional ai" when using cot reasoning to generate preference labels.
# tutoral: https://huggingface.co/docs/trl/main/en/dataset_formats#preference

"""
python ./rlaif/gpt2/summarization/tldr/preprocess_preference_dataset_for_rm_training.py \
    --dataset_path_or_name "./rlaif/gpt2/summarization/tldr/exps/ai_feedback_preference_dataset/full-train-split-14776samples/" \
    --output_dir "./rlaif/gpt2/summarization/tldr/exps/ai_feedback_preference_dataset/full-train-split-14776samples" \
    --dataset_save_name "preference_ds_two_orders_implicit" \
    --save_json True \
"""

import argparse
from datasets import load_from_disk
import os
import pathlib
import numpy as np

import utils

arg_lists = []
parser = argparse.ArgumentParser(description="Exp")

def str2bool(v):
    return v.lower() in ("true", "1")

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def get_config():
    configs, unparsed = parser.parse_known_args()
    return configs, unparsed

def save_ds(ds, exp_dir, ds_save_name, save_json):
    ds_savepath = os.path.join(exp_dir, ds_save_name)
    ds.save_to_disk(dataset_path=ds_savepath)
    if save_json:
        ds.to_json(os.path.join(ds_savepath, f"{ds_save_name}.json"))
    else:
        pass
    return ds_savepath

def calc_soft_label(sample):
    # calculate average probability
    logp_original_order = sample["preference_log_prob_cot_0_shot"]
    logp_reversed_order = sample["preference_log_prob_cot_0_shot_reversed"]
    p_a = np.exp(logp_original_order)
    p_b = np.flip(np.exp(logp_reversed_order))
    p_average = (p_a + p_b) / 2
    # clamping probability
    p_clamp = np.clip(p_average, 0.4, 0.6).tolist()
    return p_clamp


def construct_sample(sample):
    soft_label = calc_soft_label(sample)
    chosen_index = np.argmax(soft_label).item()
    soft_label_chosen = np.max(soft_label).item() # only return soft label of the chosen response

    if chosen_index == 0:
        chosen_summary = sample["summary_a"]
        rejected_summary = sample["summary_b"]
    else:
        chosen_summary = sample["summary_b"]
        rejected_summary = sample["summary_a"]

    implicit_preference_sample = {
        # "chosen": f"SUBREDDIT: r/{sample["subreddit"]}\n\nTITLE: {sample["title"]}\n\nPOST: {sample["post"]}\n\nTL;DR:{chosen_summary}<|endoftext|>", #manually add the eos token for gpt2 as its tokenizer does not automatilcally do it. 
        # "rejected": f"SUBREDDIT: r/{sample["subreddit"]}\n\nTITLE: {sample["title"]}\n\nPOST: {sample["post"]}\n\nTL;DR:{rejected_summary}<|endoftext|>",
        "chosen": f"{sample["prompt"]}{chosen_summary}<|endoftext|>", #manually add the eos token for gpt2 as its tokenizer does not automatilcally do it. 
        "rejected": f"{sample["prompt"]}{rejected_summary}<|endoftext|>",
        "soft_label_chosen": soft_label_chosen
    }
    return implicit_preference_sample

def load_and_merge_dss(dataset_path_or_name):
    ds_original_order = load_from_disk(os.path.join(dataset_path_or_name, "preference_ds_only_original_order"))
    ds_reversed_order = load_from_disk(os.path.join(dataset_path_or_name, "preference_ds_only_reversed_order"))
    # ds_reversed_order = ds_reversed_order.select_columns(["preference_log_prob_cot_0_shot_reversed"])
    ds = ds_original_order.add_column("preference_log_prob_cot_0_shot_reversed", ds_reversed_order["preference_log_prob_cot_0_shot_reversed"])
    return ds

if __name__ == "__main__":
    # params
    args = add_argument_group("Exp")
    args.add_argument(
        "--dataset_path_or_name", type=str, default="", help="dataset to load."
    )
    args.add_argument(
        "--output_dir", type=str, default="", help="directory to save inference results."
    )
    args.add_argument(
        "--dataset_save_name", type=str, default="", help="save name of the generated dataset."
    )
    args.add_argument(
        "--save_json",
        type=str2bool,
        default=True,
        help="whether to save the json file of the generated dataset.",
    )
    configs, unparsed = get_config()

    # load dataset
    ds = load_and_merge_dss(configs.dataset_path_or_name)

    ds = ds.map(construct_sample, remove_columns=ds.features)

    # save dataset
    ds_savepath = save_ds(ds, configs.output_dir, configs.dataset_save_name, configs.save_json)
    print(f"an implicit preference dataset was saved to {ds_savepath}")
