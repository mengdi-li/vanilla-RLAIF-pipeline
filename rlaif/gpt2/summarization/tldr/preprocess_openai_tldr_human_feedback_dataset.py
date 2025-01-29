# obtain prompts from the training split of the tl;dr preference dataset: https://huggingface.co/datasets/openai/summarize_from_feedback
# the obtained prompts will be used to build our own preference dataset where the preference is labeled by an off-the-shelf LLM model


"""
python ./rlaif/gpt2/summarization/tldr/preprocess_openai_tldr_human_feedback_dataset.py \
    --dataset_name "openai/summarize_from_feedback" \
    --output_dir "./rlaif/gpt2/summarization/tldr/exps/openai_tldr_human_feedback_train_split_prompts_only" \
    --dataset_save_name "openai_tldr_human_feedback_train_split_prompts_only" \
    --save_json True \
"""

import argparse
from datasets import load_dataset, Dataset
import os
import pandas

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


if __name__ == "__main__":

    # params
    args = add_argument_group("Exp")
    args.add_argument(
        "--dataset_name", type=str, default="", help="dataset to load."
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
    ds = load_dataset(configs.dataset_name, "comparisons", split="train") # load the "comparisons" subset of the dataset

    # select the "info" column and remove other columns
    ds = ds.select_columns(["info"])
    ds_p = pandas.DataFrame(ds)
    ds_p = ds_p.drop_duplicates()
    ds = Dataset.from_pandas(ds_p)
    ds = ds.select_columns(["info"])

    # create output directory
    output_dir = utils.prepare_dirs(exp_dir=configs.output_dir, create_random_sub_dir=True)

    # save dataset
    save_ds(ds, output_dir, configs.dataset_save_name, configs.save_json)
