# generate responses on the validation split. 
# the generated responses will be used to compute win-rate against other policies then. 
"""
python ./rlaif/gpt2/summarization/tldr/inference_validation.py \
    --model_path "./rlaif/gpt2/summarization/tldr/exps/sft/2024-12-29 18:26:08.039195/checkpoint-1824" \
    --dataset_name "trl-internal-testing/tldr-preference-sft-trl-style" \
    --output_dir "./rlaif/gpt2/summarization/tldr/exps/inference_validation" \
    --dataset_save_name "sft-policy" \
    --save_json True \
    --batch_size 64 \
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import utils

arg_lists = []
parser = argparse.ArgumentParser(description="Generate on Valiation")

def str2bool(v):
    return v.lower() in ("true", "1")

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def get_config():
    configs, unparsed = parser.parse_known_args()
    return configs, unparsed

# data preprocess
def preprocess_summary(example):

    return tokenizer(
        example["prompt"],
        padding=False,
        # max_length=tokenizer.model_max_length,  # default: tokenizer.model_max_length, which is 1024 for gpt-2
    )

def save_ds(ds, exp_dir, ds_save_name, save_json):
    ds_savepath = os.path.join(exp_dir, ds_save_name)
    ds.save_to_disk(dataset_path=ds_savepath)
    if save_json:
        ds.to_json(os.path.join(ds_savepath, f"{ds_save_name}.json"))
    else:
        pass

if __name__ == "__main__":

    # params
    args = add_argument_group("Inference Params")
    args.add_argument(
        "--model_path", type=str, default="", help="model to load."
    )
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
        "--batch_size", type=int, default=32, help="batch size for inference."
    )
    args.add_argument(
        "--save_json",
        type=str2bool,
        default=True,
        help="whether to save the json file of the generated dataset.",
    )
    configs, unparsed = get_config()

    # load model and tokenizer
    model_kwargs = dict(
        torch_dtype=torch.float32, # check model card: https://huggingface.co/openai-community/gpt2-large
        use_cache=False, 
        device_map=None,
    )
    model = AutoModelForCausalLM.from_pretrained(configs.model_path, **model_kwargs)
    model.eval()
    model.to("cuda")
    
    # load tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_path, 
        use_fast=True, 
        padding_side = "left", # "right" in training, "left" in inference
    )

    # config model generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.max_new_tokens = 500
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    
    # load dataset
    ds = load_dataset(configs.dataset_name, split="validation") # load the validation split of the dataset
    # ds = ds.select(range(10)) #for debug
    ds_processed = ds.map(preprocess_summary, batched=True, batch_size=1000, remove_columns=ds.column_names)
    data_collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
    dataloader = DataLoader(
        ds_processed,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=configs.batch_size,
        pin_memory=True,
    )

    # create output directory
    output_dir = utils.prepare_dirs(exp_dir=configs.output_dir, create_random_sub_dir=True)

    # inference
    summary_sft = []
    for step, batch in enumerate(tqdm(dataloader)):
        inputs = {k: v.to("cuda") for k, v in batch.items()}

        # generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        input_length = inputs["input_ids"].shape[-1]
        responses = outputs[:,input_length:]

        # decode into text
        responses_text = tokenizer.batch_decode(
            responses.detach().cpu().numpy(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        summary_sft.extend(responses_text)

    ds = ds.add_column(name="summary_sft", column=summary_sft)
    save_ds(ds, output_dir, configs.dataset_save_name, configs.save_json)
