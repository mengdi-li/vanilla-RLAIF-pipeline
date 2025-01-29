# reproduce the method used in paper: "rlaif: scaling reinforcement learning from human feedback with ai feedback"
# build a preference dataset for a sft model.
# the preference dataset is built using prompts for the train split of the openai trdl human feedback dataset (https://huggingface.co/datasets/openai/summarize_from_feedback). we filtered out duplicates prompts with script: "./rlaif/gpt2/summarization/tldr/preprocess_openai_tldr_human_feedback_dataset.py"
# preference is labeled using an off-the-shelf llm, e.g., llama-3.2-70b

""" generate summary dataset
python ./rlaif/gpt2/summarization/tldr/build_ai_feedback_preference_dataset_vllm.py \
    --build_mode_index 0 \
    --policy_model_name_or_path "./rlaif/gpt2/summarization/tldr/exps/sft/2024-12-29 18:26:08.039195/checkpoint-1824" \
    --dataset_name "./rlaif/gpt2/summarization/tldr/exps/openai_tldr_human_feedback_train_split_prompts_only" \
    --output_dir "./rlaif/gpt2/summarization/tldr/exps/ai_feedback_preference_dataset" \
    --save_json True \
    --batch_size 128 \
"""

""" generate preference dataset using an existing summary dataset
python ./rlaif/gpt2/summarization/tldr/build_ai_feedback_preference_dataset_vllm.py \
    --build_mode_index 1 \
    --labeling_model_name_or_path "meta-llama/Llama-3.3-70B-Instruct" \
    --preference_ds_order "both" \
    --tensor_parallel_size 4 \
    --target_dir "./rlaif/gpt2/summarization/tldr/exps/ai_feedback_preference_dataset/full-train-split-14776samples/" \
    --save_json True \
    --debug False 
""" 
# --tensor_parallel_size 1 \
# --labeling_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \

import argparse
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # default_data_collator,
    DataCollatorWithPadding,
    GenerationConfig,
)
from datasets import load_dataset, load_from_disk, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import os
import random
import pathlib
import numpy as np
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

import utils

# configs
arg_lists = []
parser = argparse.ArgumentParser(description="Params Parser")

def str2bool(v):
    return v.lower() in ("true", "1")

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def get_config():
    configs, unparsed = parser.parse_known_args()
    return configs, unparsed

def save_ds(ds, exp_dir, ds_savename, save_json):
    ds_savepath = os.path.join(exp_dir, ds_savename)
    ds.save_to_disk(dataset_path=ds_savepath)
    if save_json:
        ds.to_json(os.path.join(ds_savepath, f"{ds_savename}.json"))
    else:
        pass
    return ds_savepath

def config_policy_model_generation(model, tokenizer, configs):
    # model generation config
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.max_new_tokens = configs.policy_max_new_tokens
    model.generation_config.do_sample = configs.policy_do_sample
    model.generation_config.temperature = configs.policy_temperature
    model.generation_config.top_p = configs.policy_top_p

def build_summary_ds(model, tokenizer, ds, configs):
    if configs.debug:
        ds = ds.select(range(configs.debug_num_samples))

    # add the "prompt" column as in the the column in dataset https://huggingface.co/datasets/trl-internal-testing/tldr-preference-sft-trl-style
    prompt = [f"SUBREDDIT: r/{item["subreddit"]}\n\nTITLE: {item["title"]}\n\nPOST: {item["post"]}\n\nTL;DR:" for item in ds["info"]]
    ds = ds.add_column(name="prompt", column=prompt)
    
    # data preprocess
    def preprocess_summary(example):
        return tokenizer(
            example["prompt"],
            padding=False,
            # max_length=tokenizer.model_max_length,  # default: tokenizer.model_max_length, which is 1024 for gpt-2
        )

    ds_processed = ds.map(preprocess_summary, batched=True, batch_size=1000, remove_columns=ds.column_names)
    data_collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
    dataloader = DataLoader(
        ds_processed,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=configs.batch_size,
        pin_memory=True,
    )

    summary_a = []
    summary_b = []
    # model generate
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = {k: v.to("cuda") for k, v in batch.items()}
            input_length = inputs["input_ids"].shape[-1]

            # generate response a
            outputs_a = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            responses_a = outputs_a[:,input_length:]
            responses_a_text = tokenizer.batch_decode(
                responses_a.detach().cpu().numpy(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            summary_a.extend(responses_a_text)

            # generate response b
            outputs_b = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            responses_b = outputs_b[:,input_length:]
            responses_b_text = tokenizer.batch_decode(
                responses_b.detach().cpu().numpy(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            summary_b.extend(responses_b_text)

    ds = ds.add_column(name="summary_a", column=summary_a)
    ds = ds.add_column(name="summary_b", column=summary_b)
    return ds


def build_preference_ds_detailed_cot_0_shot(model, ds, configs, reverse_summary_order=False):
    if reverse_summary_order:
        print("start building the preference dataset in an reversed summary order. ")
    else:
        print("start building the preference dataset in an original summary order. ")
        
    ### cot labeling stage 1: generate rationale
    # config model and model generation
    # data preprocess
    def preprocess_prompt_detailed_cot_0_shot_rationale(example):
        def to_prompt_detailed_cot_0_shot_rationale(ds_prompt, summary_a, summary_b): # see table 2, and appendix N in paper "rlaif: scaling reinforcement learning from human feedback with ai feedback" for more information
            ds_prompt_clean = ds_prompt.split("TL;DR:")[0].strip() # remove the appended suffix.
            prompt = [
                {
                    "role": "user",
                    "content": f"A good summary is a shorter piece of text that has the essence of the original. It tries to accomplish the same purpose and conveys the key information from the original post. Below we define four evaluation axes for summary quality: coherence, accuracy, coverage, and overall quality. \nCoherence: This axis answers the question “how coherent is the summary on its own?” A summary is coherent if it’s easy to understand when read on its own and free of English errors. A summary is not coherent if it’s difficult to understand what the summary is trying to say. Generally, it’s more important that the summary is understandable than it being free of grammar errors. \nAccuracy: This axis answers the question “does the factual information in the summary accurately match the post?” A summary is accurate if it doesn’t say things that aren’t in the article, it doesn’t mix up people, and generally is not misleading. \nCoverage: This axis answers the question “how well does the summary cover the important information in the post?” A summary has good coverage if it mentions the main information from the post that’s important to understand the situation described in the post. A summary has poor coverage if someone reading only the summary would be missing several important pieces of information about the situation in the post. A summary with good coverage should also match the purpose of the original post (e.g. to ask for advice). \nOverall quality: This axis answers the question “how good is the summary overall at representing the post?” This can encompass all of the above axes of quality, as well as others you feel are important. If it’s hard to find ways to make the summary better, the overall quality is good. If there are lots of different ways the summary can be made better, the overall quality is bad. \nYou are an expert summary rater. Given a piece of text and two of its possible summaries, explain which summary best adheres to coherence, accuracy, coverage, and overall quality as defined above. \nText - {ds_prompt_clean} \nSummary 1 - {summary_a} \nSummary 2 - {summary_b} \nConsider the coherence, accuracy, coverage, and overall quality of each summary and explain which one is better."
                },
                {"role": "assistant", "content": "Rationale:"}, 
            ]

            return prompt

        if not reverse_summary_order:
            labeling_prompt = [
                to_prompt_detailed_cot_0_shot_rationale(ds_prompt, s_a, s_b)
                for (ds_prompt, s_a, s_b) in zip(
                    example["prompt"],
                    example["summary_a"],
                    example["summary_b"],
                )
            ]
        else:
            labeling_prompt = [
                to_prompt_detailed_cot_0_shot_rationale(ds_prompt, s_a, s_b)
                for (ds_prompt, s_a, s_b) in zip(
                    example["prompt"],
                    example["summary_b"],
                    example["summary_a"],
                )
            ]

        return {"prompt_rationale": labeling_prompt}
  
    assert torch.cuda.device_count() == configs.tensor_parallel_size
    
    ds_processed = ds.map(preprocess_prompt_detailed_cot_0_shot_rationale, batched=True, batch_size=1000, remove_columns=ds.column_names)

    print("start cot stage 1: generating rationales.")
    sampling_params = SamplingParams(temperature=configs.labeling_temperature, top_p=configs.labeling_top_p, max_tokens=configs.labeling_rationale_max_new_tokens)  # very generous max token length
    outputs = model.chat(ds_processed["prompt_rationale"], sampling_params, add_generation_prompt=False, continue_final_message=True)
    rationale = [output.outputs[0].text.strip() for output in outputs]

    if not reverse_summary_order:
        ds = ds.add_column(name="rationale_cot_0_shot", column=rationale)
    else:
        ds = ds.add_column(name="rationale_cot_0_shot_reversed", column=rationale)

    ### cot labeling stage 2: generate preference

    tokenizer = AutoTokenizer.from_pretrained(
        configs.labeling_model_name_or_path,
        padding_side = "left"
    )

    # data preprocess
    def preprocess_prompt_detailed_cot_0_shot_preference(example):
        def to_prompt_detailed_cot_0_shot_preference(ds_prompt, summary_a, summary_b, rationale): # see table 2, and appendix N in paper "rlaif: scaling reinforcement learning from human feedback with ai feedback" for more information
            ds_prompt_clean = ds_prompt.split("TL;DR:")[0].strip() # remove the appended suffix.
            prompt = [
                {
                    "role": "user",
                    "content": f"A good summary is a shorter piece of text that has the essence of the original. It tries to accomplish the same purpose and conveys the key information from the original post. Below we define four evaluation axes for summary quality: coherence, accuracy, coverage, and overall quality. \nCoherence: This axis answers the question “how coherent is the summary on its own?” A summary is coherent if it’s easy to understand when read on its own and free of English errors. A summary is not coherent if it’s difficult to understand what the summary is trying to say. Generally, it’s more important that the summary is understandable than it being free of grammar errors. \nAccuracy: This axis answers the question “does the factual information in the summary accurately match the post?” A summary is accurate if it doesn’t say things that aren’t in the article, it doesn’t mix up people, and generally is not misleading. \nCoverage: This axis answers the question “how well does the summary cover the important information in the post?” A summary has good coverage if it mentions the main information from the post that’s important to understand the situation described in the post. A summary has poor coverage if someone reading only the summary would be missing several important pieces of information about the situation in the post. A summary with good coverage should also match the purpose of the original post (e.g. to ask for advice). \nOverall quality: This axis answers the question “how good is the summary overall at representing the post?” This can encompass all of the above axes of quality, as well as others you feel are important. If it’s hard to find ways to make the summary better, the overall quality is good. If there are lots of different ways the summary can be made better, the overall quality is bad. \nYou are an expert summary rater. Given a piece of text and two of its possible summaries, explain which summary best adheres to coherence, accuracy, coverage, and overall quality as defined above. \nText - {ds_prompt_clean} \nSummary 1 - {summary_a} \nSummary 2 - {summary_b} \nConsider the coherence, accuracy, coverage, and overall quality of each summary and explain which one is better."
                },
                {"role": "assistant", "content": f"Rationale: {rationale} \nPreferred Summary="}, 
            ]

            return prompt

        if not reverse_summary_order:
            labeling_prompt = [
                to_prompt_detailed_cot_0_shot_preference(ds_prompt, s_a, s_b, r)
                for (ds_prompt, s_a, s_b, r) in zip(
                    example["prompt"],
                    example["summary_a"],
                    example["summary_b"],
                    example["rationale_cot_0_shot"],
                )
            ]
        else:
            labeling_prompt = [
                to_prompt_detailed_cot_0_shot_preference(ds_prompt, s_a, s_b, r)
                for (ds_prompt, s_a, s_b, r) in zip(
                    example["prompt"],
                    example["summary_b"],
                    example["summary_a"],
                    example["rationale_cot_0_shot_reversed"],
                )
            ]
            
        return {"prompt_preference": labeling_prompt}

    ds_processed = ds.map(preprocess_prompt_detailed_cot_0_shot_preference, batched=True, batch_size=1000, remove_columns=ds.column_names)

    print("start cot stage 2: generating preferences.")

    sampling_params = SamplingParams(temperature=configs.labeling_temperature, top_p=configs.labeling_top_p, max_tokens=configs.labeling_preference_max_new_tokens, logprobs=100) # return logp of the top 100 or 101 (when the sampled token is not in the top 100. )
    outputs = model.chat(ds_processed["prompt_preference"], sampling_params, add_generation_prompt=False, continue_final_message=True)
    preference_text = [output.outputs[0].text.strip() for output in outputs]

    token_id_1 = tokenizer.convert_tokens_to_ids("1")
    token_id_2 = tokenizer.convert_tokens_to_ids("2")
    preference_log_prob = []
    preference_prob = []
    for i in range(len(outputs)):
        logprobs_i = outputs[i].outputs[0].logprobs # a list of dict corresponding to each generated token. length of logprobs is the number of generated tokens. length of each dict is 100 or 101. 
        token_ids_i = outputs[i].outputs[0].token_ids # a list of generated tokens
        # find the location of the token of "1" or "2". 
        idx_1 = np.where(np.array(list(token_ids_i)) == token_id_1)[0]
        idx_2 = np.where(np.array(list(token_ids_i)) == token_id_2)[0]
        if idx_1.size != 0:
            loc = idx_1[0].item()
        elif idx_2.size != 0:
            loc = idx_2[0].item()
        else: 
            loc = -1 # the case when a summary number is not predicted.
        # retrive the log-probability of predicting token "1" and "2". 
        if loc != -1:
            logprobs_i_loc = logprobs_i[loc]
            if idx_1.size != 0:
                log_prob_1 = logprobs_i_loc[token_id_1].logprob
                if token_id_2 in logprobs_i_loc:
                    log_prob_2 = logprobs_i_loc[token_id_2].logprob 
                else: 
                    last_ranked_token_id = list(logprobs_i[0].keys())[-1]
                    logprob_last_ranked_token = logprobs_i_loc[last_ranked_token_id]
                    log_prob_2 = logprob_last_ranked_token
            else:
                if token_id_1 in logprobs_i_loc:
                    log_prob_1 = logprobs_i_loc[token_id_1].logprob 
                else: 
                    last_ranked_token_id = list(logprobs_i[0].keys())[-1]
                    logprob_last_ranked_token = logprobs_i_loc[last_ranked_token_id]
                    log_prob_1 = logprob_last_ranked_token
                log_prob_2 = logprobs_i_loc[token_id_2].logprob

            prob_1 = np.exp(log_prob_1)
            prob_2 = np.exp(log_prob_2)
            preference_log_prob.append([log_prob_1, log_prob_2])
            preference_prob.append([prob_1, prob_2])
        else:
            preference_log_prob.append([np.log(0.5), np.log(0.5)]) # add random probabilities for the two labels
            preference_prob.append([0.5, 0.5])
            
    if not reverse_summary_order:
        ds = ds.add_column(name="preference_text_cot_0_shot", column=preference_text)
        ds = ds.add_column(name="preference_log_prob_cot_0_shot", column=preference_log_prob)
        ds = ds.add_column(name="preference_prob_cot_0_shot", column=preference_prob)

    else:
        ds = ds.add_column(name="preference_text_cot_0_shot_reversed", column=preference_text)
        ds = ds.add_column(name="preference_log_prob_cot_0_shot_reversed", column=preference_log_prob)
        ds = ds.add_column(name="preference_prob_cot_0_shot_reversed", column=preference_prob)
        
    return ds

def extract_preference_label(p):
    chosen_option = p.split("\n")[0]
    if chosen_option in ["(A)", "(B)"]:
        return chosen_option
    else:
        if p.find("(A)") != -1:
            chosen_option = "(A)"
        else:
            chosen_option = "(B)"
        return chosen_option

if __name__ == "__main__":

    # params
    args = add_argument_group("Params")
    args.add_argument(
        "--policy_model_name_or_path", type=str, default="", help="policy model to load."
    )
    args.add_argument(
        "--labeling_model_name_or_path", type=str, default="", help="labeling model to load."
    )
    args.add_argument(
        "--dataset_name", type=str, default="", help="dataset to load."
    )
    args.add_argument(
        "--output_dir", type=str, default="", help="directory to save inference results."
    )
    args.add_argument(
        "--batch_size", type=int, default=32, help="batch size for inference."
    )
    args.add_argument(
        "--rnd_seed", type=int, default=0, help="random seed. "
    )
    args.add_argument(
        "--save_json",
        type=str2bool,
        default=True,
        help="whether to save the json file of the generated dataset.",
    )
    args.add_argument(
        "--build_mode_index", type=int, default=0, help="build mode in [0,1,2]."
    )
    args.add_argument(
        "--target_dir", type=str, default="", help="directory of the summary dataset. only set in build mode 2."
    )
    args.add_argument(
        "--preference_ds_order", type=str, default="original", help="summary orders when labeling preferences. "
    )
    args.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="number of GPUs for vllm distributed inference"
    )
    args.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="whether to run the code in debug mode.",
    )

    configs, unparsed = get_config()

    # generation config summary
    configs.policy_max_new_tokens = 500
    configs.policy_do_sample = True
    configs.policy_temperature = 1.0
    configs.policy_top_p = 1.0  # 0.9

    # generation config preference
    configs.labeling_rationale_max_new_tokens = 512 # see Appendix D
    configs.labeling_preference_max_new_tokens = 5 # see Appendix D
    configs.labeling_temperature = 0.0
    configs.labeling_top_p = 1.0

    # ds building mode
    if configs.debug:
        configs.debug_num_samples = 16

    build_mode_list = [
        "build_summary_ds_only",
        "build_preference_ds_based_on_existing_summary_ds",
        "build_summary_ds_and_preference_ds",
    ]
    build_mode = build_mode_list[configs.build_mode_index]

    # set random seed
    utils.set_all_seeds(configs.rnd_seed)

    if build_mode == "build_summary_ds_only":
        assert configs.target_dir == ""
        assert configs.labeling_model_name_or_path == ""

        # load model and tokenizer
        policy_model_kwargs = {"torch_dtype": torch.float32}
        policy_model = AutoModelForCausalLM.from_pretrained(configs.policy_model_name_or_path, **policy_model_kwargs)
        policy_model = policy_model.to("cuda")
        policy_model.eval()

        policy_tokenizer = AutoTokenizer.from_pretrained(
            configs.policy_model_name_or_path,
            padding_side = "left"
        )
        if policy_tokenizer.pad_token_id is None:
            policy_tokenizer.pad_token_id = policy_tokenizer.eos_token_id

        # prepare exp dir
        exp_dir = utils.prepare_dirs(configs.output_dir, True)
        # config model and model generation
        config_policy_model_generation(policy_model, policy_tokenizer, configs)

        ds = load_from_disk(configs.dataset_name)

        ds = build_summary_ds(policy_model, policy_tokenizer, ds, configs)

        ds_savepath = save_ds(ds, exp_dir, "summary_ds", configs.save_json)
        print(f"a summary dataset was saved to {ds_savepath}")

    elif build_mode == "build_preference_ds_based_on_existing_summary_ds":
        assert configs.policy_model_name_or_path == ""
        assert configs.target_dir != ""
        assert configs.output_dir == ""
        assert configs.dataset_name == ""

        # setup the exp dir
        exp_dir = configs.target_dir

        # load an existing summary dataset
        summary_ds_path = os.path.join(exp_dir, "summary_ds")
        ds = load_from_disk(summary_ds_path)
        print(f"a summary dataset was loaded from {summary_ds_path}")

        if configs.debug:
            ds = ds.select(range(configs.debug_num_samples))


        # load model and tokenizer
        engine_args = {"max_model_len": 4096, "gpu_memory_utilization": 0.90, "max_logprobs": 100}
        if configs.debug:
            model = LLM(model=configs.labeling_model_name_or_path, tensor_parallel_size=configs.tensor_parallel_size, dtype=torch.float16, **engine_args) # to run on GPUs with computation capability lower than 8.0
        else:
            model = LLM(model=configs.labeling_model_name_or_path, tensor_parallel_size=configs.tensor_parallel_size, **engine_args)

        if configs.preference_ds_order == "original":
            # build preference dataset in an original summary order
            ds = build_preference_ds_detailed_cot_0_shot(model, ds, configs, reverse_summary_order=False)
            ds_savepath = save_ds(ds, exp_dir, "preference_ds_only_original_order", configs.save_json)
            print(f"a preference dataset with only original summary order was saved to {ds_savepath}")
        elif configs.preference_ds_order == "reversed":
            # build preference dataset in an reversed summary order
            ds = build_preference_ds_detailed_cot_0_shot(model, ds, configs, reverse_summary_order=True)
            ds_savepath = save_ds(ds, exp_dir, "preference_ds_only_reversed_order", configs.save_json)
            print(f"a preference dataset with only reversed summary order was saved to {ds_savepath}")
        elif configs.preference_ds_order == "both":
            # build preference dataset in both orders
            ds = build_preference_ds_detailed_cot_0_shot(model, ds, configs, reverse_summary_order=False)
            ds = build_preference_ds_detailed_cot_0_shot(model, ds, configs, reverse_summary_order=True)
            ds_savepath = save_ds(ds, exp_dir, "preference_ds_both_orders", configs.save_json)
            print(f"a preference dataset with both original and reversed summary orders was saved to {ds_savepath}")
        else:
            raise

    else:
        raise
