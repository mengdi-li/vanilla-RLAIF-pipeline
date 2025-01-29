# use GPT-4o as a judge model to caculate win rates. 
# based on example: trl/examples/scripts/evals/judge_tldr.py

""" compete with human annoatations
python ./rlaif/gpt2/summarization/tldr/evaluate_win_rate.py \
    --model_name_or_path "./rlaif/gpt2/summarization/tldr/exps/policy_rlhf/2025-01-28 03:49:49.365820/checkpoint-912" \
    --judge_model "gpt-4o-2024-08-06" \
    --num_examples 1000 \
    --ref_mode "human_annotations" 
"""

""" compete with a reference model 
python ./rlaif/gpt2/summarization/tldr/evaluate_win_rate.py \
    --model_name_or_path "./rlaif/gpt2/summarization/tldr/exps/policy/2025-01-22 13:51:26.686550/checkpoint-912" \
    --judge_model "gpt-4o-2024-08-06" \
    --num_examples 1000 \
    --ref_mode "reference_model" \
    --ref_model_name_or_path "./rlaif/gpt2/summarization/tldr/exps/policy_rlhf/2025-01-28 03:49:49.365820/checkpoint-912"
"""

from dataclasses import dataclass, field
from typing import Optional
import gc

import torch
from datasets import load_dataset
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from trl import HfPairwiseJudge, OpenAIPairwiseJudge

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "The model name or path to the model to evaluate."})
    judge_model: str = field(
        default="gpt-4o-2024-08-06",
        metadata={
            "help": "The model name or path to the model to use as a judge. E.g., 'gpt-3.5-turbo-0125', 'meta-llama/Meta-Llama-3-70B-Instruct'."
        },
    )
    num_examples: Optional[int] = field(default=None, metadata={"help": "The number of examples to evaluate."})
    ref_mode: str = field(
        default="human_annotations",
        metadata={
            "help": "Reference completions to use in the competition to calulate win rate. Possible modes: 'human_annotations', 'reference_model'. "
        }
    )
    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "the reference model to use when the 'ref_mode' is 'reference_model'. "}
    ) 


# Parse the arguments
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the dataset
dataset = load_dataset("trl-lib/tldr", split="validation")
if script_args.num_examples is not None:
    dataset = dataset.select(range(script_args.num_examples))

# Extract the prompts and config generation
prompts = dataset["prompt"]
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=128)  # very generous max token length

# Generate reference completions
if script_args.ref_mode == "human_annotations":
    reference_completions = dataset["completion"]
elif script_args.ref_mode == "reference_model":
    assert script_args.ref_model_name_or_path != None
    cuda_cap_max = torch.cuda.get_device_capability(0)[0]
    if cuda_cap_max < 8.0:
        ref_llm = LLM(model=script_args.ref_model_name_or_path, tensor_parallel_size=1, dtype=torch.float16)
    else:
        ref_llm = LLM(model=script_args.ref_model_name_or_path, tensor_parallel_size=1, dtype=torch.float32)
    ref_outputs = ref_llm.generate(prompts, sampling_params) # use the sample sampling parameters as the evaluation model
    reference_completions = [output.outputs[0].text.strip() for output in ref_outputs]
    # Delete the llm object and free the memory
    destroy_model_parallel()
    del ref_llm.llm_engine.model_executor.driver_worker
    del ref_llm
    gc.collect()
    torch.cuda.empty_cache()
    print("Successfully delete the llm pipeline and free the GPU memory!")
else:
    raise

# Generate model completions
cuda_cap_max = torch.cuda.get_device_capability(0)[0]
if cuda_cap_max < 8.0:
    llm = LLM(model=script_args.model_name_or_path, tensor_parallel_size=1, dtype=torch.float16)
else:
    llm = LLM(model=script_args.model_name_or_path, tensor_parallel_size=1, dtype=torch.float32)
outputs = llm.generate(prompts, sampling_params)
model_completions = [output.outputs[0].text.strip() for output in outputs]


# Judge the outputs
if "gpt" in script_args.judge_model:
    judge = OpenAIPairwiseJudge(script_args.judge_model)
else:
    judge = HfPairwiseJudge(script_args.judge_model)

completions = [[c0, c1] for c0, c1 in zip(reference_completions, model_completions)]
best_idxs = judge.judge(prompts, completions)
model_win_rate = best_idxs.count(1) / len(best_idxs)
print(f"Model win rate: {model_win_rate*100:.2f}%")
