This codebase is an implementation of a vanilla RLAIF pipeline as described in the paper "RLAIF vs. RLHF: Scaling Reinforcement Learning  from Human Feedback with AI Feedback" (https://arxiv.org/abs/2309.00267). 

The implementation is intended for research and learning purposes, utilizing GPT-2-Large for the summarization task with the TL;DR dataset. 

# Installation
## clone project code
```
git clone git@github.com:mengdi-li/Vanilla-RLAIF-Pipeline.git
cd Vanilla-RLAIF-Pipeline
```

## create conda environment
```
conda create -n rlaif python=3.12.7
conda activate rlaif
```

## install basic python packages
```
pip install -r requirements.txt
```

# RLAIF Pipeline
Use default parameters in each script to reproduce our reported results. 

## SFT
Implemented in python script `./rlaif/gpt2/summarization/tldr/sft.py`. See detailed running commands in the python script. 

## RLAIF

### Preference dataset construction
The following two steps can be integrated later. We use seperated steps for easier and more efficient debugging. 

- Filter out unique posts from the original openai preference dataset
  - Implemented in python script `./rlaif/gpt2/summarization/tldr/preprocess_openai_tldr_human_feedback_dataset.py`. See detailed running commands in the python script. 

- Generate a summary dataset using the SFT model
  - Implemented in python script `./rlaif/gpt2/summarization/tldr/build_ai_feedback_preference_dataset_vllm.py`. See detailed running commands in the python script. 

- Generate preference dataset using an off-the-shelf model, e.g, Llama-3.3-70B-Instruct, gpt-4o
  - Implemented in python script `./rlaif/gpt2/summarization/tldr/build_ai_feedback_preference_dataset_vllm.py`. See detailed running commands in the python script. 

- Analyse position bias in the constructed preference dataset
  - Implemented in python script `./rlaif/gpt2/summarization/tldr/analyse_aif_positional_bias.py`. See detailed running commands in the python script. 

### Traing Reward Model 
- Preprocess preference data
  - Implemented in python script `rlaif/gpt2/summarization/tldr/preprocess_preference_dataset_for_rm_training.py`. See detailed running commands in the python script. 

- Training 
  - Implemented in python script `rlaif/gpt2/summarization/tldr/train_rm.py`. See detailed running commands in the python script. 
  - Note: To train the reward model using soft labels, we need to modify the TRL library. We observed that training with soft labels results in smoother training curves compared to training with hard labels; however, their final accuracies are similar.

- Evaluate reward model
  - Implemented in python script `rlaif/gpt2/summarization/tldr/evaluate_rm_acc.py`. See detailed running commands in the python script. 
  - Accuracy on ai-annotated preference dataset: 72.9%
  - Accuracy on human-annotated preference dataset (with a confidence threshold of 8): 61.0% 

### Train Policy using PPO
Implemented in python script `rlaif/gpt2/summarization/tldr/train_policy.py`. See detailed running commands in the python script. 

- Evaluate the win rate of the policy
  - Implemented in python script `rlaif/gpt2/summarization/tldr/evaluate_win_rate.py`. See detailed running commands in the python script. 
  - Win rate
    - rlaif policy v.s. sft policy: 68.1%
    - rlaif policy v.s. human annotations: 47.6%

# RLHF pipeline (for reference)
In this part, we finetune the model obtained from SFT using RLHF.

## RLHF
### Train Reward Model
Implemented in python script `./rlaif/gpt2/summarization/tldr/train_rm_rlhf.py`. See detailed running commands in the python script. 
- Evaluate reward model
  - Implemented in python script `rlaif/gpt2/summarization/tldr/evaluate_rm_acc.py`. See detailed running commands in the python script. 
  - Accuracy on human-annotated preference dataset (with a confidence threshold of 8): 69.3% 

### Train Policy
Implemented in python script `./rlaif/gpt2/summarization/tldr/train_policy_rlhf.py`. See detailed running commands in the python script. 
- Evaluate the win rate of the policy
  - Implemented in python script `rlaif/gpt2/summarization/tldr/evaluate_win_rate.py`. See detailed running commands in the python script. 
  - Win rate
    - rlaif policy v.s. sft policy: 54.9% (need further tuning)
    - rlaif policy v.s. human annotations: 35.1% (need further tuning)