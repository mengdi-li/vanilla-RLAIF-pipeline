# analyse the exisitence of positional bias when using llama-3.3-70B-Instruct for annotating preference feedback.

"""
python ./rlaif/gpt2/summarization/tldr/analyse_aif_positional_bias.py
"""

from datasets import load_from_disk
import numpy as np

preference_ds_orginal_order = load_from_disk("./rlaif/gpt2/summarization/tldr/exps/ai_feedback_preference_dataset/full-train-split-14776samples/preference_ds_only_original_order")
preference_ds_reversed_order = load_from_disk("./rlaif/gpt2/summarization/tldr/exps/ai_feedback_preference_dataset/full-train-split-14776samples/preference_ds_only_reversed_order") 


ds1 = preference_ds_orginal_order
ds2 = preference_ds_reversed_order

matched = 0
pos1_preferred = 0
pos2_preferred = 0
not_answered = 0 # we set the logp values equal for two options when the key words "1" or "2" are not appeared in the response. these samples can be removed from the training data for RM training. 

for i in range(ds1.num_rows):
    logp_ds1_i = ds1[i]["preference_log_prob_cot_0_shot"]
    logp_ds2_i = ds2[i]["preference_log_prob_cot_0_shot_reversed"]
    if logp_ds1_i[0] == logp_ds1_i[1] or logp_ds2_i[0] == logp_ds2_i[1]:
        not_answered += 1
    else:
        pref_ds1 = np.argmax(logp_ds1_i)
        pref_ds2 = np.argmax(logp_ds2_i)
        if pref_ds1 != pref_ds2:
            matched += 1 
        elif pref_ds1 == 0:
            pos1_preferred += 1
            breakpoint()
        else:
            pos2_preferred += 1

print(f"matched: {matched / ds1.num_rows}")
print(f"position 1 preferred: {pos1_preferred / ds1.num_rows}")
print(f"position 2 preferred: {pos2_preferred / ds1.num_rows}")
print(f"not answered: {not_answered / ds1.num_rows}")