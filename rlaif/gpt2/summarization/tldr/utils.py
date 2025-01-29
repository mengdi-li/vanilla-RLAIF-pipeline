import os
import sys
import random
import numpy as np
import torch
from transformers import set_seed


import pathlib
import datetime
import dateutil
import dateutil.tz


def prepare_dirs(exp_dir, create_random_sub_dir):
    if create_random_sub_dir:
        dir_name = str(datetime.datetime.now(dateutil.tz.tzlocal())).split("+")[0]
        exp_dir = os.path.join(exp_dir, dir_name)
    else:
        exp_dir = exp_dir

    pathlib.Path(exp_dir).mkdir(parents=True, exist_ok=False)
    return exp_dir


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    print(f"Random seed set as {seed}")


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
