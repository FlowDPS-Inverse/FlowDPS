import random
from typing import List
from pathlib import Path

import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def process_prompt_file(prompt_file: str, **kwargs):
    parse_fn = kwargs.get('parse_fn', None)
    if parse_fn is None:
        # default setup for DIV2K prompt from DAPE
        def _parse_fn(text):
            tmp = text.split(": ")[-1].strip()
            if tmp == '':
                return "a high quality photo"
            else:
                return "a high quality photo of " + tmp

        parse_fn = lambda x: _parse_fn(x)

    with open(prompt_file, 'r') as f:
        prompts = f.readlines()
    prompts = [parse_fn(x) for x in prompts]
    return prompts


def process_text(prompt: str=None, prompt_file: str=None, **kwargs) -> List[str]:
    assert prompt is not None or prompt_file is not None, \
        print("Either prompt of prompt_file must be given.")

    if prompt is not None:
        if prompt_file is not None:
            print("Both prompt and prompt_file are given. We will use prompt.")
        prompts = [prompt]
    else:
        prompts = process_prompt_file(prompt_file, **kwargs)

    return prompts


def get_img_list(root: Path):
    if root.is_dir():
        files = list(sorted(root.glob('*.png'))) \
                + list(sorted(root.glob('*.jpg'))) \
                + list(sorted(root.glob('*.jpeg')))
    else:
        files = [root]

    for f in files:
        yield f
