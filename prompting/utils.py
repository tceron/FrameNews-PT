from datetime import datetime
import os
import pandas as pd
import logging
import json
from pathlib import Path
import hashlib
import time
import copy
from collections import defaultdict


def get_experiment_start_date():
    return datetime.now().strftime("%Y %m %d %H:%M:%S").replace(' ', '_').replace(':', '_')


def get_hash(string, algorithm="sha256"):
    """
    Calculates the hash of a given string using the specified algorithm.
    Defaults to SHA-256 if no algorithm is provided.
    """

    hash_object = getattr(hashlib, algorithm)()
    hash_object.update(string.encode('utf-8'))
    return hash_object.hexdigest()


def timer(func):
    """Decorator that prints the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timer_end_message = f"  :):):) Finished {func.__name__!r} in {run_time:.4f} seconds"
        print(timer_end_message)
        return value

    return wrapper_timer

def get_prompt_guidelines_manifestos():
    with open('./data_prompts/mfc_guidelines_short.txt', 'r') as f:
        data = f.read().replace('\n\n', '|')
        data = data.replace('\n', ' ')
        data = data.replace('|', '\n\n')

    return data


def format_prompt(prompt_template, content, guidelines):

    prompt = copy.deepcopy(prompt_template)
    # check if prompt is of type list
    if isinstance(prompt, list):
        prompt[-1]['content'] = prompt[-1]['content'].format(content=content, 
                                                                guidelines=guidelines
            )
    else:
        prompt = prompt.format(content=content, 
                                guidelines=guidelines
            )
    return prompt
