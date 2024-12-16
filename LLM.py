import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, StopStringCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

class LLM:
    
    def __init__(self, model_name):
        
        self.model_name = model_name
        self.stopping_criteria = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        for stop_word in self.stop_words:
            self.stopping_criteria.append(StopStringCriteria(self.tokenizer, stop_word))

    def generate(self, input_text, **kwargs):
        
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
            
            outputs = self.model.generate(input_ids, num_return_sequences=1, output_scores=True, return_dict_in_generate=True, stopping_criteria=self.stopping_criteria, **kwargs)
                
            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

        return output_str