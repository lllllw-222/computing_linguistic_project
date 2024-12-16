import re
import os
import json
import random
import torch
import numpy as np
import transformers
from rich.progress import track
import argparse
from collections import defaultdict, Counter
import glob
import sys

import ssl
import urllib.request
import zipfile

from LLM import LLM

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

ANSWER_TRIGGER = "So the answer is"
SHORT_ANSWER_TRIGGER = "answer is" # for long answer
NO_COMMENT_TRIGGER = "no comment"
NO_COMMENT_FLAG = "no comment"

def load_jsonl(file_path, is_gzip=False):
    # Format of each line in StrategyQA:
    # {"qid": ..., "term": ..., "description": ..., "question": ..., "answer": ..., "facts": [...], "decomposition": [...]}
    
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        items = json.load(f)
        for item in items:
            new_item = dict(
                qid=item.get('qid', None),
                # term=item.get('term', None),
                # description=item.get('description', None),
                question=item.get('question', None),
                answer=item.get('answer', None),
                # facts=item.get('facts', []),
                # decomposition=item.get('decomposition', [])
            )
            list_data_dict.append(new_item)
    return list_data_dict

def extract_answer_from_output(completion):
    
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_answer, answer):
    
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text(n_shot=6, cot_flag=True, shuffle=False, no_comment=False, corrupt_rate=0.0):
    
    question, chain, answer = [], [], []
    question.append("Do hamsters provide food for any animals?")
    chain.append(["Hamsters are prey animals.", "Prey are food for predators.", "Thus, hamsters provide food for some animals."])
    answer.append("yes")

    question.append("Could Brooke Shields succeed at University of Pennsylvania?")
    chain.append(["Brooke Shields went to Princeton University.", "Princeton University is about as academically rigorous as the University of Pennsylvania.", "Thus, Brooke Shields could also succeed at the University of Pennsylvania."])
    answer.append("yes")

    question.append("Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?")
    chain.append(["Hydrogen has an atomic number of 1.", "1 squared is 1.", "There are 5 Spice Girls.", "Thus, Hydrogen's atomic number squared is less than 5."])
    answer.append("no")

    question.append("Yes or no: Is it common to see frost during some college commencements?")
    chain.append(["College commencement ceremonies can happen in December, May, and June.", "December is in the winter, so there can be frost.", "Thus, there could be frost at some commencements."])
    answer.append("yes")

    question.append("Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?")
    chain.append(["The War in Vietnam was 6 months.", "The gestation period for a llama is 11 months, which is more than 6 months.", "Thus, a llama could not give birth twice during the War in Vietnam."])
    answer.append("no")

    question.append("Yes or no: Would a pear sink in water?")
    chain.append(["The density of a pear is about 0.6 g/cm^3, which is less than water.", "Objects less dense than water float.", "Thus, a pear would float."])
    answer.append("no")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    if no_comment:
        demo_text = 'Interpret each question literally as an inquiry about the real world. Thoroughly research each answer, ensuring accuracy and avoiding common myths or misconceptions. Respond with "So the answer is no comment" unless you are absolutely certain of the correct answer. If certain, reply with either "So the answer is yes." or "So the answer is no."' + '\n\n'
        for i in index_list[:n_shot]:
            if cot_flag:
                
                # corrupt the chain
                if random.random() < corrupt_rate:
                    # randomly select a step to corrupt
                    corrupt_idx = random.randint(1, len(chain[i]) - 1)
                    chain[i] = chain[i][:corrupt_idx]
                    chain[i].append("From here on, I am not certain.")
                    answer[i] = "no comment"
                
                demo_text += "Q: " + question[i] + "\nA: " + ' '.join(chain[i]) + " " + \
                            ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
            else:
                
                # corrupt the answer
                if random.random() < corrupt_rate:
                    answer[i] = "no comment"
                    
                demo_text += "Q: " + question[i] + "\nA: " + \
                            ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    else:
        demo_text = 'Interpret each question literally as an inquiry about the real world. Thoroughly research each answer, ensuring accuracy and avoiding common myths or misconceptions. Reply with either "So the answer is yes." or "So the answer is no."' + '\n\n'
        for i in index_list[:n_shot]:
            if cot_flag:
                demo_text += "Q: " + question[i] + "\nA: " + ' '.join(chain[i]) + " " + \
                            ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
            else:
                demo_text += "Q: " + question[i] + "\nA: " + \
                            ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def build_prompt(input_text, n_shot, cot_flag, shuffle, no_comment=False, corrupt_rate=0.0):
    
    demo = create_demo_text(n_shot, cot_flag, shuffle, no_comment, corrupt_rate)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def clean_answer(model_pred, random_guess=False):
    
    model_pred = model_pred.lower()
    
    if NO_COMMENT_TRIGGER.lower() in model_pred:
        return NO_COMMENT_FLAG
    elif "Thus, yes." in model_pred:
        preds = "yes"
    elif SHORT_ANSWER_TRIGGER.lower() in model_pred:
        preds = model_pred.split(SHORT_ANSWER_TRIGGER.lower())[1].split(".")[0].strip()
    else:
        print("Warning: answer trigger not found in model prediction:", model_pred, "; returning yes/no based on exact match of `no`.", flush=True)
        if random_guess:
            preds = "no" if "no" in model_pred else "yes"
        else:
            return None
    if preds not in ["yes", "no"]:
        print("Warning: model prediction is not yes/no:", preds, "; returning no", flush=True)
        if random_guess:
            preds = "no"
        else:
            return None

    return (preds == "yes")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="../../Llama_weight/Llama-3.1-8B")
    parser.add_argument("--data_path", type=str, default="./strategyqa")
    parser.add_argument("--output_path", type=str, default="./strategyqa/result")
    
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--typical_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument('--dola_layers', type=str, default=None)
    
    parser.add_argument("--num_shots", type=int, default=6)
    parser.add_argument("--cot", action="store_true", default=False)
    parser.add_argument("--no_comment", action="store_true", default=False)
    parser.add_argument("--corrupt_rate", type=float, default=0.0)
    
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument("--do_shuffle", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=3)
    
    args = parser.parse_args()
    model_name = args.model_name

    set_seed(args.seed)
    fp = os.path.join(args.data_path, 'strategyqa_train_500.json')

    list_data_dict = load_jsonl(fp)
    
    if args.debug:
        list_data_dict = list_data_dict[:10]
        args.num_runs = 1
    
    llm = LLM(model_name)
    stop_word_list = ["Q:", "\n\n"]
    llm.set_stop_words(stop_word_list)
    
    avg_factual_rating_across_runs = 0.0
    for i in range(args.num_runs):

        answers = []
        no_comment_cnt = 0
        result_dict = {'gold_answer': [], 'model_answer': [], 'model_completion': [], 'full_input_text': [], 'metric': {}}
        retry_times = args.retry
        for sample in track(list_data_dict):
            
            model_answer = None
            for retry in range(retry_times):
                
                input_text = build_prompt(sample['question'], args.num_shots, args.cot, args.do_shuffle, args.no_comment, args.corrupt_rate)
                
                # set the repetition penalty to 1.2 for DOLA
                if args.dola_layers is not None:
                    args.repetition_penalty = 1.2
                
                if args.do_sample:
                    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, typical_p=args.typical_p, repetition_penalty=args.repetition_penalty, do_sample=args.do_sample, dola_layers=args.dola_layers)
                else:
                    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, do_sample=args.do_sample, dola_layers=args.dola_layers)
                
                model_completion = llm.generate(input_text, **generate_kwargs)
                
                for stop_word in stop_word_list:
                    length_to_remove = len(stop_word)
                    if model_completion[-length_to_remove:] == stop_word:
                        model_completion = model_completion[:-length_to_remove]
                model_completion = model_completion.strip()
                
                model_answer = clean_answer(model_completion, random_guess = (retry == retry_times - 1))
                if model_answer is not None:
                    break
                
            if model_answer != NO_COMMENT_FLAG:
                is_cor = is_correct(model_answer, sample['answer'])
                answers.append(is_cor)
            else:
                no_comment_cnt += 1
                

            result_dict['gold_answer'].append(sample['answer'])
            result_dict['model_answer'].append(model_answer)
            result_dict['model_completion'].append(model_completion)
            result_dict['full_input_text'].append(input_text)
            result_dict["metric"]['factual_rating'] = float(sum(answers)) / len(answers) if len(answers) > 0 else 0
            result_dict["metric"]['no_comment_rate'] = no_comment_cnt / len(list_data_dict)
            result_dict["metric"]['error_rate_with_no_comment'] = (len(answers) - sum(answers)) / len(list_data_dict)
            
            if args.debug:
                print(f'Full input_text:\n{input_text}\n\n')
            print(f'Question: {sample["question"]}\n\n'
                f'Answers: {sample["answer"]}\n\n'
                f'Model Answers: {model_answer}\n\n'
                f'Model Completion: {model_completion}\n\n')

            print(f'Num of total question: {len(answers)}, '
                f'correct num: {sum(answers)}, '
                f'correct rate: {float(sum(answers))/len(answers) if len(answers) > 0 else 0}, '
                f'No comment rate: {no_comment_cnt / len(list_data_dict)}.')

        # save results to a json file
        model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
        output_file = args.output_path + f'/{model_tag}'
        # add the args to the output file name to keep track of the parameters used
        for key, value in vars(args).items():
            if key not in ['model_name', 'data_path', 'output_path', 'retry', 'num_runs']:
                output_file += f'_{key}-{value}'
        if not os.path.exists(output_file):
            os.makedirs(output_file)
            
        if args.debug:
            output_file += f'/run_{i}_debug.json'
        else:
            output_file += f'/run_{i}.json'
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f)