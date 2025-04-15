import argparse
import json
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as numpy
from transformers import AutoTokenizer

def run_eval(
    problem,
    model_path = "internlm/internlm2-math-base-7b",
    model_id = "internlm/internlm2-math-base-7b",
    max_new_token = 1000,
    temperature = 0.01,
    tp_size = 1,
):
    print('##################'+str(torch.cuda.is_available()))
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ['RANK'])
    print(os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"))

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = '<unk>'
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = '</s>'
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = '<s>'
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = '<unk>'
    if len(special_tokens_dict) > 0 and model_path.find('Qwen') == -1:
        tokenizer.add_special_tokens(special_tokens_dict)
    
    rank = os.environ.get("SLURM_PROCID", "0")
    num_replicas = os.environ.get("SLURM_NTASKS", "1")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")

    print(f"RANK: {rank} | NUM_REPLICAS: {num_replicas} | DEVICE {cuda_visible_devices}")
    print(f"Question: {question}")
    print(f"TP: {tp_size}")

    device = 'cuda:' + rank
  
    try:
        model = LLM(model=model_path, tensor_parallel_size=tp_size, trust_remote_code=True, dtype="bfloat16")
    except RecursionError:
        model = LLM(model=model_path, tokenizer_mode='slow', tensor_parallel_size=tp_size, trust_remote_code=True, dtype="bfloat16")
    
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_token, stop=['[UNUSED_TOKEN_146]', '[UNUSED_TOKEN_145]', 'by', 'sorry'])

    def get_query(example):
        if 'answer' in example and 'prove' not in example['problem'].split(' ') and 'Prove' not in example['problem'].split(' ') and example['answer'] != '' and len(example['answer']) <= 30:
            return "[UNUSED_TOKEN_146]user\nConvert following problem into LEAN 4:\n" + str(example['problem']) + "Show that it is " + str(example['answer']) + "[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nHere is the formal statement in LEAN 4:\n```lean\ntheorem"
        else:
            return "[UNUSED_TOKEN_146]user\nConvert following problem into LEAN 4:\n" + str(example['problem']) + "[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nHere is the formal statement in LEAN 4:\n```lean\ntheorem"

    questions = [{ "problem": problem }]
    prompts = [get_query(example) for example in questions]

    prompt_id_map = {prompt: idx for idx, prompt in enumerate(prompts)}

    outputs = model.generate(prompts, sampling_params)

    for _, output in enumerate(outputs):
        output_ids = output.outputs[0].token_ids
        question = questions[prompt_id_map[output.prompt]]

        output = model.get_tokenizer().decode(
            output_ids,
            spaces_between_special_tokens=False,
        )

        for special_token in model.get_tokenizer().special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()

        question['output'] = output
        question['generator'] = model_id

        return json.dumps(question, ensure_ascii=False) + "\n"

if __name__ == "__main__":
    run_eval(
        "Show that the sum of two even numbers is always even."
    )