import argparse
import json
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as numpy
from transformers import AutoTokenizer

def run_eval(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    temperature,
    tp_size,
):
    print('##################'+str(torch.cuda.is_available()))
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ['RANK'])
    print(os.environ["CUDA_VISIBLE_DEVICES"])

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
    
    print(f"RANK: {os.environ['SLURM_PROCID']} | NUM_REPLICAS: {os.environ['SLURM_NTASKS']} | DEVICE {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Output to {answer_file}")
    print(f"Num Questions: {len(questions)}")
    print(f"TP: {tp_size}")
    device = 'cuda:' + os.environ['SLURM_PROCID']
  
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

        dir_name = os.path.dirname(answer_file)
        if dir_name:  # Nur wenn ein Verzeichnis vorhanden ist
            os.makedirs(dir_name, exist_ok=True)

        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps(question, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
   
    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        required=True,
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=None,
        required=True,
        help="The output answer file.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--repeat_times",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ['SLURM_PROCID'])

    with open(args.question_file, 'r') as f:
        questions = [json.loads(item) for item in f.readlines()]
    
    if args.repeat_times > 1:
        questions = questions * args.repeat_times
    
    if 'SLURM_NTASKS' in os.environ and int(os.environ['SLURM_NTASKS']) > 1:
        num_replicas = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        
        tp_size = 1
        device = ','.join([str(i) for i in range(rank*tp_size, (rank+1)*tp_size)])

        print(num_replicas, rank, tp_size, device)

        total_size = len(questions)
        questions = questions[rank:total_size:num_replicas]
        args.answer_file = args.answer_file.replace(".jsonl", f"_{rank}.jsonl")
    else:
        tp_size = 1

    run_eval(
        args.model_path,
        args.model_path,
        questions,
        args.answer_file,
        args.max_new_token,
        args.temperature,
        tp_size,
    )