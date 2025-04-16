import os
import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class Lean4Translator:
    def __init__(
        self,
        model_path="internlm/internlm2-math-base-7b",
        model_id="internlm/internlm2-math-base-7b",
        max_new_token=1000,
        temperature=0.01,
        tp_size=1,
    ):
        print('CUDA available:', torch.cuda.is_available())
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"))

        self.model_id = model_id
        self.tp_size = tp_size
        self.max_new_token = max_new_token
        self.temperature = temperature

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        special_tokens_dict = {}
        if self.tokenizer.pad_token is None: special_tokens_dict["pad_token"] = '<unk>'
        if self.tokenizer.eos_token is None: special_tokens_dict["eos_token"] = '</s>'
        if self.tokenizer.bos_token is None: special_tokens_dict["bos_token"] = '<s>'
        if self.tokenizer.unk_token is None: special_tokens_dict["unk_token"] = '<unk>'
        if special_tokens_dict and 'Qwen' not in model_path:
            self.tokenizer.add_special_tokens(special_tokens_dict)

        # Load model
        try:
            self.model = LLM(model=model_path, tensor_parallel_size=tp_size, trust_remote_code=True, dtype="bfloat16")
        except RecursionError:
            self.model = LLM(model=model_path, tokenizer_mode='slow', tensor_parallel_size=tp_size, trust_remote_code=True, dtype="bfloat16")

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_token,
            stop=['[UNUSED_TOKEN_146]', '[UNUSED_TOKEN_145]', 'by', 'sorry']
        )

    def get_prompt(self, example):
        if example['type'] == 1:
            print(">> Translating: ")
            if 'answer' in example and 'prove' not in example['problem'].split() and 'Prove' not in example['problem'].split() and example['answer'] and len(example['answer']) <= 30:
                prompt = f"[UNUSED_TOKEN_146]user\nConvert following problem into LEAN 4:\n{example['problem']} Show that it is {example['answer']}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nHere is the formal statement in LEAN 4:\n```lean\ntheorem"
            else:
                prompt = f"[UNUSED_TOKEN_146]user\nConvert following problem into LEAN 4:\n{example['problem']}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nHere is the formal statement in LEAN 4:\n```lean\ntheorem"
            print(prompt)
            return prompt

        elif example['type'] == 2:
            print(">> Comparing: ")
            prompt = f"Given a question and two answers, which one is better? \nQuestion: {example['problem']}\nAnswer 1: {example['cot1']}\nAnswer 2: {example['cot2']}"
            print(prompt)
            return prompt

        else:
            raise Exception("not valid prompt type")
    


    def translate(self, problem):
        question = {"type": 1, "problem": problem}
        prompt = self.get_prompt(question)
        outputs = self.model.generate([prompt], self.sampling_params)

        output_ids = outputs[0].outputs[0].token_ids
        output = self.model.get_tokenizer().decode(output_ids, spaces_between_special_tokens=False)

        for special_token in self.model.get_tokenizer().special_tokens_map.values():
            if isinstance(special_token, list):
                for tok in special_token:
                    output = output.replace(tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        print(">>> " + output)
        return output

    def compare(self, problem, cot1, cot2):
        question = {"type": 2, "problem": problem, "cot1": cot1, "cot2": cot2}
        prompt = self.get_prompt(question)
        outputs = self.model.generate([prompt], self.sampling_params)

        output_ids = outputs[0].outputs[0].token_ids
        output = self.model.get_tokenizer().decode(output_ids, spaces_between_special_tokens=False)

        for special_token in self.model.get_tokenizer().special_tokens_map.values():
            if isinstance(special_token, list):
                for tok in special_token:
                    output = output.replace(tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        print(">>> " + output)
        return output


# Beispielnutzung
if __name__ == "__main__":
    translator = Lean4Translator()
    problem = "Show that the sum of two even numbers is always even."
    cot1 = (
        translator.translate("Definition of an even number a: a = 2n where n is a natural number.") + "\n"
        + translator.translate("Addition of a + b = 2n + 2m") + "\n"
        + translator.translate("Factoring: 2n + 2m = 2(n+m)") + "\n"
        + translator.translate("a + b = 2(n + m) is even.")
    )

    print("> " + cot1)
    cot2 = (
        translator.translate("Definition of an even number a: a = 2n where n is a natural number.") + "\n"
        + translator.translate("Addition of a + b = 2n + 2m") + "\n"
        + translator.translate("Factoring: 2n + 2m = 2(n+m)") + "\n"
        + translator.translate("a + b = 2(n + m) is even.")
    )

    print("> " + cot2)
    
    translator.compare(problem, cot1, cot2)