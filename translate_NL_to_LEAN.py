import os
import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class Lean4Verification:
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
    
    def run_model(self, question):
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
        #print(">>> " + output)
        return output

    def get_prompt(self, example):
        if example['type'] == 1:
            #print(">> Translating: ")
            if 'answer' in example and 'prove' not in example['problem'].split() and 'Prove' not in example['problem'].split() and example['answer'] and len(example['answer']) <= 30:
                prompt = f"[UNUSED_TOKEN_146]user\nConvert following problem into LEAN 4:\n{example['problem']} Show that it is {example['answer']}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nHere is the formal statement in LEAN 4:\n```lean\ntheorem"
            else:
                prompt = f"[UNUSED_TOKEN_146]user\nConvert following problem into LEAN 4:\n{example['problem']}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nHere is the formal statement in LEAN 4:\n```lean\ntheorem"
            #print(prompt)
            return prompt
        elif example['type'] == 2:
            #print(">> Comparing: ")
            prompt = f"[UNUSED_TOKEN_146]user\nGiven a question and two answers, which one is better? \nQuestion: {example['problem']}\nAnswer 1: {example['cot1']}\nAnswer 2: {example['cot2']}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"
            #print(prompt)
            # if "Answer 1 is better" in prompt:
            #     return 1
            # elif "Answer 2 is better" in prompt:
            #     return 2
            # elif "Answer 2 is better" in prompt:
            #     return 0
            # else:
            #     raise Exception("Neither answer one nor answer 2 is better.")
            
            return prompt
        else:
            raise Exception("not valid prompt type")
    
    def translate(self, problem):
        question = {"type": 1, "problem": problem}
        return self.run_model(question)

    def compare(self, problem, cot1, cot2):
        question = {"type": 2, "problem": problem, "cot1": cot1, "cot2": cot2}
        return self.run_model(question)


# Beispielnutzung
if __name__ == "__main__":
    lean = Lean4Verification()

    problem = "Show that the sum of two even numbers is always even."
    print(lean.compare(problem, ("""An even number can be written as 2 times a natural number. Let a = 2n and b = 2m for some natural numbers n and m. Then a + b = 2n + 2m. Factor the expression: 2n + 2m = 2(n + m). Since n + m is a natural number, a + b is divisible by 2, hence even."""), 
                             ("""Even numbers are defined as numbers that are divisible by 2. Assume a and b are even, so there exist integers k and l such that a = 2k and b = 2l. Then, a + b = 2k + 2l. This simplifies to a + b = 2(k + l). Therefore, a + b is divisible by 2, so it is even.""")))
    
    problem = "Prove that the product of two odd numbers is always odd."
    print(lean.compare(problem,
        ("""Let a = 2n + 1 and b = 2m + 1 be two odd numbers. Then a * b = (2n + 1)(2m + 1) = 4nm + 2n + 2m + 1 = 2(2nm + n + m) + 1, which is of the form 2k + 1, so it is odd."""),
        ("""Odd numbers can be written as 2k + 1. The product is (2k + 1)(2l + 1) = 4kl + 2k + 2l + 1 = 2(2kl + k + l) + 1. Since this is one more than an even number, the product is odd.""")))

    problem = "Prove that the square of an even number is even."
    print(lean.compare(problem,
        ("""Let a be an even number, so a = 2n for some integer n. Then a² = (2n)² = 4n² = 2(2n²), which is divisible by 2, hence even."""),
        ("""An even number can be expressed as 2k. Squaring gives (2k)² = 4k² = 2(2k²), which is clearly even since it's a multiple of 2.""")))

    problem = "Show that the sum of an odd number of consecutive integers is divisible by the number of terms."
    print(lean.compare(problem,
        ("""Let the sequence be centered at 0: -n, ..., 0, ..., n. The number of terms is 2n + 1, which is odd. Their sum is zero due to symmetry, and zero is divisible by any integer, including 2n + 1."""),
        ("""Let the numbers be a, a+1, ..., a+2n. The number of terms is 2n + 1. The sum is (2n + 1)(a + n), and since (2n + 1) divides the product, it divides the sum.""")))


