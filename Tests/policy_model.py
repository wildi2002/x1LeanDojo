from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "internlm/internlm2_5-step-prover"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Prepare your input prompt
prompt = """---
NAME: square_sub_one_divisible_eight
---
PROOF_BEFORE: rw [h, pow_two]
---
STATE_BEFORE: m n : ℕ
h : n = 2 * m + 1
⊢ 8 ∣ (2 * m + 1) * (2 * m + 1) - 1
---
TACTIC:
"""

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate the output
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode and print the generated tactic
generated_tactic = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_tactic)
