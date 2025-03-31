import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_input(theorem_name, proof_before, state):
    return (
        f"---\nNAME: {theorem_name}\n\n"
        f"---\nPROOF_BEFORE: {proof_before}\n\n"
        f"---\nSTATE_BEFORE: {state}\n\n"
        f"---\nTACTIC: "
    )

def generate_proof_step(theorem_name, proof_before, state, max_length=100):
    input_text = format_input(theorem_name, proof_before, state)
    
    # Tokenize input with attention mask
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)  # Fix: Add attention mask

    # Generate output
    output_ids = model.generate(input_ids, attention_mask=attention_mask)
    
    # Decode response
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)



# Define the model name
model_name = "internlm/internlm2_5-step-prover"

# Load model & tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example theorem
theorem_name = "square_sub_one_divisible_eight"
proof_before = "rw [h, pow_two]"
state = "m n : ℕ\nh : n = 2 * m + 1\n⊢ 8 | (2 * m + 1) * (2 * m + 1) - 1"

# Generate proof step
output = generate_proof_step(theorem_name, proof_before, state)
print("Generated Tactic:", output)
