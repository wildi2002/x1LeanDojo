import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "internlm/internlm2_5-step-prover-critic", 
    device_map="cuda" if torch.cuda.is_available() else "cpu", 
    torch_dtype=torch.float16, 
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-step-prover-critic", trust_remote_code=True)

chat_1 = [
    {"role": "user", "content": "Which state is closer to 'no goals'?"},
    {"role": "assistant", "content": "no goals"}
]
chat_2 = [
    {"role": "user", "content": "Which state is closer to 'no goals'?"},
    {"role": "assistant", "content": "x : ℕ\nh₀ : ↑x + 4 / 100 * ↑x = 598\n⊢ 100 * x = 100 * 575"}
]
chat_3 = [
    {"role": "user", "content": "Is this step correct?"},
    {"role": "assistant", "content": "x^2 = 4 -> x \in {-2,2}"}
]
chat_4 = [
    {"role": "user", "content": "Which state is closer to 'no goals'?"},
    {"role": "assistant", "content": "x : ℕ\nh₀ : ↑x + 4 / 100 * ↑x = 598\n⊢ 104 * x = 100 * 598"}
]

score1 = model.get_score(tokenizer, chat_1)
score2 = model.get_score(tokenizer, chat_2)
score3 = model.get_score(tokenizer, chat_3)
score4 = model.get_score(tokenizer, chat_4)
print("score1: ", score1)
print("score2: ", score2)
print("score3: ", score3)
print("score4: ", score4)
