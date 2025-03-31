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

score1 = model.get_score(tokenizer, chat_1)
score2 = model.get_score(tokenizer, chat_2)
print("score1: ", score1)
print("score2: ", score2)
