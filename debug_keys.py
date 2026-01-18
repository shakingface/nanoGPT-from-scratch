from src.train_gpt2 import GPT, GPTConfig
from transformers import GPT2LMHeadModel
import torch

# Init local model
config = GPTConfig(bias=True)
model = GPT(config)
sd = model.state_dict()
sd_keys = sd.keys()
sd_keys = [k for k in sd_keys if not k.endswith('attn.bias')]
print(f"Local keys: {len(sd_keys)}")
sorted_keys = sorted(list(sd_keys))
# print(sorted_keys)

# Init HF model
model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
sd_hf = model_hf.state_dict()
sd_keys_hf = sd_hf.keys()
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
print(f"HF keys: {len(sd_keys_hf)}")

# Compare
keys_local = set(sd_keys)
keys_hf = set(sd_keys_hf)

print("In HF but not in Local:")
print(keys_hf - keys_local)

print("In Local but not in HF:")
print(keys_local - keys_hf)
