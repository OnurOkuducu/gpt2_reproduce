import math
import torch
from datasets import load_dataset
import tiktoken
from train_gpt_with_inference import GPT

device = "cuda"

# === VANILLA GPT-2 CONFIG ===
vocab_size = 50304
num_layers = 12      # GPT-2 small
num_heads = 6
model_dim = 768
max_seq_len = 1024

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

sliding_window_num_blocks = torch.tensor(1, device="cuda")

@torch.no_grad()
def score_continuation_vanilla(model, prompt: str, continuation: str):
    model.eval()

    #breakpoint()
    # Tokenize
    prompt_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    cont_ids   = torch.tensor(encode(" " + continuation), dtype=torch.int32, device=device)

    input_ids = torch.cat([prompt_ids, cont_ids])  # [T]
    #print('early input shape:', input_ids.shape)
    BLOCK_SIZE = 128
    PAD_TOKEN  = 50256  # <|endoftext|> used as pad in your generate()

    # --- Left-pad to multiple of BLOCK_SIZE (exactly like generate) ---
    pad_len = (-len(input_ids)) % BLOCK_SIZE
    if pad_len > 0:
        pad = torch.full((pad_len,), PAD_TOKEN, dtype=input_ids.dtype, device=device)
       # print('pad shape: ', pad.shape)
        input_ids = torch.cat([pad, input_ids])  # [pad ...][prompt][cont]

    # Forward
    logits = model.inference(input_ids,sliding_window_num_blocks)  # [1, T, V]
    logits = logits[0]
    # We need next-token logits for each position: use all except last position
    logits = logits[:-1, :vocab_size]             # [T-1, V]
    logprobs = torch.log_softmax(logits, dim=-1)  # [T-1, V]

    # Targets are inputs shifted left by 1
    target = input_ids[1:]                        # [T-1]

    # Continuation region indices (account for left padding)
    cont_start = len(prompt_ids)                  # tokens after the prompt start
    cont_start_idx = pad_len + cont_start        # absolute index in padded sequence

    # Off-by-one: token at position i is predicted by logits[i-1]
    # We want to score tokens from cont_start_idx .. end-1, so slice from cont_start_idx-1
    lp_slice   = logprobs[cont_start_idx - 1 :, :]            # [C, V]
    tgt_slice  = target  [cont_start_idx - 1 :]               # [C]

    token_lps = lp_slice.gather(1, tgt_slice.unsqueeze(-1)).squeeze(-1)
    sum_lp = token_lps.sum().item()
    avg_lp = sum_lp / max(1, tgt_slice.numel())
    return sum_lp, avg_lp


print("Loading HellaSwag...")
ds = load_dataset("hellaswag", split="validation")

n = 400
correct = 0
correct_norm = 0

checkpoint_path = '/workspace/gpt2_reproduce/logs/13fc61ea-5ae8-4645-8bea-d15a25c49709/state_step015999.pt'

print(checkpoint_path)
model = GPT(vocab_size, num_layers, num_heads, model_dim).to(device)
state = torch.load(checkpoint_path, map_location=device)
state_dict = state["model"]

# Handle compiled checkpoints
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    print("Detected compiled checkpoint ?~@~T stripping '_orig_mod.' prefixes...")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
model.eval()
print("?~\~E Model loaded successfully!")
mean_weight = torch.mean(torch.stack([p.float().mean() for p in model.parameters()]))
print(f"Mean weight value for {checkpoint_path}: {mean_weight.item():.6f}")

for i, ex in enumerate(ds):
    ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
    endings = ex["endings"]
    gold = int(ex["label"])

    scores_sum = []
    scores_avg = []

    for cand in endings:
        s_sum, s_avg = score_continuation_vanilla(model, ctx, cand)
        scores_sum.append(s_sum)
        scores_avg.append(s_avg)

    pred = int(max(range(4), key=lambda j: scores_sum[j]))
    pred_norm = int(max(range(4), key=lambda j: scores_avg[j]))

    correct += (pred == gold)
    correct_norm += (pred_norm == gold)

    if (i + 1) % 100 == 0:
        print(f"{i+1}/{n} | acc={correct/(i+1):.3f} | acc_norm={correct_norm/(i+1):.3f}")

    if i + 1 == n:
        break

print("Final:")
print("Raw accuracy:", correct / n)
print("Length-normalized accuracy:", correct_norm / n)
