# eval_subreddit_benchmark.py
import json
import random
import torch
import tiktoken
from tqdm import tqdm
from train_gpt_with_inference import GPT
import pandas as pd

device = "cuda"
vocab_size = 50304
num_layers = 12
num_heads = 6
model_dim = 768

BLOCK_SIZE = 128
PAD_TOKEN  = 50256  # <|endoftext|> as pad

REDDIT_SUBS = [
    "programming", "science", "bestof", "Games",
    "Fitness", "DIY", "personalfinance", "philosophy", "history",'fashion','boardgames'
]

def build_encoder(with_ins_ctx: bool):
    base_enc = tiktoken.get_encoding("gpt2")
    if not with_ins_ctx:
        return base_enc

    custom_specials = {"<ins>": 50257, "<ctx>": 50258}
    enc = tiktoken.Encoding(
        name="gpt2-with-ins-ctx",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens={**base_enc._special_tokens, **custom_specials},
    )
    return enc

@torch.no_grad()
def score_continuation_like_hellaswag(model, enc, prompt: str, continuation: str, sliding_window_num_blocks):
    """
    Returns (sum_logprob, avg_logprob) of continuation tokens ONLY,
    with exact left-padding behavior like your HellaSwag scorer.
    """
    model.eval()

    # allow <|endoftext|> plus optionally <ins>/<ctx>
    allowed = {"<|endoftext|>"}
    if "<ins>" in enc._special_tokens:
        allowed |= {"<ins>", "<ctx>"}

    encode = lambda s: enc.encode(s, allowed_special=allowed)

    prompt_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    cont_ids   = torch.tensor(encode(" " + continuation), dtype=torch.int32, device=device)

    input_ids = torch.cat([prompt_ids, cont_ids])  # [T]

    pad_len = (-len(input_ids)) % BLOCK_SIZE
    if pad_len > 0:
        pad = torch.full((pad_len,), PAD_TOKEN, dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([pad, input_ids])

    # Model.inference must return (logits [1,T,V] or [T,V], gates [T])
    logits, gates = model.inference(input_ids, sw_blocks)

    if logits.ndim == 3: logits = logits[0]  # [T,V]
    # gates expected shape [T]
    assert gates.ndim == 1, f"Expected gates [T], got {gates.shape}"

    # Next-token logprobs
    VOCAB_SIZE = getattr(model.lm_head, "out_features", vocab_size)
    logits   = logits[:-1, :vocab_size]          # [T-1,V]
    logprobs = torch.log_softmax(logits, dim=-1) # [T-1,V]
    target   = input_ids[1:]                     # [T-1]
    gates_t  = gates[:-1]                        # [T-1]

    # Continuation slice starts at the boundary between prompt and continuation
    cont_start = len(prompt_ids)
    cont_start_idx = pad_len + cont_start

    lp_slice  = logprobs[cont_start_idx - 1 :, :]            # [C,V]
    tgt_slice = target  [cont_start_idx - 1 :]               # [C]
    g_slice   = gates_t [cont_start_idx - 1 :]               # [C]

    # Gather true-token logprobs
    token_lps = lp_slice.gather(1, tgt_slice.unsqueeze(-1)).squeeze(-1)  # [C]

    sum_lp = token_lps.sum().item()
    avg_lp = sum_lp / max(1, tgt_slice.numel())

    # Gated score (no lambda)
    sum_lp_gated = (g_slice * token_lps).sum().item()
    avg_lp_gated = sum_lp_gated / max(1, tgt_slice.numel())

    mean_gate = g_slice.mean().item() if g_slice.numel() else 1.0
    min_gate  = g_slice.min().item()  if g_slice.numel() else 1.0

    return sum_lp,avg_lp, {
        "sum_lp": sum_lp,
        "avg_lp": avg_lp,
        "sum_lp_gated": sum_lp_gated,
        "avg_lp_gated": avg_lp_gated,
        "mean_g": mean_gate,
        "min_g": min_gate,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_json", type=str, default="reddit_subreddit_benchmark.json")
    checkpoint_def = ''
    parser.add_argument("--ckpt", type=str,default = checkpoint_def)
    parser.add_argument("--with_ins_ctx", action="store_true")
    parser.add_argument("--n", type=int, default=None)  # optional subset
    args = parser.parse_args()

    enc = build_encoder(args.with_ins_ctx)
    sliding_window_num_blocks = torch.tensor(1, device="cuda")

    # load model
    model = GPT(vocab_size, num_layers, num_heads, model_dim).to(device)
    state = torch.load(args.ckpt, map_location=device)
    state_dict = state["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with open(args.benchmark_json) as f:
        data = json.load(f)
    if args.n is not None:
        data = data[:args.n]

    correct_sum = 0
    correct_avg = 0

    rows = []
    for i, ex in enumerate(tqdm(data)):
        instr = ex["instruction"]
        post  = ex["post"]
        label = ex["label"]

        prompt = f"<|endoftext|><ins>{instr}<ins><ctx>{post}<ctx>"

        scores_sum = []
        scores_avg = []
        for sub in REDDIT_SUBS:
            s_sum, s_avg , s = score_continuation_like_hellaswag(
                model, enc, prompt, f"r/{sub}", sliding_window_num_blocks
            )
            scores_sum.append(s_sum)
            scores_avg.append(s_avg)
            rows.append({
                "example_id": i,
                "ctx": ctx,
                "option_idx": j,
                "option_text": cand,
                "is_gt": int(j == gold),
                "sum_lp": s["sum_lp"],
                "avg_lp": s["avg_lp"],
                "sum_lp_gated": s["sum_lp_gated"],
                "avg_lp_gated": s["avg_lp_gated"],
                "mean_g": s["mean_g"],
                "min_g": s["min_g"],
            })



        pred_sum = REDDIT_SUBS[int(max(range(len(REDDIT_SUBS)), key=lambda j: scores_sum[j]))]
        pred_avg = REDDIT_SUBS[int(max(range(len(REDDIT_SUBS)), key=lambda j: scores_avg[j]))]

        correct_sum += int(pred_sum == label)
        correct_avg += int(pred_avg == label)

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(data)} | acc_sum={correct_sum/(i+1):.3f} | acc_avg={correct_avg/(i+1):.3f}")

    df = pd.DataFrame(rows)
    OUT_CSV = checkpoint_def + '_reddit_eval.csv'
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")
    print("Columns:", list(df.columns))
    print("Final:")
    print("Raw accuracy:", correct_sum / len(data))
    print("Length-normalized accuracy:", correct_avg / len(data))

if __name__ == "__main__":
    main()
