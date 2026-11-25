"""Qwen3.8-Flash-Next smoke run on the backported vLLM 0.23 (one engine, TP 4).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python qwen38_smoke.py <model dir> [dummy] [eager]

dummy: random weights (start-up and shapes only); eager: no CUDA graphs.
Prints the KV cache layout, then a greedy completion of a few prompts twice:
the second pass hits the prefix cache.
"""

import sys

from vllm import LLM, SamplingParams


def main() -> None:
    model = sys.argv[1]
    flags = set(sys.argv[2:])
    llm = LLM(
        model=model,
        tensor_parallel_size=4,
        load_format="dummy" if "dummy" in flags else "auto",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        block_size=1024,
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        language_model_only=True,
        enforce_eager="eager" in flags,
        compilation_config={"cudagraph_mode": "NONE"} if "nocg" in flags else None,
        async_scheduling=False if "noasync" in flags else None,
    )
    prompts = [
        "The capital of France is",
        "Write a Python function that returns the n-th Fibonacci number.\n",
        "Q: What is 17 * 23?\nA:",
    ]
    params = SamplingParams(max_tokens=48, temperature=0)
    for attempt in range(2):
        outs = llm.generate(prompts, params)
        for out in outs:
            print(f"[{attempt}] {out.prompt!r} -> {out.outputs[0].text!r}")

    # Decode reads the n-gram context step by step; a prefill over prompt +
    # output must pick the same greedy token at every generated position.
    forced = [
        {"prompt_token_ids": list(o.prompt_token_ids) + list(o.outputs[0].token_ids)}
        for o in outs
    ]
    check = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    for o, f in zip(outs, llm.generate(forced, check)):
        n = len(o.prompt_token_ids)
        gen = o.outputs[0].token_ids
        gaps = []
        for i, tok in enumerate(gen):
            lp = f.prompt_logprobs[n + i]
            if lp[tok].rank != 1:
                top = max(v.logprob for v in lp.values())
                gaps.append((i, round(top - lp[tok].logprob, 4)))
        print(f"[forced] {o.prompt[:24]!r}: {len(gaps)}/{len(gen)} decode "
              f"tokens not greedy under prefill, (position, logprob gap) {gaps}")

    # Needle far back in a long context: QSA selects 2048 of the earlier
    # tokens, and the second pass reuses the cached prefix (block 1024).
    filler = " ".join(
        f"Line {i}: the weather report says it is cloudy with light wind."
        for i in range(400)
    )
    long_prompt = (
        "Remember this: the secret code is 7349.\n" + filler +
        "\nQuestion: what is the secret code mentioned at the very beginning? "
        "Answer with the number only.\nAnswer:"
    )
    tokens = len(llm.get_tokenizer().encode(long_prompt))
    long_params = SamplingParams(max_tokens=64, temperature=0)
    for attempt in range(3):
        out = llm.generate([long_prompt], long_params)[0]
        print(f"[long {attempt}] {tokens} tokens -> {out.outputs[0].text!r}")


if __name__ == "__main__":
    main()
