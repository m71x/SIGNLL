import easydel as ed

elm = (
    ed.eLargeModel.from_pretrained("Qwen/Qwen2.5-Coder-34B-Instruct")
    .set_dtype("bf16")
    .set_sharding(axis_dims=(1, 1, 1, -1, 1))
    .set_esurge(max_model_len=4096, max_num_seqs=32)
)

# Build and use eSurge engine
esurge = elm.build_esurge()

for output in esurge.chat(
    [{"role": "user", "content": "Write a recursive fibonacci sequence implementation in python using memoization"}],
    sampling_params=ed.SamplingParams(max_tokens=512),
    stream=True,
):
    print(output.delta_text, end="", flush=True)

print(f"\nTokens/s: {output.tokens_per_second:.2f}")