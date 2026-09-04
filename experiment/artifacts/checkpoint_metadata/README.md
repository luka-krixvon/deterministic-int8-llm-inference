# Checkpoint metadata

One directory per quantized checkpoint, holding the non-weight files that sat
beside `model.safetensors` on the measurement machine: `config.json` (with the
`quantization_config` block the engine reads), `recipe.yaml` (the llm-compressor
modifier request), `generation_config.json`, `tokenizer_config.json` and
`chat_template.jinja`.

`tokenizer.json` is stored once, at the top of this directory, because all five
checkpoints carried a byte-identical copy (11,422,650 bytes,
sha256 `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506`
— recomputed per checkpoint in `../checkpoint_digests_2026-08-14.json`, where the
`metadata_file_sha256` map records every file in every directory including this
one). Five identical copies would add no evidence the digest map does not already
carry.

The weights themselves are not here. Their identity is the `checkpoint_digest`
in `../checkpoint_digests_2026-08-14.json`, computed over the concatenated bytes
of the sorted `*.safetensors` shards, the same scope the harness uses
(`../../harness/tf_v6.py`).
