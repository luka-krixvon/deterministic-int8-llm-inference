import json, hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                  revision="b08601e04326c79dfdd32d625aee71d232d685c3")
text = "".join(ds["text"])
tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok(text, add_special_tokens=False)["input_ids"]
prompts = []
i = 0
while len(prompts) < 64 and i + 320 <= len(ids):
    prompts.append(tok.decode(ids[i:i+320]))
    i += 320
json.dump(prompts, open("calib_prompts.json","w"))
print("prompts:", len(prompts), "| sha:", hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16])
