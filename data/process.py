import os
import numpy as np
from datasets import load_dataset
import tiktoken


def main():
    DATASET_NAME = "vietgpt/openwebtext_en"  
    MAX_TOKENS = 5_000_000_000
    OUT_DIR = "processed"
    OUT_BASENAME = "openwebtext_gpt2_5B"
    OUT_BIN = os.path.join(OUT_DIR, OUT_BASENAME + ".bin")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading dataset {DATASET_NAME} in streaming mode ...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    print("Loading tiktoken GPT-2 encoder ...")
    enc = tiktoken.get_encoding("gpt2")

    total_tokens = 0
    num_docs = 0
    dtype = np.uint16  # GPT-2 vocab < 65535

    with open(OUT_BIN, "wb") as f:
        for example in ds:
            text = example["text"]

            # 🔧 FIXED: use a set, or just drop allowed_special entirely
            ids = enc.encode(text, allowed_special=set())
            # ids = enc.encode(text)  # also fine

            if not ids:
                continue

            arr = np.asarray(ids, dtype=dtype)
            f.write(arr.tobytes())

            total_tokens += arr.size
            num_docs += 1

            if num_docs % 1000 == 0:
                print(f"Docs: {num_docs:,}, tokens so far: {total_tokens:,}")

            if total_tokens >= MAX_TOKENS:
                print(f"Reached MAX_TOKENS = {MAX_TOKENS:,}, stopping.")
                break

    print(f"\nFinished.")
    print(f"Docs processed: {num_docs:,}")
    print(f"Tokens written: {total_tokens:,}")
    print(f"Output file: {OUT_BIN}")
    print(f"Approx size on disk: {total_tokens * 2 / (1024**3):.2f} GB (uint16)")


if __name__ == "__main__":
    main()
