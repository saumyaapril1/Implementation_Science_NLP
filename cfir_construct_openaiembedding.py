#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI

# ---------- config ----------
CFIR_JSON = "cfir_lexicon.json"
OUT_PHRASE_CSV = "cfir_lexicon_openai_phrases.csv"
OUT_PROTO_CSV  = "cfir_lexicon_openai_prototypes.csv"
MODEL_NAME = "text-embedding-3-small"
# ----------------------------

def embed_openai_small(texts, batch_size=128, normalize=True):
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Please set OPENAI_API_KEY in your environment.")
    client = OpenAI()
    all_vecs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=MODEL_NAME, input=batch)
        vecs = [d.embedding for d in resp.data]
        all_vecs.extend(vecs)

    V = np.asarray(all_vecs, dtype="float32")
    if normalize:
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V

def main():
    # 1) Load CFIR lexicon
    with open(CFIR_JSON, "r") as f:
        cfir_lex = json.load(f)

    # 2) Flatten into (construct, phrase) lists
    constructs = []
    phrases = []
    for k, plist in cfir_lex.items():
        for p in plist:
            constructs.append(k)
            phrases.append(p)

    print(f"[info] Found {len(phrases)} CFIR phrases in the lexicon")

    # 3) Embed all phrases with OpenAI
    print("[info] Embedding CFIR phrases with OpenAI...")
    V = embed_openai_small(phrases)   # shape [N, 1536]
    print("[info] Embedding shape:", V.shape)

    # 4) Save phrase-level embeddings
    #    -> one row per phrase; columns: construct, phrase, e0, e1, ..., e1535
    dim = V.shape[1]
    cols = ["construct", "phrase"] + [f"e{i}" for i in range(dim)]
    data = []
    for c, p, v in zip(constructs, phrases, V):
        data.append([c, p] + v.tolist())
    df_phrases = pd.DataFrame(data, columns=cols)
    df_phrases.to_csv(OUT_PHRASE_CSV, index=False)
    print(f"[OK] Saved phrase-level CFIR embeddings → {OUT_PHRASE_CSV}")

    # 5) Build construct-level prototype vectors (mean of phrase embeddings per construct)
    proto_dict = {}
    for k in sorted(set(constructs)):
        idx = [i for i, c in enumerate(constructs) if c == k]
        vec = V[idx].mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        proto_dict[k] = vec

    # 6) Save prototypes
    proto_rows = []
    for k, v in proto_dict.items():
        proto_rows.append([k] + v.tolist())
    cols_proto = ["construct"] + [f"e{i}" for i in range(dim)]
    df_proto = pd.DataFrame(proto_rows, columns=cols_proto)
    df_proto.to_csv(OUT_PROTO_CSV, index=False)
    print(f"[OK] Saved construct-level CFIR prototypes → {OUT_PROTO_CSV}")

if __name__ == "__main__":
    main()
