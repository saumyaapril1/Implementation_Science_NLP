#!/usr/bin/env python3
import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity


# ---------------- OpenAI embedding ----------------
def embed_openai_small(texts, batch_size=128, normalize=True):
    """
    Embeds a list of texts using OpenAI text-embedding-3-small.
    Returns a [N, D] float32 numpy array.
    """
    from openai import OpenAI

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    client = OpenAI()
    all_vecs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        vecs = [d.embedding for d in resp.data]
        all_vecs.extend(vecs)

    V = np.asarray(all_vecs, dtype="float32")
    if normalize:
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V


# ---------------- CFIR prototypes ----------------
def build_prototypes(cfir_lex, embed_fn):
    """
    Build CFIR prototype vectors from lexicon using the same embedder
    that was used for sentences (OpenAI here).
    """
    keys, phrases = [], []
    for k, plist in cfir_lex.items():
        for p in plist:
            keys.append(k)
            phrases.append(p)

    V = embed_fn(phrases)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    protos = {}
    for k in sorted(set(keys)):
        idx = [i for i, kk in enumerate(keys) if kk == k]
        c = V[idx].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        protos[k] = c

    return protos


# ---------------- UMAP for visualization only ----------------
def fit_umap_2d(X, n_neighbors=30, min_dist=0.10, seed=42):
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(X)
    return reducer, coords


def plot_clusters_and_cfir_hdbscan(coords, labels, proto_labels, proto_coords, out_png):
    """
    coords: [N, 2] UMAP for sentences
    labels: [N] HDBSCAN labels (0..C-1, -1 for noise)
    proto_labels: list of CFIR labels
    proto_coords: [K', 2] UMAP coords for CFIR prototypes
    """
    unique_labels = sorted(list(set(labels)))
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    for lab in unique_labels:
        mask = (labels == lab)
        if lab == -1:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=10,
                alpha=0.5,
                c="lightgray",
                label="Noise",
            )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=12,
                alpha=0.8,
                label=f"Cluster {lab}",
            )

    # CFIR prototypes
    for lbl, (x, y) in zip(proto_labels, proto_coords):
        ax.scatter([x], [y], marker="x", s=250, linewidths=3, c="k")
        ax.text(x + 0.05, y + 0.05, lbl, fontsize=12, weight="bold", color="k")

    ax.set_title("Sentence clusters (HDBSCAN, OpenAI embeddings, UMAP visualization)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right", title="Clusters", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print("[OK] Saved cluster UMAP plot →", out_png)


# ---------------- MAIN PIPELINE ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="filtered_sentences.csv",
                        help="Input CSV file with sentences")
    parser.add_argument("--text_col", default="sentence",
                        help="Column name containing sentences")
    parser.add_argument("--cfir_lexicon_json", default="cfir_lexicon.json",
                        help="CFIR lexicon JSON file")

    parser.add_argument("--min_cluster_size", type=int, default=5,
                        help="HDBSCAN min_cluster_size (minimum cluster size)")
    parser.add_argument("--min_samples", type=int, default=3,
                        help="HDBSCAN min_samples (if None, defaults to min_cluster_size)")

    parser.add_argument("--umap_neighbors", type=int, default=15,
                        help="UMAP n_neighbors (visualization only)")
    parser.add_argument("--umap_min_dist", type=float, default=0.05,
                        help="UMAP min_dist (visualization only)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for UMAP")

    args = parser.parse_args()

    # derive suffix from input file, e.g. filtered_sentences.csv -> filtered_sentences
    input_stem = os.path.splitext(os.path.basename(args.input_csv))[0]

    # ---- 1) Load sentences ----
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found in {args.input_csv}")

    sentences = df[args.text_col].astype(str).tolist()
    print(f"[info] Loaded {len(sentences)} sentences from {args.input_csv}.")

    # ---- 2) Embed sentences with OpenAI (ORIGINAL SPACE) ----
    print("[info] Embedding sentences with OpenAI text-embedding-3-small...")
    X = embed_openai_small(sentences)  # shape [N, 1536]
    print("[info] Embedding shape:", X.shape)

    # ---- 3) HDBSCAN clustering in ORIGINAL space ----
    import hdbscan

    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples if args.min_samples is not None else min_cluster_size

    print(f"[info] Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)      # -1 = noise
    probs = clusterer.probabilities_

    df["cluster"] = labels
    df["cluster_prob"] = probs

    unique_labels = sorted(list(set(labels)))
    print("[info] Unique cluster labels (HDBSCAN):", unique_labels)

    # Save sentences per cluster and noise
    for lab in unique_labels:
        mask = (labels == lab)
        subset = df[mask][args.text_col].tolist()
        if lab == -1:
            fname = f"cluster_noise_sentences_{input_stem}.txt"
        else:
            fname = f"cluster_{lab}_sentences_{input_stem}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            for s in subset:
                f.write(s + "\n")
        print(f"[OK] Saved {len(subset)} sentences to {fname}")

    # ---- 4) CFIR prototypes (OpenAI embedding) ----
    with open(args.cfir_lexicon_json, "r") as f:
        cfir_lex = json.load(f)

    print("[info] Embedding CFIR lexicon with OpenAI...")
    protos = build_prototypes(cfir_lex, embed_openai_small)
    proto_labels = sorted(list(protos.keys()))
    proto_matrix = np.stack([protos[p] for p in proto_labels])  # [num_cfirs, 1536]

    # ---- 5) Cluster-centroid → CFIR similarity matrix (ORIGINAL SPACE) ----
    cluster_centroids = []
    cluster_names = []
    for lab in unique_labels:
        if lab == -1:
            continue  # skip noise
        mask = (labels == lab)
        if np.sum(mask) == 0:
            continue
        c = X[mask].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        cluster_centroids.append(c)
        cluster_names.append(f"Cluster_{lab}")

    if cluster_centroids:
        cluster_centroids = np.stack(cluster_centroids, axis=0)  # [C, 1536]
        sim_matrix = cosine_similarity(cluster_centroids, proto_matrix)
        sim_df = pd.DataFrame(
            sim_matrix,
            columns=proto_labels,
            index=cluster_names,
        )
        sim_csv = f"cluster_cfir_similarity_openai_hdbscan_{input_stem}.csv"
        sim_df.to_csv(sim_csv)
        print(f"[OK] Saved cluster → CFIR similarity matrix → {sim_csv}")

        plt.figure(figsize=(8, 5))
        plt.imshow(sim_matrix, cmap="viridis")
        plt.colorbar(label="cosine similarity")
        plt.xticks(range(len(proto_labels)), proto_labels, rotation=45)
        plt.yticks(range(len(cluster_names)), cluster_names)
        plt.title(f"Cluster Centroid → CFIR Prototype Similarity (OpenAI + HDBSCAN, {input_stem})")
        plt.tight_layout()
        heatmap_png = f"cluster_cfir_similarity_heatmap_openai_hdbscan_{input_stem}.png"
        plt.savefig(heatmap_png, dpi=150)
        plt.close()
        print(f"[OK] Saved heatmap → {heatmap_png}")
    else:
        print("[warn] No non-noise clusters found by HDBSCAN with these hyperparameters.")

    # ---- 6) UMAP visualization (2-D) with CFIR constructs overlaid ----
    print("[info] Fitting UMAP on sentence embeddings (for visualization only)...")
    reducer, coords = fit_umap_2d(
        X,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    proto_coords = reducer.transform(proto_matrix)

    umap_png = f"cluster_umap_openai_hdbscan_with_cfir_{input_stem}.png"
    plot_clusters_and_cfir_hdbscan(
        coords,
        labels,
        proto_labels,
        proto_coords,
        out_png=umap_png,
    )

    # ---- 7) Save master CSV ----
    out_csv = f"sentences_with_clusters_openai_hdbscan_{input_stem}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")

# Counts per cluster (including noise = -1)
    print(df["cluster"].value_counts())



if __name__ == "__main__":
    main()
