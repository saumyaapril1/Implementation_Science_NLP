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
    Returns dict: {construct_name: prototype_vector}
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


# ---------------- UMAP helpers ----------------
def fit_umap_nd(X, n_neighbors=30, min_dist=0.10, n_components=15, seed=42):
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(X)
    return reducer, coords


def plot_clusters_and_cfir_hdbscan_strict(coords, labels, proto_labels, proto_coords, out_png):
    """
    STRICT plot: uses raw HDBSCAN labels; noise (-1) shown in gray.
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

    ax.set_title("STRICT clusters (HDBSCAN on UMAP-15D) + CFIR prototypes")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right", title="Clusters", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print("[OK] Saved STRICT cluster UMAP plot →", out_png)


def plot_clusters_and_cfir_hdbscan_filled(coords, filled_labels, proto_labels, proto_coords, out_png):
    """
    FILLED plot: uses cluster_filled; no noise color (all assigned).
    """
    unique_labels = sorted(list(set(filled_labels)))
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    for lab in unique_labels:
        mask = (filled_labels == lab)
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

    ax.set_title("FILLED clusters (noise reassigned) + CFIR prototypes")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right", title="Clusters", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print("[OK] Saved FILLED cluster UMAP plot →", out_png)


# ---------------- MAIN PIPELINE ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="interview_sentences.csv",
                        help="Input CSV file with sentences")
    parser.add_argument("--text_col", default="sentence",
                        help="Column name containing sentences")
    parser.add_argument("--cfir_lexicon_json", default="cfir_lexicon.json",
                        help="CFIR lexicon JSON file")

    # HDBSCAN params
    parser.add_argument("--min_cluster_size", type=int, default=15,
                        help="HDBSCAN min_cluster_size (minimum cluster size)")
    parser.add_argument("--min_samples", type=int, default=None,
                        help="HDBSCAN min_samples (if None, defaults to min_cluster_size)")

    # UMAP params (used for both clustering-UMAP and viz-UMAP; only n_components differs)
    parser.add_argument("--umap_neighbors", type=int, default=30,
                        help="UMAP n_neighbors")
    parser.add_argument("--umap_min_dist", type=float, default=0.10,
                        help="UMAP min_dist")
    parser.add_argument("--cluster_umap_dim", type=int, default=15,
                        help="UMAP dimensionality for clustering (e.g. 15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Derive suffix from input file, e.g. filtered_sentences.csv -> filtered_sentences
    input_stem = os.path.splitext(os.path.basename(args.input_csv))[0]

    # ---- 1) Load sentences ----
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found in {args.input_csv}")

    sentences = df[args.text_col].astype(str).tolist()
    print(f"[info] Loaded {len(sentences)} sentences from {args.input_csv}.")

    # ---- 2) OpenAI embeddings in ORIGINAL space (1536D) ----
    print("[info] Embedding sentences with OpenAI text-embedding-3-small...")
    X_full = embed_openai_small(sentences)  # [N, 1536]
    print("[info] X_full shape (original):", X_full.shape)

    # ---- 3) UMAP to 15D (or cluster_umap_dim) for clustering ----
    print(f"[info] Fitting UMAP to {args.cluster_umap_dim}D for clustering...")
    reducer_cluster, X_cluster = fit_umap_nd(
        X_full,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.cluster_umap_dim,
        seed=args.seed,
    )
    print("[info] X_cluster shape (UMAP for clustering):", X_cluster.shape)

    # ---- 4) HDBSCAN clustering in UMAP space ----
    import hdbscan

    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples if args.min_samples is not None else min_cluster_size

    print(f"[info] Running HDBSCAN on UMAP-{args.cluster_umap_dim}D "
          f"(min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # UMAP space; euclidean is standard here
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X_cluster)      # -1 = noise
    probs = clusterer.probabilities_              # membership strength

    df["cluster"] = labels
    df["cluster_prob"] = probs

    unique_labels = sorted(list(set(labels)))
    print("[info] Unique STRICT cluster labels (HDBSCAN):", unique_labels)

    # ---- 5) Compute centroids in ORIGINAL 1536D space (STRICT clusters only) ----
    cluster_centroids = []
    cluster_names = []
    for lab in unique_labels:
        if lab == -1:
            continue  # skip noise
        mask = (labels == lab)
        if np.sum(mask) == 0:
            continue
        c = X_full[mask].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        cluster_centroids.append(c)
        cluster_names.append(f"Cluster_{lab}")

    # ---- 6) Option B: reassign noise to nearest centroid (in ORIGINAL space) ----
    filled_labels = labels.copy()
    if cluster_centroids:
        cluster_centroids_arr = np.stack(cluster_centroids, axis=0)  # [C, 1536]

        # Cluster IDs as actual integers, e.g. [0, 1, 3]
        cluster_ids = [int(name.split("_")[1]) for name in cluster_names]

        noise_mask = (labels == -1)
        if np.any(noise_mask):
            sims = cosine_similarity(X_full[noise_mask], cluster_centroids_arr)  # [N_noise, C]
            best_idx = np.argmax(sims, axis=1)  # closest centroid index
            filled_labels[noise_mask] = np.array(cluster_ids)[best_idx]
            print(f"[info] Re-assigned {noise_mask.sum()} noise points to nearest clusters.")
    else:
        print("[warn] No non-noise clusters; cannot fill noise.")

    df["cluster_filled"] = filled_labels

    # ---- 7) Save sentences per STRICT cluster and noise ----
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

    # ---- 8) CFIR prototypes in ORIGINAL space ----
    with open(args.cfir_lexicon_json, "r") as f:
        cfir_lex = json.load(f)

    print("[info] Embedding CFIR lexicon with OpenAI...")
    protos = build_prototypes(cfir_lex, embed_openai_small)
    proto_labels = sorted(list(protos.keys()))
    proto_matrix = np.stack([protos[p] for p in proto_labels])  # [num_cfirs, 1536]

    # ---- 9) Cluster-centroid → CFIR similarity matrix (STRICT clusters, ORIGINAL space) ----
    if cluster_centroids:
        cluster_centroids_arr = np.stack(cluster_centroids, axis=0)  # [C, 1536]
        sim_matrix = cosine_similarity(cluster_centroids_arr, proto_matrix)
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
        plt.title(f"Cluster Centroid → CFIR Prototype Similarity "
                  f"(OpenAI original space + HDBSCAN on UMAP-{args.cluster_umap_dim}D, {input_stem})")
        plt.tight_layout()
        heatmap_png = f"cluster_cfir_similarity_heatmap_openai_hdbscan_{input_stem}.png"
        plt.savefig(heatmap_png, dpi=150)
        plt.close()
        print(f"[OK] Saved heatmap → {heatmap_png}")
    else:
        print("[warn] No non-noise clusters found by HDBSCAN with these hyperparameters.")

    # ---- 9.5) Save counts per cluster (STRICT + FILLED) ----
    strict_counts = df["cluster"].value_counts().sort_index()
    strict_counts_csv = f"cluster_counts_STRICT_{input_stem}.csv"
    strict_counts.to_csv(strict_counts_csv, header=["num_sentences"])
    print(f"[OK] Saved strict cluster counts → {strict_counts_csv}")
    print(strict_counts)

    filled_counts = df["cluster_filled"].value_counts().sort_index()
    filled_counts_csv = f"cluster_counts_FILLED_{input_stem}.csv"
    filled_counts.to_csv(filled_counts_csv, header=["num_sentences"])
    print(f"[OK] Saved filled cluster counts → {filled_counts_csv}")
    print(filled_counts)

    # ---- 10) UMAP to 2D for visualization ONLY (still from ORIGINAL space) ----
    print("[info] Fitting UMAP to 2D for visualization (original 1536D → 2D)...")
    reducer_2d, coords_2d = fit_umap_nd(
        X_full,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        seed=args.seed,
    )

    # Project CFIR prototypes to same 2D UMAP space
    proto_coords_2d = reducer_2d.transform(proto_matrix)

    # STRICT plot
    umap_strict_png = f"cluster_umap_openai_hdbscan_with_cfir_STRICT_{input_stem}.png"
    plot_clusters_and_cfir_hdbscan_strict(
        coords_2d,
        labels,
        proto_labels,
        proto_coords_2d,
        out_png=umap_strict_png,
    )

    # FILLED plot
    umap_filled_png = f"cluster_umap_openai_hdbscan_with_cfir_FILLED_{input_stem}.png"
    plot_clusters_and_cfir_hdbscan_filled(
        coords_2d,
        filled_labels,
        proto_labels,
        proto_coords_2d,
        out_png=umap_filled_png,
    )

    # ---- 11) Save master CSV ----
    out_csv = f"sentences_with_clusters_openai_hdbscan_{input_stem}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")


if __name__ == "__main__":
    main()
