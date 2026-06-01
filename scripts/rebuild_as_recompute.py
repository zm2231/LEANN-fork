#!/usr/bin/env python3
"""Rebuild an existing --no-recompute LEANN index as --recompute, reusing the
stored embeddings (no re-call of the embedding endpoint)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-name", default="documents.leann")
    args = ap.parse_args()

    src = Path(args.source_index)
    src_meta = json.loads(Path(str(src) + ".meta.json").read_text())
    print(f"Source: model={src_meta['embedding_model']}, dim={src_meta['dimensions']}, "
          f"recompute={src_meta['backend_kwargs']['is_recompute']}", file=sys.stderr)
    assert not src_meta["backend_kwargs"]["is_recompute"], "Source must be --no-recompute"

    from leann_backend_hnsw import faiss
    # FAISS index file is named documents.index, not documents.leann.index
    candidate_a = str(src) + ".index"
    candidate_b = str(src.parent / src.stem) + ".index"  # documents.index
    src_index_file = candidate_a if Path(candidate_a).exists() else candidate_b
    print(f"Loading FAISS index from {src_index_file}...", file=sys.stderr)
    index = faiss.read_index(src_index_file)
    ntotal = index.ntotal
    dim = src_meta["dimensions"]
    print(f"Reconstructing {ntotal} vectors at dim={dim}...", file=sys.stderr)
    vectors = np.empty((ntotal, dim), dtype=np.float32)
    for i in range(ntotal):
        row = np.empty(dim, dtype=np.float32)
        index.reconstruct(i, faiss.swig_ptr(row))
        vectors[i] = row
        if i % 2000 == 0:
            print(f"  {i}/{ntotal}", file=sys.stderr)

    ids_path = src.parent / "documents.ids.txt"
    if ids_path.exists():
        passage_ids = [s for s in ids_path.read_text().splitlines() if s]
        assert len(passage_ids) == ntotal, f"ids count {len(passage_ids)} != {ntotal}"
    else:
        passage_ids = [str(i) for i in range(ntotal)]

    passages_path = Path(str(src) + ".passages.jsonl")
    passages_by_id = {}
    with passages_path.open() as f:
        for line in f:
            p = json.loads(line)
            passages_by_id[p["id"]] = p

    from leann.api import LeannBuilder
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_index = str(out_dir / args.out_name)
    print(f"Building --recompute index at {out_index}...", file=sys.stderr)

    builder = LeannBuilder(
        backend_name=src_meta["backend_name"],
        embedding_model=src_meta["embedding_model"],
        embedding_mode=src_meta.get("embedding_mode", "openai"),
        embedding_options=src_meta.get("embedding_options"),
        dimensions=dim,
        is_recompute=True,
        is_compact=False,
    )

    for pid in passage_ids:
        p = passages_by_id.get(pid)
        if p is None:
            print(f"  WARN: no passage for id {pid}", file=sys.stderr)
            continue
        builder.add_text(p["text"], metadata=p.get("metadata", {}))

    builder.build_index_from_arrays(out_index, passage_ids, vectors)
    print(f"DONE: {out_index}", file=sys.stderr)


if __name__ == "__main__":
    main()
