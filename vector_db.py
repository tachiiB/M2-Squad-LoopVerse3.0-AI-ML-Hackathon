"""Sprint 2 Vector Database: dense vector storage + similarity search for RAG.

Pipeline:
  1. INGEST  - 13 docs -> 58 chunks (docstore.py: metadata preserved)
  2. VECTORIZE - chunks -> TF-IDF sparse vectors -> SVD -> 64-dim DENSE vectors
  3. STORE    - dense vectors in a numpy matrix (the vector database)
  4. SEARCH   - query -> dense vector -> cosine similarity -> top-k
  5. RANK     - similarity x currency x authority (current > superseded)

The dense vectors come from TruncatedSVD (Latent Semantic Analysis) applied
to TF-IDF: a classic dense retrieval method. No torch/transformers needed,
runs everywhere, deterministic.

Security: document text is UNTRUSTED DATA (indexed/quoted as evidence,
never executed). No LLM in generation (see ask.py) -> injection-proof.
"""

import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from docstore import chunk_doc, parse_doc


class VectorDB:
    """Dense vector database for RAG retrieval."""

    def __init__(self, doc_paths, n_components=64):
        # 1. ingest + chunk
        self.chunks = []
        for p in sorted(doc_paths):
            self.chunks.extend(chunk_doc(parse_doc(p)))
        texts = [c["text"] for c in self.chunks]

        # 2. vectorize: TF-IDF -> SVD -> dense vectors
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        sparse = self._tfidf.fit_transform(texts)
        n_comp = min(n_components, sparse.shape[0] - 1, sparse.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        dense = self._svd.fit_transform(sparse)
        # 3. store: L2-normalized dense vectors (cosine = dot product)
        self.vectors = normalize(dense.astype(np.float32))
        print(f"VectorDB: {len(self.chunks)} chunks -> "
              f"{self.vectors.shape[1]}-dim dense vectors stored")

    def _boost(self, chunk):
        b = 1.0
        if chunk["status"] == "current":
            b *= 1.3
        elif chunk["status"] == "superseded":
            b *= 0.25
        auth = chunk["authority"].lower()
        if any(k in auth for k in ["environmental protection", "school education"]):
            b *= 1.2
        if "unofficial" in auth:
            b *= 0.7
        return b

    def _embed_query(self, query):
        q_sparse = self._tfidf.transform([query])
        q_dense = self._svd.transform(q_sparse).astype(np.float32)
        n = np.linalg.norm(q_dense)
        return q_dense[0] / n if n > 0 else q_dense[0]

    def search(self, query, top_k=3):
        """Vector search: cosine similarity between query vector and all
        stored document vectors, re-ranked by currency x authority."""
        q = self._embed_query(query)
        sims = self.vectors @ q
        ranked = sorted(
            ((float(sims[i]) * self._boost(c), c)
             for i, c in enumerate(self.chunks) if sims[i] > 0),
            key=lambda x: -x[0],
        )
        return [c for _, c in ranked[:top_k]]


if __name__ == "__main__":
    vdb = VectorDB(glob.glob("DOC-*.md"))
    for q in ["school closure smog policy", "mask N95 PM2.5 health",
              "old school regulations 2023"]:
        print(f"\nQ: {q}")
        for c in vdb.search(q, top_k=3):
            print(f"  [{c['doc_id']}] ({c['status']}) {c['text'][:90]}...")
