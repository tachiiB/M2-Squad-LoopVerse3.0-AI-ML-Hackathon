"""Sprint 2 document store: ingest, chunk, TF-IDF retrieval.

Security posture: document text is UNTRUSTED DATA. It is chunked, indexed,
and quoted as evidence -- it is never executed, never treated as
instructions, and the generator (ask.py) contains no LLM, so embedded
instructions cannot alter behavior by construction.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_doc(path):
    """Parse YAML frontmatter + markdown body. Returns dict with metadata."""
    text = open(path, encoding="utf-8").read()
    meta, body = {}, text
    # frontmatter may not be at the very start (e.g. leading note); find first --- block
    m = re.search(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL | re.MULTILINE)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        # body = text before frontmatter + text after (drop the frontmatter itself)
        body = (text[:m.start()] + "\n" + m.group(2)).strip()
    # strip HTML comments (untrusted directives live here; never indexed as text)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    meta["text"] = body.strip()
    meta["doc_id"] = meta.get("document_id", "UNKNOWN")
    return meta


def chunk_doc(meta):
    """Split body into coherent passages; each chunk keeps full metadata."""
    body = meta["text"]
    # split on headings / blank lines into passages
    parts = [p.strip() for p in re.split(r"\n#{1,3} |\n\n+", body) if p.strip()]
    chunks = []
    for i, p in enumerate(parts):
        if len(p) < 20:
            continue
        chunks.append({
            "doc_id": meta["doc_id"],
            "title": meta.get("title", ""),
            "authority": meta.get("authority", ""),
            "published_date": meta.get("published_date", ""),
            "status": meta.get("status", ""),
            "chunk_id": f"{meta['doc_id']}#c{i}",
            "text": re.sub(r"\s+", " ", p),
        })
    return chunks


class DocStore:
    def __init__(self, paths):
        self.chunks = []
        for p in paths:
            self.chunks.extend(chunk_doc(parse_doc(p)))
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._mat = self._vec.fit_transform([c["text"] for c in self.chunks])

    def _currency_boost(self, chunk, query):
        """Prefer current, authoritative guidance; bury superseded docs."""
        boost = 1.0
        if chunk["status"] == "current":
            boost *= 1.3
        elif chunk["status"] == "superseded":
            boost *= 0.25
        # official departments outrank unofficial bulletins for policy/health
        auth = chunk["authority"].lower()
        if any(k in auth for k in ["environmental protection", "school education"]):
            boost *= 1.2
        if "unofficial" in auth:
            boost *= 0.7
        return boost

    def retrieve(self, query, top_k=3):
        """Keyword retrieval ranked by relevance x currency x authority."""
        q = self._vec.transform([query])
        scores = (self._mat @ q.T).toarray().ravel()
        ranked = []
        for i, c in enumerate(self.chunks):
            if scores[i] <= 0:
                continue
            ranked.append((scores[i] * self._currency_boost(c, query), c))
        ranked.sort(key=lambda x: -x[0])
        return [c for _, c in ranked[:top_k]]


if __name__ == "__main__":
    import glob
    ds = DocStore(sorted(glob.glob("/home/hatch/workspace/user/files/DOC-*.md")))
    print(f"{len(ds.chunks)} chunks from 13 docs")
    for q in ["school closure smog policy", "mask N95 PM2.5 health",
              "old school regulations 2023"]:
        print(f"\nQ: {q}")
        for c in ds.retrieve(q, top_k=3):
            print(f"  [{c['doc_id']}] ({c['status']}) {c['text'][:100]}...")
