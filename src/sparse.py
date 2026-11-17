"""
Fast sparse BM25 + compound-aware analyzers (Step 5: M1, M2, M3).
Put in  src/sparse.py  (next to utils.py, compound.py).

SparseBM25 reproduces rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25) but scores with a scipy
sparse matrix, so it stays fast even when every token is a character n-gram.
"""
import re
from array import array
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

WORD = re.compile(r"\w+")


def tokenize(text):                      # same as utils.tokenize
    return WORD.findall(text.lower())


# ------------------------------------------------------------------ analyzers
def decompound_tokens(text, splitter):
    """M1: every word + (if it is a compound) its parts, e.g. etanolbilarna -> etanolbilarna etanol bilarna"""
    out = []
    for t in tokenize(text):
        out.append(t)
        if t.isalpha():
            parts = splitter.split(t)
            if len(parts) > 1:
                out.extend(parts)
    return out


def char_ngrams(text, ns=(3, 4, 5)):
    """M2: character n-grams inside each word, with boundary markers: 'bil' -> '<bi','bil','il>', ..."""
    out = []
    for t in tokenize(text):
        p = f"<{t}>"
        for n in ns:
            if len(p) < n:
                continue
            out.extend(p[i:i + n] for i in range(len(p) - n + 1))
    return out


def expand_query_inline(text, splitter):
    """M3: 'Slutet nära för etanolbilarna' -> 'Slutet nära för etanolbilarna (etanol bilarna)'"""
    def repl(m):
        w = m.group(0)
        if w.isalpha():
            parts = splitter.split(w)
            if len(parts) > 1:
                return f"{w} ({' '.join(parts)})"
        return w
    return WORD.sub(repl, text)


# ------------------------------------------------------------------ BM25
class SparseBM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75, epsilon=0.25):
        vocab, rows, cols, vals = {}, array("i"), array("i"), array("f")
        dl = np.zeros(len(docs_tokens), dtype=np.float64)
        for i, toks in enumerate(docs_tokens):
            dl[i] = len(toks)
            for t, f in Counter(toks).items():
                j = vocab.get(t)
                if j is None:
                    j = vocab[t] = len(vocab)
                rows.append(i); cols.append(j); vals.append(f)
        N, V = len(docs_tokens), len(vocab)
        rows = np.frombuffer(rows, dtype=np.int32); cols = np.frombuffer(cols, dtype=np.int32)
        tf = np.frombuffer(vals, dtype=np.float32).astype(np.float64)
        df = np.bincount(cols, minlength=V).astype(np.float64)
        idf = np.log((N - df + 0.5) / (df + 0.5))
        idf[idf < 0] = epsilon * idf.mean()                 # rank_bm25 convention
        avgdl = dl.mean()
        w = idf[cols] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / avgdl))
        self.vocab, self.N = vocab, N
        self.W = csr_matrix((w.astype(np.float32), (rows, cols)), shape=(N, V)).tocsc()

    def get_scores(self, q_tokens):
        c = Counter(t for t in q_tokens if t in self.vocab)
        if not c:
            return np.zeros(self.N, dtype=np.float32)
        ids = [self.vocab[t] for t in c]
        cnt = np.array([c[t] for t in c], dtype=np.float32)
        return np.asarray(self.W[:, ids] @ cnt).ravel()

    def top_ids(self, q_tokens, ids, k=100):
        s = self.get_scores(q_tokens)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return [ids[i] for i in top]


# ------------------------------------------------------------------ precision filter for a noisy splitter
class FilteredSplitter:
    """Wraps a splitter (e.g. CompoundPiece). A split is accepted only if EVERY part is a real,
    reasonably frequent word (len >= min_part_len and corpus frequency >= min_part_freq).
    Removes nonsense parts such as 'rna', 'len', 'ast'. Same interface as the other splitters."""
    def __init__(self, base, freq, min_part_len=3, min_part_freq=1000):
        self.base, self.freq = base, freq
        self.min_part_len, self.min_part_freq = min_part_len, min_part_freq

    def split(self, word):
        parts = self.base.split(word)
        if len(parts) > 1 and all(len(p) >= self.min_part_len and self.freq.get(p, 0) >= self.min_part_freq for p in parts):
            return parts
        return [word.lower()]

    def is_compound(self, word):
        return len(self.split(word)) > 1
