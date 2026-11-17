"""
Swedish compound splitter (Koehn & Knight 2003 frequency method) + density helpers.
Put this file in  src/compound.py  of the repo (next to utils.py).

Idea (Koehn & Knight): for a word, try every way to cut it into 2-3 parts (optionally with a
linking element -s-, -e-, -a-, -o- between parts). Each part must be a known word. Score a split
by the geometric mean of its parts' corpus frequencies. Accept the best split only if that score
is higher than the frequency of the whole word (otherwise the word is probably not a compound).
"""
import re, json, os
from collections import Counter

WORD_RE = re.compile(r"[a-zåäöéü]+")


def words_of(text):
    """Alphabetic lowercase tokens (digits dropped). Used for frequency list + density."""
    return WORD_RE.findall(text.lower())


def build_freq(texts, counter=None):
    c = counter if counter is not None else Counter()
    for t in texts:
        c.update(words_of(t))
    return c


def save_freq(counter, path):
    with open(path, "w") as f:
        json.dump(counter, f)


def load_freq(path):
    with open(path) as f:
        return Counter(json.load(f))


class CompoundSplitter:
    def __init__(self, freq, min_part_len=3, min_part_freq=30, min_word_len=7,
                 max_parts=3, linkers=("", "s", "e", "a", "o"), ratio=1.0):
        self.freq = freq
        self.min_part_len = min_part_len
        self.min_part_freq = min_part_freq
        self.min_word_len = min_word_len
        self.max_parts = max_parts
        self.linkers = linkers
        self.ratio = ratio          # split accepted if score > ratio * freq(whole word)
        self._cache = {}

    def _ok(self, p):
        return len(p) >= self.min_part_len and self.freq.get(p, 0) >= self.min_part_freq

    def _segmentations(self, w, n_left):
        """All ways to cut w into <= n_left known parts (linker allowed between parts)."""
        if self._ok(w):
            yield [w]
        if n_left <= 1:
            return
        for i in range(self.min_part_len, len(w) - self.min_part_len + 1):
            left = w[:i]
            if not self._ok(left):
                continue
            for lk in self.linkers:
                if lk and not w[i:].startswith(lk):
                    continue
                rest = w[i + len(lk):]
                if len(rest) < self.min_part_len:
                    continue
                for tail in self._segmentations(rest, n_left - 1):
                    yield [left] + tail

    def split(self, word):
        """Return list of parts ([word] if it is not split)."""
        w = word.lower()
        if w in self._cache:
            return self._cache[w]
        res = [w]
        if len(w) >= self.min_word_len:
            best, best_score = None, 0.0
            for seg in self._segmentations(w, self.max_parts):
                if len(seg) < 2:
                    continue
                prod = 1.0
                for p in seg:
                    prod *= self.freq[p]
                score = prod ** (1.0 / len(seg))
                if score > best_score:
                    best, best_score = seg, score
            if best is not None and best_score > self.ratio * self.freq.get(w, 0):
                res = best
        self._cache[w] = res
        return res

    def is_compound(self, word):
        return len(self.split(word)) > 1


def density(text, splitter):
    """(n_tokens, n_compound_tokens, density). density = compound tokens / all word tokens."""
    toks = words_of(text)
    if not toks:
        return 0, 0, 0.0
    n_c = sum(1 for t in toks if splitter.is_compound(t))
    return len(toks), n_c, n_c / len(toks)


def expand_decompound(tokens, splitter):
    """For M1 later: keep the compound AND add its parts."""
    out = []
    for t in tokens:
        out.append(t)
        parts = splitter.split(t)
        if len(parts) > 1:
            out.extend(parts)
    return out


# ---------------------------------------------------------------------------------------------
# CompoundPiece decompounder (Minixhofer, Pfeiffer & Vulic, EMNLP 2023): neural, multilingual,
# released as  benjamin/compoundpiece  on HuggingFace (MIT). Input: one word. Output: parts joined
# by "-" (e.g. "Hauswirtschaftslehre" -> "Haus-Wirtschaft-Lehre"). Same interface as CompoundSplitter,
# so density() / expand_decompound() work with either.
# ---------------------------------------------------------------------------------------------
class CompoundPieceSplitter:
    def __init__(self, model_name="benjamin/compoundpiece", cache_path=None,
                 min_word_len=7, device=None):
        self.model_name = model_name
        self.cache_path = cache_path
        self.min_word_len = min_word_len
        self.device = device
        self._tok = self._model = None
        self._cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)          # word -> raw model output

    # ---- model ----
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        dev = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(dev).eval()
        self.device = dev

    def _generate(self, words):
        import torch
        self._load()
        enc = self._tok(words, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self._model.generate(**enc, max_length=64, num_beams=1)   # greedy, fp32
        return self._tok.batch_decode(out, skip_special_tokens=True)

    def precompute(self, words, batch_size=128, save_every=50, verbose=True):
        """Run the model once for every word that is long enough and not cached yet."""
        todo = sorted({w.lower() for w in words if len(w) >= self.min_word_len} - set(self._cache),
                      key=len)                                   # sorted by length = less padding
        if verbose:
            print(f"CompoundPiece: {len(todo)} new words ({len(self._cache)} cached)")
        for bi, i in enumerate(range(0, len(todo), batch_size)):
            batch = todo[i:i + batch_size]
            for w, o in zip(batch, self._generate(batch)):
                self._cache[w] = o
            if self.cache_path and (bi + 1) % save_every == 0:
                self.save()
                if verbose:
                    print(f"  {min(i + batch_size, len(todo))}/{len(todo)}")
        self.save()

    def save(self):
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f)

    # ---- same interface as CompoundSplitter ----
    def split(self, word):
        w = word.lower()
        if len(w) < self.min_word_len:
            return [w]
        if w not in self._cache:
            self._cache[w] = self._generate([w])[0]
        parts = [p for p in self._cache[w].lower().strip().split("-") if p]
        return parts if len(parts) > 1 else [w]

    def is_compound(self, word):
        return len(self.split(word)) > 1
